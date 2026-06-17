#!/usr/bin/env python3
"""
Media Bias/Fact Check scraper.

Searches mediabiasfactcheck.com for news outlets and extracts the
Detailed Report section (Bias Rating, Factual Reporting, Country, etc.).

Usage:
    python scrape_mbfc.py "New York Times" "Fox News" "BBC"
    python scrape_mbfc.py --input outlets.txt --output results.json
    python scrape_mbfc.py --input outlets.txt  # saves to media_bias_results.json

Requirements:
    pip install cloudscraper beautifulsoup4
"""

import argparse
import json
import os
import re
import time
from urllib.parse import quote_plus

import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://mediabiasfactcheck.com"

_COMMON_TLDS = {
    "com", "org", "net", "gov", "edu", "io", "info", "biz", "int", "mil",
    "tv", "co", "uk", "au", "ca", "de", "fr", "nl", "eu", "us", "ru", "cn",
    "jp", "br", "in", "mx", "es", "it", "se", "no", "dk", "fi", "ch", "at",
    "be", "pl", "cz", "hu", "ro", "bg", "gr", "pt", "il", "za", "nz", "ar",
}
_SKIP_SUBDOMAINS = {"www", "en", "m", "mobile", "amp", "news", "media", "web", "cdn"}

# MBFC pages that are definitely not outlet pages
_MBFC_NON_OUTLET_SLUGS = {
    "gift-memberships", "support-media-bias-fact-check", "filtered-search",
    "mbfcs-data-api", "login", "about", "methodology", "donate", "contact",
    "privacy-policy", "terms-of-service", "advertise", "newsletter",
    "left", "right", "center", "least-biased", "pro-science", "conspiracy",
    "satire", "fake-news", "questionable-sources",
}

FIELDS = [
    "Bias Rating",
    "Factual Reporting",
    "Country",
    "MBFC’s Country Freedom Rating",
    "Media Type",
    "Traffic/Popularity",
    "MBFC Credibility Rating",
]

# Some pages use "Rank" instead of "Rating" — treat both as the canonical key.
_FIELD_ALIASES = ["MBFC’s Country Freedom Rank"]
_ALL_LABELS = FIELDS + _FIELD_ALIASES


def _normalize_quotes(s: str) -> str:
    """Replace curly/typographic apostrophes with straight ones for matching."""
    return s.replace("‘", "’").replace("’", "’")


def _canonical_field(label: str) -> str | None:
    """Return the canonical FIELDS key for a label, handling aliases."""
    norm = _normalize_quotes(label).lower()
    for f in FIELDS:
        if norm == _normalize_quotes(f).lower():
            return f
    # Aliases
    if norm == _normalize_quotes("MBFC’s Country Freedom Rank").lower():
        return "MBFC’s Country Freedom Rating"
    return None


# Splits paragraph text on any field label (canonical + aliases).
# Quote-normalized so curly apostrophes match.
_FIELD_SPLIT_RE = re.compile(
    r"(" + "|".join(re.escape(_normalize_quotes(f)) for f in _ALL_LABELS) + r")\s*:\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _is_domain(name: str) -> bool:
    """Return True if name looks like a web domain rather than a publication name."""
    return "." in name and " " not in name


def domain_to_outlet_name(domain: str) -> str:
    """Extract a plausible outlet name from a domain for MBFC lookup.

    en.wikipedia.org      → wikipedia
    www.nytimes.com       → nytimes
    pmc.ncbi.nlm.nih.gov  → nih
    foxnews.com           → foxnews
    """
    domain = re.sub(r"^https?://", "", domain).split("/")[0].split(":")[0].lower()
    parts = domain.split(".")
    # Strip TLD suffixes from the right
    while len(parts) > 1 and parts[-1] in _COMMON_TLDS:
        parts.pop()
    # Strip non-informative prefixes from the left
    while len(parts) > 1 and parts[0] in _SKIP_SUBDOMAINS:
        parts.pop(0)
    return parts[-1] if parts else domain


def _is_outlet_url(url: str) -> bool:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug not in _MBFC_NON_OUTLET_SLUGS


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Convert a human-readable outlet name to a URL slug."""
    slug = name.lower()
    # Remove common leading articles that are often dropped in slugs
    slug = re.sub(r"^the\s+", "", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


def candidate_urls(outlet_name: str) -> list[str]:
    """Generate plausible direct URLs to try before falling back to search."""
    candidates = []
    if _is_domain(outlet_name):
        variants = [domain_to_outlet_name(outlet_name)]
    else:
        variants = [outlet_name, re.sub(r"^The\s+", "", outlet_name, flags=re.I)]
    for variant in variants:
        slug = slugify(variant)
        candidates.append(f"{BASE_URL}/{slug}/")
    return list(dict.fromkeys(candidates))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_outlet(sc, outlet_name: str) -> list[str]:
    """Query the WordPress search and return result page URLs."""
    url = f"{BASE_URL}/?s={quote_plus(outlet_name)}"
    try:
        resp = sc.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  Search request failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls: list[str] = []

    # WordPress standard search results live inside <article> elements
    for article in soup.find_all("article"):
        for a in article.find_all("a", href=True):
            href: str = a["href"]
            if re.match(r"https://mediabiasfactcheck\.com/[^/?#]+/?$", href):
                urls.append(href)
                break  # first link per article is the title link

    # Fallback: heading links with entry-title / post-title classes
    if not urls:
        for el in soup.find_all(class_=re.compile(r"entry-title|post-title")):
            a = el.find("a", href=True)
            if a:
                href = a["href"]
                if re.match(r"https://mediabiasfactcheck\.com/[^/?#]+/?$", href):
                    urls.append(href)

    # Last resort: any MBFC link that looks like a post page
    if not urls:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.match(r"https://mediabiasfactcheck\.com/[^/?#]+/?$", href):
                urls.append(href)

    # Deduplicate, preserve order, and filter out known non-outlet pages
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen and _is_outlet_url(u):
            seen.add(u)
            unique.append(u)
    return unique


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_detailed_report(html: str) -> dict[str, str]:
    """
    Parse the 'Detailed Report' section from a MBFC outlet page.

    Actual HTML structure (verified):
        <h3>...<span>Detailed Report</span></h3>
        <p>
          <span>Bias Rating: <strong>LEFT-CENTER (-4.1)</strong>...
                Factual Reporting: <strong>HIGH (1.4)</strong>...
          </span>
          <span>MBFC Credibility Rating: <strong>HIGH CREDIBILITY</strong></span>
        </p>

    Strategy A — find the <p> after the header, collapse to one line, split on
    field names. Cleanest when the DOM structure holds.

    Strategy B — paired consecutive lines: get_text(separator="\\n") puts each
    label ("Bias Rating: ") on one line and the value on the very next line.
    """
    soup = BeautifulSoup(html, "html.parser")
    data: dict[str, str] = {}

    # --- Strategy A: <p> after "Detailed Report" header --------------------
    header = None
    for tag in soup.find_all(["h3", "h4", "h2", "h5"]):
        if re.search(r"Detailed\s+Report", tag.get_text(), re.I):
            header = tag
            break

    if header:
        p = header.find_next_sibling("p")
        if p:
            # Collapse the paragraph to a single line, normalize quotes, then
            # split on field labels to get interleaved [label, value, …] tokens.
            flat = _normalize_quotes(re.sub(r"\s+", " ", p.get_text(separator=" ")).strip())
            parts = _FIELD_SPLIT_RE.split(flat)
            # parts[0] = text before first label (discard); then pairs (label, value)
            for i in range(1, len(parts) - 1, 2):
                canon = _canonical_field(parts[i].strip())
                value = re.sub(r"\s+", " ", parts[i + 1]).strip()
                if canon:
                    data[canon] = value

    if len(data) >= 4:
        return data

    # --- Strategy B: paired consecutive non-empty lines --------------------
    # get_text(separator="\n") emits each element's text on its own line, so
    # "Bias Rating: " lands on line N and "LEFT-CENTER (-4.1)" lands on line N+1.
    lines = [
        _normalize_quotes(ln.strip())
        for ln in soup.get_text(separator="\n").split("\n")
        if ln.strip()
    ]
    label_re = [
        (re.compile(rf"^{re.escape(_normalize_quotes(f))}\s*:(.*)$", re.I), f)
        for f in _ALL_LABELS
    ]

    for i, line in enumerate(lines):
        for pat, raw_label in label_re:
            m = pat.match(line)
            if m:
                canon = _canonical_field(raw_label)
                if not canon:
                    continue
                inline = m.group(1).strip()
                if inline:
                    data[canon] = inline           # "Field: VALUE" on one line
                elif i + 1 < len(lines):
                    data[canon] = lines[i + 1]     # value on the next line

    return data


# ---------------------------------------------------------------------------
# Per-outlet orchestration
# ---------------------------------------------------------------------------

def scrape_outlet(sc, outlet_name: str) -> dict:
    """Return extracted data dict for one outlet, or a dict with 'error'."""

    def _try_url(url: str) -> dict | None:
        try:
            resp = sc.get(url, timeout=20, allow_redirects=True)
        except Exception as exc:
            print(f"    Fetch failed ({url}): {exc}")
            return None
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {url}")
            return None
        data = extract_detailed_report(resp.text)
        if len(data) >= 3:
            data["source_url"] = resp.url
            return data
        return None

    # 1. Try plausible direct URLs (avoids a search round-trip)
    for url in candidate_urls(outlet_name):
        print(f"  Trying direct URL: {url}")
        result = _try_url(url)
        if result:
            return result
        time.sleep(0.8)

    # 2. Fall back to search
    print("  Falling back to search...")
    search_query = domain_to_outlet_name(outlet_name) if _is_domain(outlet_name) else outlet_name
    candidates = search_outlet(sc, search_query)
    if not candidates:
        return {"error": "no search results found"}

    print(f"  Search returned {len(candidates)} candidate(s)")
    for url in candidates[:5]:
        print(f"  Trying: {url}")
        result = _try_url(url)
        if result:
            return result
        time.sleep(1.0)

    return {"error": "could not extract data from any candidate URL", "candidates_tried": candidates[:5]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Media Bias/Fact Check Detailed Reports for news outlets."
    )
    parser.add_argument(
        "outlets",
        nargs="*",
        metavar="OUTLET",
        help="One or more news outlet names (quoted if they contain spaces)",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Text file with one outlet name per line",
    )
    parser.add_argument(
        "--output", "-o",
        default="media_bias_results.json",
        metavar="FILE",
        help="Output JSON file (default: media_bias_results.json)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Delay in seconds between outlets (default: 2.0)",
    )
    args = parser.parse_args()

    # Collect outlet names
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            outlets = [line.strip() for line in fh if line.strip()]
    elif args.outlets:
        outlets = args.outlets
    else:
        outlets = ["New York Times", "Fox News", "BBC", "Reuters", "Washington Post"]
        print("No outlets specified. Using defaults:", outlets)
        
    print(f"\nScraping {len(outlets)} outlet(s) ...\n")

    results: dict[str, dict] = {}
    sc = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    for i, outlet in enumerate(outlets):
        print(f"[{i + 1}/{len(outlets)}] {outlet}")
        data = scrape_outlet(sc, outlet)
        results[outlet] = data

        if "error" not in data:
            display = {k: v for k, v in data.items() if k != "source_url"}
            print(f"  OK → {display}")
        else:
            print(f"  FAILED → {data['error']}")

        with open(args.output, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"outlet": outlet, "data": data}, ensure_ascii=False) + "\n")

        # Rate-limit between outlets (not after the last one)
        if i < len(outlets) - 1:
            time.sleep(args.delay)

    success = sum(1 for v in results.values() if "error" not in v)
    print(f"\nDone. {success}/{len(outlets)} outlets scraped successfully.")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
