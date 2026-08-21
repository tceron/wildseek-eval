# wildseek-eval

Code and analysis for the WildSeek evaluation study.

## Data

All data used in this study is available on OSF:

**https://osf.io/hxvj2**

Download the files from that project page before running any of the scripts or notebooks below — download and place them under the appropriate folder in this repo (e.g. `data_prompts/`), matching the paths expected by the scripts.

## Repository structure

- `factuality/` — factuality-related evaluation code
- `query_classification/` — query classification models and scripts (see `query_classification/README.md` for details on running the classifier)
- `create_samples_annotations.py` — builds annotation samples
- `high_stakes_prompts.py` / `safety_prompts.py` — prompt sets used in the safety/high-stakes evaluation
- `llm_safety_evaluator.py` — runs the safety evaluation
- `process_data.py` / `utils.py` — shared data processing utilities
- `prompt_claude.py` / `prompt_gpt.py` / `prompt_gemini.py` / `prompt_hf_models.py` / `prompt_models.py` — scripts for querying different model providers
- `scrape_mbfc.py` — scrapes source credibility ratings (Media Bias/Fact Check)
- `analyze_search_results.ipynb`, `credibility.ipynb`, `safety_analysis_results.ipynb`, `safety_eval.ipynb`, `topic_analysis.ipynb`, `visualize_results.ipynb` — analysis and visualization notebooks

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{ceron2026wildseek,
  title     = {WildSEEK: Evaluating Language Models for Information-Seeking},
  author    = {Ceron, Tanise and Baumann, Joachim and Bassignana, Elisa and Cabuk, Berat and Hovy, Dirk and Nozza, Debora},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
