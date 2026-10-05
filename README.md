# WildSeek: Evaluating Language Models for Information-Seeking

Code and analysis for **WildSeek** (EMNLP 2026), a manually annotated dataset of 3,077 real-world information-seeking queries and an evaluation framework for the reliability, fairness and safety of LLM responses.

## 🔎 Explore the dataset and findings online

**👉 https://tceron.github.io/wildseek-eval/**

## Resources

[Paper](https://arxiv.org/abs/2608.30683) · [Data (OSF)](https://osf.io/hxvj2) · [Classifiers (Hugging Face)](https://huggingface.co/tceron)

## Data

All model responses generated in this study are available on OSF:

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
- `credibility.ipynb`, `safety_analysis_results.ipynb`, `topic_analysis.ipynb`, `visualize_results.ipynb` — analysis and visualization notebooks
- `docs/` — source of the project website (see below)

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
