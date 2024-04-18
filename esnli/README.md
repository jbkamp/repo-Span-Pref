### Welcome
This is the repository of the long paper [The Role of Syntactic Span Preferences in Post-Hoc Explanation Disagreement](https://arxiv.org/abs/2403.19424) by Jonathan Kamp, Lisa Beinborn and Antske Fokkens. LREC-Coling 2024, 
Turin.

### Description of the code files in `esnli/` directory
Fine-tuning the models:
* `esnli_distilbert.py`

Results inspection:
* `esnli_distilbert_evalonly.py`

Model selection:
* `create_explanation_pickles_distilbert.py` (stored in `esnli/explanations/`)
* `classifier_model_selection.py`

Analysis script:
* `analysis.py`

### Data
* Original dataset description in `esnli/data_original/`
* Parsed dataset in `esnli/parses/`

### Dependencies
* `Python 3.9.13`
* `torch 1.14.0.dev20221104`
* `transformers 4.24.0`
* `ferret 0.4.1`
* `flair 0.12.2`

### Cite us :)
Please cite our LREC-Coling paper (soon available) or alternatively the arXiv version.