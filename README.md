### Welcome
This is the repository of the long paper [The Role of Syntactic Span Preferences in Post-Hoc Explanation Disagreement](https://aclanthology.org/2024.lrec-main.1397) by Jonathan Kamp, Lisa Beinborn and Antske Fokkens. LREC-Coling 2024, 
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
Please cite our LREC-Coling version:
```bibtex
@inproceedings{kamp-etal-2024-role-syntactic,
    title = "The Role of Syntactic Span Preferences in Post-Hoc Explanation Disagreement",
    author = "Kamp, Jonathan  and
      Beinborn, Lisa  and
      Fokkens, Antske",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1397",
    pages = "16066--16078",
}
```
