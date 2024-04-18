The **e-SNLI dataset** is available at [the authors' repository](https://github.com/OanaMariaCamburu). Please cite
Camburu et al. (2018) if you are using their data.

Move the following 4 files to this directory (i.e. `data_original/`):
* `esnli_dev.csv`
* `esnli_test.csv`
* `esnli_train_1.csv`
* `esnli_train_2.csv`

A clarification that has been useful to us, reported in one of the readme files at https://github.com/OanaMariaCamburu:
* There are 2 splits for the train set due to the github sie restrictions, please simply merge them.
* Clarification on the two potentially confusing headers: Sentence1_marked_1: is the premise (Sentence2 for hypothesis)
were words between star (*) were highlighted by the annotators. The annotators had to click on every word individually
to highlight it. The punctuation has not been separated from the words, hence highlighting a word automatically included
any punctuation near it. Please use only this header to retrieve the highlighted words simply by retrieving the words
between stars without space between them, i.e., things like *w1* w2 *w3* only w1 and w3 were highlighted.
Please *ignore* the fields Sentence_Highlighted_ and retrieve the highlighted words from the Sentence_marked_ fields as
stated above.