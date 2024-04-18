from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from ferret import Benchmark
import torch #used for mps instead of cuda
import pandas as pd
import pickle

"""
1) Load tokenizer and data
2) Load model
3) Compute explanations for the dataset (takes 30 hours per model)

if we look at object `explanations` >> [0] >> `tokens`, we see that [CLS] and trailing [SEP] are already there and 
attribution values are computed for them, but they do not appear in the html render, which may cause confusion.

4) pickle explanations per modelXdataset
"""
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

df_dev = pd.read_csv("./esnli/data_original/esnli_test.csv") # CHANGE to 'test' or 'dev'
sep_token = tokenizer.sep_token
df_dev["Sentence1_Sentence2"] = df_dev["Sentence1"] + " " + sep_token + " " + df_dev["Sentence2"]
lab2int = {"neutral":0, "contradiction":1, "entailment":2}
dev = [(row["Sentence1_Sentence2"], lab2int[row["gold_label"]]) for index, row in df_dev.iterrows()]

# wandb models that we fine-tuned on the esnli dataset
runs_dict = {"db_01" : "ck4a29wu_glowing-puddle-8",
            "db_02" : "i08mmqx9_clone-federation-9",
            "db_03" : "w9avfbct_dark-rancor-11",
            "db_04" : "jhatt1ys_mythical-bothan-13",
            "db_05" : "5ursrh2w_sith-destroyer-14",
            "db_06" : "n91022vz_stellar-mountain-16",
            "db_07" : "lan4nxhz_snowy-meadow-17",
            "db_08" : "kk1t84xd_lilac-rain-19",
            "db_09" : "qyzxn0hv_graceful-dawn-21",
            "db_10" : "6fbnp2mf_dainty-pine-22"}

def get_dataset_explanations(dataset, model, tokenizer):
    """
    RUN once
    :param dataset: e.g. `dev`
    :param model: the pretrained huggingface model
    :param tokenizer: the pretrained huggingface tokenizer
    :return: list of ferret explanation objects, one for each instance in the dataset provided
    """
    bench = Benchmark(model=model, tokenizer=tokenizer)
    dataset_explanations = []
    for i,(X, y) in enumerate(dataset):
        instance_explanations = bench.explain(X, target=y)
        dataset_explanations.append(instance_explanations)
        print(i)
    return dataset_explanations

# pickle
for idx, name in runs_dict.items():
    # load model
    run_name = name
    #device = 'cpu' if torch.has_mps else 'cpu' #mps raises some error, so falling back on cpu
    device = 'cuda'
    model = DistilBertForSequenceClassification.from_pretrained("./runs/results/"+run_name,num_labels=3)
    model.to(device)
    model.eval()
    model.zero_grad()
    # compute explanations
    dev_dataset_explanations = get_dataset_explanations(dev, model, tokenizer)
    # pickle explanations
    with open("./esnli/explanations/test_dataset_explanations_"+idx+".pickle", "wb") as file: # CHANGE to 'test' or 'dev'
         pickle.dump(dev_dataset_explanations, file)
    print(idx + " " + name + " ---------------------> done")