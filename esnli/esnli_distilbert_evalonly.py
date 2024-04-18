import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset #from Huggingface
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import json
import wandb
import os
import pandas as pd
import gc

df_train_1 = pd.read_csv("./esnli/data_original/esnli_train_1.csv")
df_train_2 = pd.read_csv("./esnli/data_original/esnli_train_2.csv")
df_train = pd.concat([df_train_1,df_train_2])
df_dev = pd.read_csv("./esnli/data_original/esnli_dev.csv")
df_test = pd.read_csv("./esnli/data_original/esnli_test.csv")

## Some (6) examples in the training set miss a "Sentence2" -> remove
print(len(df_train[df_train['Sentence1'].notna()])) #549367 vvv should be equal
print(len(df_train[df_train['Sentence2'].notna()])) #549361 ^^^ should be equal
df_train = df_train[df_train['Sentence2'].notna()] # remove 6 lines where 'Sentence2' missing
print(len(df_train[df_train['Sentence1'].notna()])) #549361 vvv are now equal
print(len(df_train[df_train['Sentence2'].notna()])) #549361 ^^^ are now equal
##
print(len(df_dev[df_dev['Sentence1'].notna()])) #9842 vvv are already equal
print(len(df_dev[df_dev['Sentence2'].notna()])) #9842 ^^^ are already equal
##
print(len(df_test[df_test['Sentence1'].notna()])) #9824 vvv are already equal
print(len(df_test[df_test['Sentence2'].notna()])) #9824 ^^^ are already equal

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
sep_token = tokenizer.sep_token

df_train["Sentence1_Sentence2"] = df_train["Sentence1"] + " " + sep_token + " " + df_train["Sentence2"]
df_dev["Sentence1_Sentence2"] = df_dev["Sentence1"] + " " + sep_token + " " + df_dev["Sentence2"]
df_test["Sentence1_Sentence2"] = df_test["Sentence1"] + " " + sep_token + " " + df_test["Sentence2"]

lab2int = {"neutral":0, "contradiction":1, "entailment":2}

train = [(row["Sentence1_Sentence2"], lab2int[row["gold_label"]]) for index, row in df_train.iterrows()]
dev = [(row["Sentence1_Sentence2"], lab2int[row["gold_label"]]) for index, row in df_dev.iterrows()]
test = [(row["Sentence1_Sentence2"], lab2int[row["gold_label"]]) for index, row in df_test.iterrows()]

# free RAM from obsolete dataframes
del [[df_train_1,df_train_2,df_train,df_dev,df_test]]
gc.collect()

# max len for truncation
MAX_LEN=0
for sent, label in train+dev+test:
    n_tokens = len(sent.split())
    if n_tokens > MAX_LEN:
        MAX_LEN=n_tokens
print(MAX_LEN) #113

MAX_LEN += 20 #add some room for segmentation error as sent.split() tokenization is not perfect

MAX_LEN = min(512, MAX_LEN)

## Convert to Huggingface input format: raw datasets (lists) --> Dataset objects --> tokenized --> tensored
dataset = {"train":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in train]),
           "dev":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in dev]),
           "test":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in test])}

# load tokenizer + start EVAL part: load previously trained model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

run_name = "kk1t84xd_lilac-rain-19"
model = DistilBertForSequenceClassification.from_pretrained("./runs/results/"+run_name,num_labels=3)

try:
    device = 'mps' if torch.has_mps else 'cpu' #m1 mac
except AttributeError:
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #linux
print(device)

model = model.to(device)
model.eval()
model.zero_grad()

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN) # setting padding to `longest` did not work.
    # Looked for alternative solution to speed up training. The max_length argument controls the length
    # of the padding and truncation. It can be an integer or None, in which case it will default to the maximum length the
    # model can accept. If the model has no specific maximum input length, truncation or padding to max_length is deactivated.

dataset_tokenized = {"train":dataset["train"].map(tokenize_function, batched=True),
                     "dev":dataset["dev"].map(tokenize_function, batched=True),
                     "test":dataset["test"].map(tokenize_function, batched=True)}

dataset_tensored = {"train":dataset_tokenized["train"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device),
                    "dev":dataset_tokenized["dev"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device),
                    "test":dataset_tokenized["test"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device)}

# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro") #average=None for per-class scores
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# to avoid attribution error when running on cuda; default is True and does not give problems on mps/m1/mac
pin_memory = False if device=="cuda" else True

training_args = TrainingArguments(
    output_dir = './runs/results',
    num_train_epochs = 10,
    per_device_train_batch_size = 32, #default 8; roberta paper: 16/32
    #per_device_eval_batch_size = 8, #default 8
    #fp16 = True, # WORKS ON CUDA, NOT ON MPS. speed-up, especially for small batch sizes: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    #gradient_accumulation_steps = 16, #default 1; good for memory, but slows down training a bit: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    #gradient_checkpointing = True #default False; slows down training but frees memory; especially for big models https://huggingface.co/docs/transformers/v4.18.0/en/performance
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    #metric_for_best_model="loss", #default "loss"
    load_best_model_at_end = True, #saves model at best epoch --> needed for wandb
    warmup_steps = 189, #default 0; 1 epoch =>[837 w batchsize = 8; 837/2 = 419 w batchsize = 16]. warmupsteps in roberta paper: 6% of total steps. e.g. 10epc,bsz16: (419*10)*6/100 = 251
    weight_decay = 0.01, #default 0
    learning_rate = 5e-6, #float(sys.argv[1]), #default 5e-5; 1e-5 is used in roberta paper; 1e-5 -> 1e-6 interval works best based on online forums
    logging_strategy = "steps",
    #logging_steps = 8, #default 500
    logging_dir = './runs/logs', #logging_dir (str, optional) — TensorBoard log directory. Will default to *output_dir/runs/CURRENT_DATETIME_HOSTNAME*.
    #dataloader_num_workers = 8, #default 0
    report_to = "wandb",
    #run_name = 'roberta-classification',
    dataloader_pin_memory = pin_memory
)

trainer = Trainer(
    model = model,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = dataset_tensored["train"],
    eval_dataset = dataset_tensored["test"],
)

trainer.evaluate()
