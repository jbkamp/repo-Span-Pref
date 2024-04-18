from transformers import DistilBertTokenizer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import random
import pickle
from itertools import product
from IPython.core.display import HTML
import re
import webbrowser
from collections import Counter
from flair.data import Sentence
from flair.models import SequenceTagger
import string
import spacy
from spacy.tokens import Doc
import benepar
import math
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from itertools import combinations

"""
Load tokenizer and pickled explanations
"""
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
with open("./esnli/explanations/test_explanations/test_dataset_explanations_db_08.pickle", "rb") as file:
    test_dataset_explanations = pickle.load(file)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
"""
Base functions to calculate agreement based on k
"""
def create_topk_mask(attribs, tokens, topk=-1, ranked=False, rm_special_tokens=True,
                     dynamic=False, dyn_threshold="mean", peaks=True):
    """
    :param attribs:             list of token attributions
    :param tokens:              list of tokens where the attribs were computed for
    :param topk:                defaults to -1. If dynamic==False, topk should be set to a positive integer
    :param ranked:              boolean
    :param rm_special_tokens:   defaults to True
    :param dynamic:             defaults to False; set to True if mask is to be computed for dyn. top-k based on loc max
    :return:                    boolean mask based on `attribs` list, with 1s indexed at topk highest attribution values
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3) returns [0, 1, 1, 1]
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3,rm_special_tokens=False)
        --> returns [0, 0, 1, 0, 1, 1] (or [0, 0, 0, 1, 1, 1], because ties are solved by random choice)
    """
    assert len(attribs) == len(tokens)
    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0

    if rm_special_tokens:
        attribs = [a for t,a in zip(tokens, attribs) if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
    if dynamic:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
        attribs2, local_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs,
                                                                       dyn_threshold=dyn_threshold,
                                                                       plot=False,
                                                                       peaks=peaks)
        dynamic_topk_mask = [0 for _ in attribs2]
        for loc_max_i in local_maxima_indices:
            dynamic_topk_mask[loc_max_i] = 1
        return dynamic_topk_mask

    assign_indices = list(enumerate(attribs))
    assign_indices.sort(key=lambda tup: (tup[1], random.random())) #if tie -> randomize
    sorted_indices = [i for i,a in assign_indices]
    topk_sorted_indices = sorted_indices[-topk:] #e.g. top-2 of [1,2,3,0] is [3,0]
    topk_mask = [0 for a in attribs] #initialize 0s mask
    if ranked:
        for i, rank in zip(topk_sorted_indices,range(len(topk_sorted_indices),0,-1)):
            topk_mask[i] = rank #assign rank at topk indices; outside the topk remains 0
    else:
        for i in topk_sorted_indices:
            topk_mask[i] = 1 #assign 1 at topk indices; outside the topk remains 0
    return topk_mask

def get_token_agr_score(tup, class_1_agreement=True):
    """
    :param tup: tuple of top-k mask for a specific token; unranked or ranked does not matter
    :param class_1_agreement: only calculate for class 1
    :return: agreement ratio for a token
        e.g. get_token_agr_score((0,0,0,1,0))                           -> 0.8
        e.g. get_token_agr_score((0,0,0,1,0), class_1_agreement=True)   -> 0.2
    """
    if sum(tup) == 0: # e.g. vertical (0,0,0,0,0)
        agr = np.nan  # we assign nan and not 100% agreement if all methods assign 0
    else:
        d = {rank:0 for rank in set(tup)} # initialize, e.g. {0:0, 1:0, 2:0, 3:0} or {0:0, 1:0}
        for rank in tup: #abs freq of ranks; e.g. {0:2, 1:1, 2:1, 3:1}
            d[rank] += 1
        for rank, freq in d.items(): #relative freq of ranks; e.g. {0:0.4, 1:0.2, 2:0.2, 3:0.2}
            d[rank] = freq/len(tup)
        if class_1_agreement:
            agr = max([rel_freq for (rank, rel_freq) in d.items()
                       if rank != 0])  # agreement is the max of relative freqs for ranks != 0; e.g. 0.2
        else:
            agr = max(d.values())  # agreement is the max of relative freqs; e.g. 0.4

    return agr

def get_instance_agr_score(explanations, topk=-1, ranked=False, rm_special_tokens=True, class_1_agreement=True,
                           leaveout_idx_list=[], human_aggreg_values=[], dynamic=False,
                           subset_dynamic_picks_lower_bound=None, subset_dynamic_picks_upper_bound=None,
                           topk_argmax_dyn_fix=None, dyn_threshold="mean", peaks=True):
    """
    :param explanations:        ferret output object
    :param topk:                top-k parameter
    :param ranked:
    :param rm_special_tokens:
    :param class_1_agreement:
    :param leaveout_idx_list:   list containing 0 or + integers in the range [0,5] to excl. from agreement computation
        0 ~ Partition SHAP
        1 ~ LIME
        2 ~ Gradient
        3 ~ Gradient (xInput)
        4 ~ Integrated Gradient
        5 ~ Integrated Gradient (xInput)
    :param human_aggreg_values: list containing human aggregation values (obtained from annotated gold highlights)
    :param topk_argmax_dyn_fix: integer to be the fixed k in an argmax(dynk, fixk); defaults to None
    :return:                    mean agreement score over tokens for the instance, based on top-k and ignoring
                                nan values. If topk provided is <1 or >n_tokens in the sentence,
                                return instance_agr_score=nan
    """
    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
    # leave one or more attribution methods out depending on parameter
    explanations = [x for n,x in enumerate(explanations) if n not in leaveout_idx_list]
    # compute list of method-wise masks, e.g. [ [0,0,0,1,1], [0,0,1,1,1], [0,0,1,0,1] ]

    if topk_argmax_dyn_fix:
        assert topk == -1
        assert dynamic is True
        list_of_masks = []
        for x in explanations:
            topk_mask_fixed = create_topk_mask(attribs=list(x.scores),
                                               tokens=list(x.tokens),
                                               topk=topk_argmax_dyn_fix,
                                               ranked=ranked,
                                               rm_special_tokens=rm_special_tokens,
                                               dynamic=False,
                                               dyn_threshold=dyn_threshold,
                                               peaks=peaks)
            topk_mask_dynamic = create_topk_mask(attribs=list(x.scores),
                                                 tokens=list(x.tokens),
                                                 topk=-1,
                                                 ranked=ranked,
                                                 rm_special_tokens=rm_special_tokens,
                                                 dynamic=True,
                                                 dyn_threshold=dyn_threshold,
                                                 peaks=peaks)
            if sum(topk_mask_fixed) > sum(topk_mask_dynamic):
                list_of_masks.append(topk_mask_fixed)
            elif sum(topk_mask_fixed) < sum(topk_mask_dynamic):
                list_of_masks.append(topk_mask_dynamic)
            else:
                random_choice = random.choice([topk_mask_dynamic,topk_mask_fixed])
                list_of_masks.append(random_choice)
            # if sum(topk_mask_fixed) > sum(topk_mask_dynamic):
            #     list_of_masks.append(topk_mask_dynamic)
            # elif sum(topk_mask_fixed) < sum(topk_mask_dynamic):
            #     list_of_masks.append(topk_mask_fixed)
            # else:
            #     random_choice = random.choice([topk_mask_dynamic,topk_mask_fixed])
            #     list_of_masks.append(random_choice)
    else:
        list_of_masks = [create_topk_mask(attribs=list(x.scores), #TODO rob tokenizer [s for s,t in zip(x.scores,x.tokens) if t != 'Ġ']
                                          tokens=list(x.tokens), #TODO rob tokenizer [t for s,t in zip(x.scores,x.tokens) if t != 'Ġ']
                                          topk=topk,
                                          ranked=ranked,
                                          rm_special_tokens=rm_special_tokens,
                                          dynamic=dynamic,
                                          dyn_threshold=dyn_threshold,
                                          peaks=peaks)
                         for x in explanations]
    ####################################################################################################################
    # ADD HUMAN TOP K MASK AS A ROW TO `list_of_masks` if it is added as an argument
    if len(human_aggreg_values) > 0:
        assert rm_special_tokens == True  # human gt does (can)not have special tokens
        assert ranked == False  # did not test for ranked
        assert len(human_aggreg_values) == len(list_of_masks[0])

        if topk_argmax_dyn_fix:
            human_assign_indices = list(enumerate(human_aggreg_values))
            human_assign_indices.sort(key=lambda tup: (tup[1], random.random()))  # if tie -> randomize
            human_sorted_indices = [i for i, a in human_assign_indices]
            human_topk_sorted_indices = human_sorted_indices[-topk_argmax_dyn_fix:]  # e.g. top-2 of [1,2,3,0] is [3,0]
            human_topk_mask_fixed = [0 for _ in human_aggreg_values]  # initialize 0s mask
            for i in human_topk_sorted_indices:
                human_topk_mask_fixed[i] = 1  # assign 1 at topk indices; outside the topk remains 0

            human_topk_mask_dynamic = create_topk_mask(attribs=human_aggreg_values,
                                                       tokens=["dummy" for _ in human_aggreg_values],
                                                       topk=-1,
                                                       ranked=ranked,
                                                       rm_special_tokens=rm_special_tokens,
                                                       dynamic=True,
                                                       dyn_threshold=dyn_threshold,
                                                       peaks=peaks)

            if sum(human_topk_mask_fixed) > sum(human_topk_mask_dynamic):
                list_of_masks.append(human_topk_mask_fixed)
            elif sum(human_topk_mask_fixed) < sum(human_topk_mask_dynamic):
                list_of_masks.append(human_topk_mask_dynamic)
            else:
                human_random_choice = random.choice([human_topk_mask_dynamic, human_topk_mask_fixed])
                list_of_masks.append(human_random_choice)
            # if sum(human_topk_mask_fixed) > sum(human_topk_mask_dynamic):
            #     list_of_masks.append(human_topk_mask_dynamic)
            # elif sum(human_topk_mask_fixed) < sum(human_topk_mask_dynamic):
            #     list_of_masks.append(human_topk_mask_fixed)
            # else:
            #     human_random_choice = random.choice([human_topk_mask_dynamic, human_topk_mask_fixed])
            #     list_of_masks.append(human_random_choice)

        else:
            if not dynamic:
                human_assign_indices = list(enumerate(human_aggreg_values))
                human_assign_indices.sort(key=lambda tup: (tup[1], random.random()))  # if tie -> randomize
                human_sorted_indices = [i for i, a in human_assign_indices]
                human_topk_sorted_indices = human_sorted_indices[-topk:]  # e.g. top-2 of [1,2,3,0] is [3,0]
                human_topk_mask = [0 for _ in human_aggreg_values]  # initialize 0s mask
                for i in human_topk_sorted_indices:
                    human_topk_mask[i] = 1  # assign 1 at topk indices; outside the topk remains 0
                list_of_masks.append(human_topk_mask)
            else:
                human_topk_mask = create_topk_mask(attribs=human_aggreg_values,
                                                   tokens=["dummy" for _ in human_aggreg_values],
                                                   topk=topk,
                                                   ranked=ranked,
                                                   rm_special_tokens=rm_special_tokens,
                                                   dynamic=dynamic,
                                                   dyn_threshold=dyn_threshold,
                                                   peaks=peaks)
                list_of_masks.append(human_topk_mask)
    ####################################################################################################################
    if not subset_dynamic_picks_lower_bound and not subset_dynamic_picks_upper_bound: #regular case
        tupled_tokenwise = list(zip(*list_of_masks))
        agreements_tokenwise = [get_token_agr_score(tup,class_1_agreement=class_1_agreement)
                                for tup in tupled_tokenwise]
        if rm_special_tokens:
            tokens = [t for t in explanations[0].tokens if t not in {"[CLS]","[SEP]","<s>","</s>"}]
        else:
            tokens = list(explanations[0].tokens)

        return (np.nanmean(agreements_tokenwise), #ignore nans
                list_of_masks,
                agreements_tokenwise,
                tokens)

    else:
        assert len(list_of_masks) == 2  # only possible for pairwise agreement computation
        lb = subset_dynamic_picks_lower_bound
        ub = subset_dynamic_picks_upper_bound
        dyn_k_0 = sum(list_of_masks[0])
        dyn_k_1 = sum(list_of_masks[1])

        if lb and ub:
            if (dyn_k_0 > lb and dyn_k_0 < ub) and (dyn_k_1 > lb and dyn_k_1 < ub):
                tupled_tokenwise = list(zip(*list_of_masks))
                agreements_tokenwise = [get_token_agr_score(tup, class_1_agreement=class_1_agreement)
                                        for tup in tupled_tokenwise]
                if rm_special_tokens:
                    tokens = [t for t in explanations[0].tokens if t not in {"[CLS]", "[SEP]", "<s>", "</s>"}]
                else:
                    tokens = list(explanations[0].tokens)

                return (np.nanmean(agreements_tokenwise),  # ignore nans
                        list_of_masks,
                        agreements_tokenwise,
                        tokens)
            else:
                return (np.nan,
                        np.nan,
                        np.nan,
                        np.nan)
        elif lb and not ub:
            if dyn_k_0 > lb and dyn_k_1 > lb:
                tupled_tokenwise = list(zip(*list_of_masks))
                agreements_tokenwise = [get_token_agr_score(tup, class_1_agreement=class_1_agreement)
                                        for tup in tupled_tokenwise]
                if rm_special_tokens:
                    tokens = [t for t in explanations[0].tokens if t not in {"[CLS]", "[SEP]", "<s>", "</s>"}]
                else:
                    tokens = list(explanations[0].tokens)

                return (np.nanmean(agreements_tokenwise),  # ignore nans
                        list_of_masks,
                        agreements_tokenwise,
                        tokens)
            else:
                return (np.nan,
                        np.nan,
                        np.nan,
                        np.nan)
        elif ub and not lb:
            if dyn_k_0 < ub and dyn_k_1 < ub:
                tupled_tokenwise = list(zip(*list_of_masks))
                agreements_tokenwise = [get_token_agr_score(tup, class_1_agreement=class_1_agreement)
                                        for tup in tupled_tokenwise]
                if rm_special_tokens:
                    tokens = [t for t in explanations[0].tokens if t not in {"[CLS]", "[SEP]", "<s>", "</s>"}]
                else:
                    tokens = list(explanations[0].tokens)

                return (np.nanmean(agreements_tokenwise),  # ignore nans
                        list_of_masks,
                        agreements_tokenwise,
                        tokens)
            else:
                return (np.nan,
                        np.nan,
                        np.nan,
                        np.nan)

def get_dataset_agr_scores(dataset_explanations, topk=-1, ranked=False, rm_special_tokens=True,
                           class_1_agreement=True, leaveout_idx_list=[], dataset_human_aggreg_values=[], dynamic=False,
                           subset_dynamic_picks_lower_bound=None, subset_dynamic_picks_upper_bound=None,
                           topk_argmax_dyn_fix=None, dyn_threshold="mean", peaks=True):
    """
    :param dataset_explanations:
    :param topk:
    :param ranked:
    :param rm_special_tokens:
    :param class_1_agreement:
    :param leaveout_idx_list:           list containing 0 or + integers in the range [0,5] to excl. from agreement comp.
        0 ~ Partition SHAP
        1 ~ LIME
        2 ~ Gradient
        3 ~ Gradient (xInput)
        4 ~ Integrated Gradient
        5 ~ Integrated Gradient (xInput)
    :param dataset_human_aggreg_values:
    :param dynamic:
    :return:                            list of agreement scores for each instance of the data for a specif. top-k value
    """
    if dynamic == False:  # regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1  # contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)

    if len(dataset_human_aggreg_values) > 0:
        assert rm_special_tokens == True #human gt does (can)not have special tokens
        assert ranked == False #did not test for ranked
        assert len(dataset_explanations) == len(dataset_human_aggreg_values)

    dataset_instance_agr_scores = []
    for i, instance_explanations in enumerate(dataset_explanations):
        if len(dataset_human_aggreg_values) > 0:
            instance_human_aggreg_values = dataset_human_aggreg_values[i]
        else:
            instance_human_aggreg_values = []

        inst_agr_score = get_instance_agr_score(explanations=instance_explanations,
                                                topk=topk,
                                                ranked=ranked,
                                                rm_special_tokens=rm_special_tokens,
                                                class_1_agreement=class_1_agreement,
                                                leaveout_idx_list=leaveout_idx_list,
                                                human_aggreg_values=instance_human_aggreg_values,
                                                dynamic=dynamic,
                                                subset_dynamic_picks_lower_bound=subset_dynamic_picks_lower_bound,
                                                subset_dynamic_picks_upper_bound=subset_dynamic_picks_upper_bound,
                                                topk_argmax_dyn_fix=topk_argmax_dyn_fix,
                                                dyn_threshold=dyn_threshold,
                                                peaks=peaks
                                                )
        if inst_agr_score == "skipped-due-to-tokenization-mismatch": #TODO rob tokenizer
            print(inst_agr_score) #TODO rob tokenizer
            continue #TODO rob tokenizer
        else:
            mean_instance_agr_score, _, __, ___ = inst_agr_score

        dataset_instance_agr_scores.append(mean_instance_agr_score)

    return dataset_instance_agr_scores

"""
Agreement XAI models - aggregated human annotations
"""
df_test = pd.read_csv("./esnli/data_original/esnli_test.csv")
df_test = df_test.drop(columns=['Sentence1_Highlighted_1',
                              'Sentence1_Highlighted_2',
                              'Sentence1_Highlighted_3',
                              'Sentence2_Highlighted_1',
                              'Sentence2_Highlighted_2',
                              'Sentence2_Highlighted_3',
                              'Explanation_1',
                              'Explanation_2',
                              'Explanation_3'])

def spanmask(sentence):
    """
    Tokenizer is important here! If using the roberta explanations, use roberta tokenizer; if distilbert, then
    distilbert tokenizer. Check the first lines of this script.
    :param sentence: e.g.                               "hello this *is* a *very* *8portant* meeting *alright?*"
    :return: spanmask of tokens between asterisks, e.g. [0,    0,    1,  0,1,     1,         0,      1       0 ]
        !! punctuation is always given 0, since Camburu's annotators could not differentiate between punct and non-p
    """
    tokenized_sentence = tokenizer.tokenize(sentence)

    tmp_mask = []
    asterisk_pair_counter = -1
    missing_pair_flag = -1
    for token in tokenized_sentence:
        if token == "*":
            tmp_mask.append(asterisk_pair_counter)
            missing_pair_flag = -missing_pair_flag
            if missing_pair_flag == -1:
                asterisk_pair_counter += -2
        elif re.match(r'^[^\w\s]+$',token):
            tmp_mask.append(0)
        else:
            tmp_mask.append(2)
    mask = []
    lookup = dict(enumerate(tmp_mask))
    for i, value in lookup.items():
        if value == 0:
            mask.append(0)
        elif value == 2:
            if sum([v for v in list(lookup.values())[:i] if v < 0]) % 2 == 0:
                mask.append(0)
            else:
                mask.append(1)
    # print(tmp_mask)
    # print(mask)
    # print(tokenized_sentence)
    return mask

def create_spanmask_col(sentence_col):
    """
    :param sentence_col: column of dataframe that contains a sentence to compute `spanmask` for
    :return: new column (list) containing the span mask for each sentence
    """
    spanmask_col = []
    for sentence in sentence_col:
        print(sentence)
        if isinstance(sentence,str):
            spanmask_col.append(spanmask(sentence))
        else:
            print(type(sentence), sentence)
            spanmask_col.append(math.nan)
    return spanmask_col

df_test["Sentence12_spanmask_1"] = create_spanmask_col(df_test["Sentence1_marked_1"]+" "+df_test["Sentence2_marked_1"])
df_test["Sentence12_spanmask_2"] = create_spanmask_col(df_test["Sentence1_marked_2"]+" "+df_test["Sentence2_marked_2"])
df_test["Sentence12_spanmask_3"] = create_spanmask_col(df_test["Sentence1_marked_3"]+" "+df_test["Sentence2_marked_3"])

def get_human_aggreg_values(spanmask1,spanmask2,spanmask3):
    """
    :param spanmask1:   [1,0,1,1,1]
    :param spanmask2:   [1,0,0,0,1]
    :param spanmask3:   [1,0,1,0,1]
    :return:            [3,0,2,1,3]
    """
    aggreg_mask = [sum(tup) for tup in zip(spanmask1,spanmask2,spanmask3)]
    return aggreg_mask

df_test["human_aggreg_values"] = [get_human_aggreg_values(l1,l2,l3)
                                  for l1,l2,l3 in zip(df_test["Sentence12_spanmask_1"],
                                                      df_test["Sentence12_spanmask_2"],
                                                      df_test["Sentence12_spanmask_3"])]

test_dataset_human_aggregate_values = df_test["human_aggreg_values"].tolist()

"""
Counting mean topk per method: different dynamic k thresholds. Results in thresholds table (Table 1).
Dynamic top-k with local maxima:
"""
def get_local_maxima(list_of_floats, dyn_threshold="mean", peaks=True):
    """
    Computes the local maxima of a list of floats and returns respective indices.
    Algorithm:
        Point is local maxima if greater than its strict left and right neighbor (except points at index = 0|-1,
        which should only be greater than right or left strict neighbor, respectively) and if greater or equal than a
        threshold. Threshold is mean of the distribution.
    :param list_of_floats:  e.g. list of attribution values
    :return:                array of indices of local maxima
    """
    try:
        if dyn_threshold=="mean":
            threshold = np.mean(list_of_floats)
        elif dyn_threshold=="mean_plus_1std":
            threshold = np.mean(list_of_floats) + 1 * np.std(list_of_floats)
        elif dyn_threshold=="mean_plus_2std":
            threshold = np.mean(list_of_floats) + 2 * np.std(list_of_floats)
        elif dyn_threshold == "mean_min_1std":
            threshold = np.mean(list_of_floats) - 1 * np.std(list_of_floats)
        elif dyn_threshold == "mean_min_2std":
            threshold = np.mean(list_of_floats) - 2 * np.std(list_of_floats)
        elif dyn_threshold=="median":
            threshold = np.median(list_of_floats)

        elif dyn_threshold=="mean_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos)
        elif dyn_threshold=="mean_plus_1std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) + 1 * np.std(list_of_floats_pos)
        elif dyn_threshold=="mean_plus_2std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) + 2 * np.std(list_of_floats_pos)
        elif dyn_threshold == "mean_min_1std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) - 1 * np.std(list_of_floats_pos)
        elif dyn_threshold == "mean_min_2std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) - 2 * np.std(list_of_floats_pos)
        elif dyn_threshold=="median_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.median(list_of_floats_pos)

        if peaks==False:
            indices = np.where(list_of_floats >= threshold)[0]
            indices = list(set(indices.tolist() + []))
            indices.sort()
            return np.array(indices)

        # Roll the input list to create arrays representing left and right neighbors of each element
        # e.g.
        # roll_left         = [0.1, 0.2, 0.3]
        # list_of_floats    = [0.3, 0.1, 0.2]
        # roll_right        = [0.2, 0.3, 0.1]
        # -> for each element list_of_floats[i], its strict neighbors are roll_left[i] and roll_right[i]
        roll_left = np.roll(list_of_floats, 1)
        roll_right = np.roll(list_of_floats, -1)

        # Find indices where the current element is greater than its strict left and right neighbors,
        # and the current element is greater than or equal to the threshold
        indices = \
        np.where((roll_left < list_of_floats) & (roll_right < list_of_floats) & (list_of_floats >= threshold))[0]
        # print(list_of_floats)
        # print(indices)
        # Create a list to store additional indices for special cases (first and last elements)
        additional_indices = []

        # Check if the first element is greater than the second and greater or equal to the threshold
        if list_of_floats[0] > list_of_floats[1] and list_of_floats[0] >= threshold:
            additional_indices.append(0)

        # Check if the last element is greater than the second-to-last and greater or equal to the threshold
        if list_of_floats[-1] > list_of_floats[-2] and list_of_floats[-1] >= threshold:
            additional_indices.append(len(list_of_floats) - 1)

        # Check for spikes with the middle point as a local maximum
        i = 1
        while i < len(list_of_floats) - 1:
            if list_of_floats[i] >= threshold:
                j = i
                # Continue iterating through the list while consecutive elements have the same value
                while j < len(list_of_floats) - 1 and list_of_floats[j] == list_of_floats[j + 1]:
                    j += 1
                if j > i:
                    # Check if the cluster is attached to a higher peak without lower points in between
                    try:
                        if (list_of_floats[i - 1] < list_of_floats[i]) and (list_of_floats[j + 1] < list_of_floats[i]):
                            cluster_size = j-i + 1 # j is index last element in cluster, i is index first element
                            # Find the middle point of the cluster and mark it as a local maximum, if n elements uneven
                            if cluster_size % 2 != 0:
                                middle_idx = i + (cluster_size//2)
                                additional_indices.append(middle_idx)
                            # Find middle two elements and take random choice if n elements in cluster are even
                            else:
                                # Calculate the index of the first center element
                                center1_index =  i + (cluster_size//2) - 1  # Subtract 1 because Python uses 0-based indexing
                                # Calculate the index of the second center element
                                center2_index =  i + (cluster_size//2)
                                random_middle_idx = random.choice([center1_index,center2_index])
                                additional_indices.append(random_middle_idx)
                        else:
                            # Skip the cluster if it is attached to a higher peak without lower points in between
                            pass
                    except IndexError:
                        # Handle the case where an error occurs (e.g., if the cluster is at the beginning or end)
                        # print("Skipped error")  # Skip an error for one specific case
                        pass
                    i = j
            i += 1

        # Combine the main indices and additional indices, remove duplicates, and sort them
        indices = list(set(indices.tolist() + additional_indices))
        indices.sort()

        return np.array(indices)
    except Exception as e:
        print("Error:", e)
        return np.array([])  # Return an empty array if an exception occurs

def compute_and_plot_local_maxima(list_of_floats, dyn_threshold="mean", plot=True, peaks=True):
    """
    Compute local maxima of a list of floats
    :param list_of_floats:  e.g. list of attribution values
    :param plot:            defaults to True    --> plots the curves with local maxima
    :return:                tuple of 2          --> (list of floats , local maxima indices)

    #example
    d = [0.1,0.5,0.8,0.2,0.3,0.1,0.5,0.6,0.6,0.1]
    compute_and_plot_local_maxima(d)
    """
    local_maxima_indices = get_local_maxima(list_of_floats, dyn_threshold=dyn_threshold, peaks=peaks)
    if plot:
        plt.plot(list_of_floats)
        plt.plot(local_maxima_indices, [list_of_floats[i] for i in local_maxima_indices], 'ro')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Peaks in Data')
        plt.show()
    return list_of_floats, local_maxima_indices

def get_mean_std_dynk(dataset_explanations, dataset_human_aggreg_values, dyn_threshold="mean", peaks=True):
    ddddd = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6:[]}
    for x in dataset_explanations:
        for i in range(6):
            k_indices = get_local_maxima([s for s, t in zip(x[i].scores, x[i].tokens) if t not in ["[CLS]", "[SEP]"]],
                                         dyn_threshold=dyn_threshold, peaks=peaks)
            ddddd[i].append(len(k_indices))
    for y in dataset_human_aggreg_values:
        k_indices = get_local_maxima(y, dyn_threshold=dyn_threshold, peaks=peaks)
        ddddd[6].append(len(k_indices))

    names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}
    for k, v in ddddd.items():
        print(names[k], round(np.mean(v),2), "+-", round(np.std(v),2))


thresholds_dynk = ["mean","mean_plus_1std","mean_plus_2std","mean_min_1std","mean_min_2std","median",
          "mean_pos","mean_plus_1std_pos","mean_plus_2std_pos","mean_min_1std_pos","mean_min_2std_pos","median_pos"]
for T in thresholds_dynk:
    print(T)
    get_mean_std_dynk(dataset_explanations=test_dataset_explanations,
                      dataset_human_aggreg_values=test_dataset_human_aggregate_values,
                      dyn_threshold=T,
                      peaks=True)
"""
Statistical tests on thresholds table: Euclidean distance between two pairs (mean1, stdev1) and (mean2, stdev2)
"""
def euclidean_distance(pair1, pair2):
    """
    Euclidean distance between two pairs (mean1, stdev1) and (mean2, stdev2)
    :param pair1: (mean1, stdev1)
    :param pair2: (mean2, stdev2)
    :return: euclidean distance
    """
    mean_diff = pair2[0] - pair1[0]
    stdev_diff = pair2[1] - pair1[1]
    mean_diff_squared = mean_diff ** 2
    stdev_diff_squared = stdev_diff ** 2
    sum_squared_diff = mean_diff_squared + stdev_diff_squared
    euclidean_distance = math.sqrt(sum_squared_diff)
    return euclidean_distance

pair1_human_preference = (4, 3)  # (mean1, stdev1), from Kamp et al. (2023)

# thresholds table (values are copied from above command output:
# `for T in thresholds_dynk:
#     print(T)
#     get_mean_std_dynk(...`
th_dict = {'th_00': [(4.54, 1.73), (5.34, 2.35), (4.58, 1.68), (6.83, 2.59), (7.30, 2.63), (5.68, 2.37)],
           'th_01': [(2.16, 0.95), (2.24, 1.05), (2.41, 1.02), (2.39, 1.12), (2.66, 1.23), (2.27, 1.08)],
           'th_02': [(1.25, 0.65), (1.23, 0.65), (1.39, 0.61), (0.68, 0.65), (0.64, 0.63), (1.02, 0.62)],
           'th_03': [(7.36, 1.89), (8.31, 2.86), (7.63, 2.68), (8.21, 2.82), (8.41, 2.88), (8.04, 2.80)],
           'th_04': [(7.37, 1.89), (8.32, 2.87), (7.64, 2.69), (8.28, 2.83), (8.46, 2.90), (8.07, 2.82)],
           'th_05': [(6.19, 1.62), (7.13, 2.48), (6.20, 2.08), (7.08, 2.51), (7.41, 2.58), (6.83, 2.39)],
           'th_06': [(3.34, 1.33), (3.56, 1.56), (4.58, 1.68), (3.51, 1.56), (3.47, 1.60), (3.83, 1.69)],
           'th_07': [(1.86, 0.82), (1.87, 0.87), (2.41, 1.02), (1.75, 0.83), (1.67, 0.81), (1.91, 0.90)],
           'th_08': [(1.07, 0.55), (1.06, 0.53), (1.39, 0.61), (0.73, 0.56), (0.62, 0.55), (0.98, 0.53)],
           'th_09': [(7.00, 2.01), (7.95, 2.91), (7.63, 2.68), (6.81, 2.82), (6.60, 2.81), (7.57, 2.74)],
           'th_10': [(7.28, 1.93), (8.25, 2.89), (7.64, 2.69), (7.86, 2.92), (7.81, 3.05), (7.99, 2.81)],
           'th_11': [(5.01, 1.61), (5.59, 2.04), (6.20, 2.08), (4.69, 1.88), (4.54, 1.86), (5.38, 1.96)]
          }
#euclidean distances per threshold
ed_dict = {'th_00': [],
           'th_01': [],
           'th_02': [],
           'th_03': [],
           'th_04': [],
           'th_05': [],
           'th_06': [],
           'th_07': [],
           'th_08': [],
           'th_09': [],
           'th_10': [],
           'th_11': []
          }

for k,v in th_dict.items():
    for pair2 in v:
        ed_dict[k].append(euclidean_distance(pair1_human_preference,pair2))

# Lowest three are highlighted with dark background in Table 1
for k,v in ed_dict.items():
    print(k, sum(v))

"""
Linguistic features
"""
benepar.download('benepar_en3')
nlp = spacy.load("en_core_web_lg")

# keep separation between tokens as given by distilbert tokenizer -> first join and then split by whitespace
# in `get_parsed_instance()` -> parsed = pos_dep_stop(" ".join(tokens))
def whitespace_tokenizer(sentence):
    tokens = sentence.split(" ")
    return Doc(nlp.vocab, tokens)

nlp.tokenizer = whitespace_tokenizer

def get_pos_dep_stop(sentence):
    """
    e.g.: get_pos_dep_stop("Apple is looking at buying U.K. startup for $1 billion")
    """
    sentence = nlp(sentence)
    parsed = []
    for tok in sentence:
        #parsed.append([tok.text, tok.lemma_, tok.pos_, tok.tag_, tok.dep_, tok.shape_, tok.is_alpha, tok.is_stop])
        parsed.append([tok.pos_, tok.dep_, tok.is_stop])
    return parsed

tagger = SequenceTagger.load("flair/chunk-english")

def get_chunks(sentence):
    """
    e.g. get_chunks("Three children in a black dog ken ##nel . the kids are being punished .")
    :param sentence: sentence string joined on whitespace
    :return:    list of chunks indices, e.g. [[0, 1], [2], [3, 4, 5, 6, 7], [8], [9, 10], [11, 12, 13], [14]]
                list of chunks tokens, e.g. [["a","b"], ["c"], ...]
                list of chunks tags, e.g. ["NP", "VP", "PP", ...]
                # all three lists are of the same length and are padded after the chunker doesn't include punctuation
    """
    s = Sentence(sentence, use_tokenizer=False)
    tagger.predict(s)
    list_of_chunks_indices = []
    list_of_chunks_tags = []
    list_of_chunks_tokens = []
    for chunk in s.get_spans('np'):
        print(chunk, "NB: indices tokens should be i-1")
        # (token.idx - 1) because indexing starts at 0 and their output starts at 1 for some reason
        list_of_chunks_indices.append([token.idx - 1 for token in chunk])
        list_of_chunks_tokens.append([token.text for token in chunk])
        list_of_chunks_tags.append(chunk.tag)

    # padding punctuation chunks that were excluded by the chunker
    n_total_tokens = len(sentence.split(" ")) # need to know in order to pad
    flattened = [idx for chunk in list_of_chunks_indices for idx in chunk] # need to know in order to pad
    for idx in range(n_total_tokens):
        if idx not in flattened:
            list_of_chunks_indices.append([idx])
            list_of_chunks_tokens.append(["<PAD_punct_token>"])
            list_of_chunks_tags.append("<PAD_punct_tag>")

    # sort all three lists based on how the first list (of indices) would be sorted. This way, the missing chunks are
    # placed at the right place and the order is maintained across lists.
    list_of_chunks_indices_sorted, list_of_chunks_tokens_sorted, list_of_chunks_tags_sorted = map(list, zip(*sorted(zip(
        list_of_chunks_indices, list_of_chunks_tokens, list_of_chunks_tags), key=lambda x: x[0][0])))

    return (list_of_chunks_indices_sorted,
           list_of_chunks_tokens_sorted,
           list_of_chunks_tags_sorted)

# dataset_parsed has two components that are computed separately.
# Eventually it will be a tuple of 3: (parsed, chunked, masks)
# 1) parsed instance: just parses and chunks the instance, doesn't matter what setting of topk we put here.
# I copied this function and didn't have time to clean it up
def get_parsed_instance(explanations, topk=-1, ranked=False, rm_special_tokens=True,
                        leaveout_idx_list=[], human_aggreg_values=[], dynamic=False, dyn_threshold="mean"):

    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
    # leave one or more attribution methods out depending on parameter
    explanations = [x for n,x in enumerate(explanations) if n not in leaveout_idx_list]
    # compute list of method-wise masks, e.g. [ [0,0,0,1,1], [0,0,1,1,1], [0,0,1,0,1] ]
    list_of_masks = [create_topk_mask(attribs=list(x.scores),
                                      tokens=list(x.tokens),
                                      topk=topk,
                                      ranked=ranked,
                                      rm_special_tokens=rm_special_tokens,
                                      dynamic=dynamic,
                                      dyn_threshold=dyn_threshold)
                     for x in explanations]

    if len(human_aggreg_values) > 0:
        assert rm_special_tokens == True #human gt does (can)not have special tokens
        assert ranked == False #did not test for ranked
        assert len(human_aggreg_values) == len(list_of_masks[0])

        if not dynamic:
            human_assign_indices = list(enumerate(human_aggreg_values))
            human_assign_indices.sort(key=lambda tup: (tup[1], random.random()))  # if tie -> randomize
            human_sorted_indices = [i for i, a in human_assign_indices]
            human_topk_sorted_indices = human_sorted_indices[-topk:]  # e.g. top-2 of [1,2,3,0] is [3,0]
            human_topk_mask = [0 for _ in human_aggreg_values]  # initialize 0s mask
            for i in human_topk_sorted_indices:
                human_topk_mask[i] = 1  # assign 1 at topk indices; outside the topk remains 0
            list_of_masks.append(human_topk_mask)
        else:
            human_topk_mask = create_topk_mask( attribs=human_aggreg_values,
                                                tokens=["dummy" for _ in human_aggreg_values],
                                                topk=topk,
                                                ranked=ranked,
                                                rm_special_tokens=rm_special_tokens,
                                                dynamic=dynamic,
                                                dyn_threshold=dyn_threshold)
            list_of_masks.append(human_topk_mask)

    # tupled_tokenwise = list(zip(*list_of_masks)) # WE DON'T MEASURE AGREEMENT HERE, JUST TOPK MASKS & PARSE
    # agreements_tokenwise = [get_token_agr_score(tup,class_1_agreement=class_1_agreement)
    #                         for tup in tupled_tokenwise]
    if rm_special_tokens:
        tokens = [t for t in explanations[0].tokens if t not in {"[CLS]","[SEP]","<s>","</s>"}]
    else:
        tokens = list(explanations[0].tokens)

    parsed = get_pos_dep_stop(" ".join(tokens))
    chunked = get_chunks(" ".join(tokens)) # tuple of 3
    assert len(parsed) == len(list_of_masks[0])
    return (parsed,
            chunked)
# 2) list of masks
def get_list_of_masks_instance(explanations, topk=-1, ranked=False, rm_special_tokens=True,
                        leaveout_idx_list=[], human_aggreg_values=[], dynamic=False, dyn_threshold="mean"):

    if dynamic == False: #regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0
    else:
        assert topk == -1 #contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
    # leave one or more attribution methods out depending on parameter
    explanations = [x for n,x in enumerate(explanations) if n not in leaveout_idx_list]
    # compute list of method-wise masks, e.g. [ [0,0,0,1,1], [0,0,1,1,1], [0,0,1,0,1] ]
    list_of_masks = [create_topk_mask(attribs=list(x.scores),
                                      tokens=list(x.tokens),
                                      topk=topk,
                                      ranked=ranked,
                                      rm_special_tokens=rm_special_tokens,
                                      dynamic=dynamic,
                                      dyn_threshold=dyn_threshold)
                     for x in explanations]

    if len(human_aggreg_values) > 0:
        assert rm_special_tokens == True #human gt does (can)not have special tokens
        assert ranked == False #did not test for ranked
        assert len(human_aggreg_values) == len(list_of_masks[0])

        if not dynamic:
            human_assign_indices = list(enumerate(human_aggreg_values))
            human_assign_indices.sort(key=lambda tup: (tup[1], random.random()))  # if tie -> randomize
            human_sorted_indices = [i for i, a in human_assign_indices]
            human_topk_sorted_indices = human_sorted_indices[-topk:]  # e.g. top-2 of [1,2,3,0] is [3,0]
            human_topk_mask = [0 for _ in human_aggreg_values]  # initialize 0s mask
            for i in human_topk_sorted_indices:
                human_topk_mask[i] = 1  # assign 1 at topk indices; outside the topk remains 0
            list_of_masks.append(human_topk_mask)
        else:
            human_topk_mask = create_topk_mask( attribs=human_aggreg_values,
                                                tokens=["dummy" for _ in human_aggreg_values],
                                                topk=topk,
                                                ranked=ranked,
                                                rm_special_tokens=rm_special_tokens,
                                                dynamic=dynamic,
                                                dyn_threshold=dyn_threshold)
            list_of_masks.append(human_topk_mask)

    # tupled_tokenwise = list(zip(*list_of_masks)) # WE DON'T MEASURE AGREEMENT HERE, JUST TOPK MASKS & PARSE
    # agreements_tokenwise = [get_token_agr_score(tup,class_1_agreement=class_1_agreement)
    #                         for tup in tupled_tokenwise]
    if rm_special_tokens:
        tokens = [t for t in explanations[0].tokens if t not in {"[CLS]","[SEP]","<s>","</s>"}]
    else:
        tokens = list(explanations[0].tokens)

    #parsed = get_pos_dep_stop(" ".join(tokens))
    #chunked = get_chunks(" ".join(tokens)) # tuple of 3
    #assert len(parsed) == len(list_of_masks[0])
    return list_of_masks

def tokenmask_to_chunkmask(tokenmask, list_of_chunks_indices):
    """
    :param tokenmask:
    :param list_of_chunks_indices:
    :return:
        tokenmask_to_chunkmask( [0,1,       0,      0,1,        1,0],
                                [[0,1],     [2],    [3,4],      [5,6]])
                             >> [1,         0,      1,          1]
    """
    chunkmask = []
    for chunk in list_of_chunks_indices:
        if 1 in [tokenmask[idx] for idx in chunk]:
            chunkmask.append(1)
        else:
            chunkmask.append(0)
    return chunkmask

def get_pairwise_combinations(iterable):
    pairwise_combinations = []
    for el_a in iterable:
        for el_b in iterable:
            if el_a == el_b: # ignore perfect agreement between same method
                continue
            if (el_a, el_b) and (el_b, el_a) not in pairwise_combinations:
                pairwise_combinations.append((el_a,el_b))
    return pairwise_combinations

def get_pairwise_instance_chunks_agreement_scores(list_of_tokenmasks, list_of_chunks_indices):
    """
    :param list_of_tokenmasks:
    :param list_of_chunks_indices:
    :return: list of instance agreement scores, one for each pairwise combination (method-method or method-human)
    """
    pairwise_combinations = get_pairwise_combinations(range(len(list_of_tokenmasks)))
    list_of_pairwise_scores = []
    for idx_method_a, idx_method_b in pairwise_combinations:
        chunkmask_a = tokenmask_to_chunkmask(list_of_tokenmasks[idx_method_a], list_of_chunks_indices)
        chunkmask_b = tokenmask_to_chunkmask(list_of_tokenmasks[idx_method_b], list_of_chunks_indices)
        tupled_chunkwise = list(zip(*[chunkmask_a,chunkmask_b]))
        # aligning chunkmasks, not tokenmasks, but same principle:
        agreements_chunkwise = [get_token_agr_score(tup,class_1_agreement=True)
                                for tup in tupled_chunkwise]
        list_of_pairwise_scores.append(np.nanmean(agreements_chunkwise)) #ignore nans
    return list_of_pairwise_scores

def get_pairwise_dataset_chunks_agreement_scores(dataset_parsed):
    dataset_lists_of_pairwise_scores = []
    c=0
    for _, (list_of_chunks_indices,__,___), list_of_tokenmasks in dataset_parsed:
        dataset_lists_of_pairwise_scores.append(
            get_pairwise_instance_chunks_agreement_scores(list_of_tokenmasks, list_of_chunks_indices))
        print(c)
        c+=1
    combination_wise_instance_scores = [list(single_combination_list_of_instance_scores)
                                        for single_combination_list_of_instance_scores
                                        in zip(*dataset_lists_of_pairwise_scores)]
    combination_wise_dataset_scores = [np.nanmean(single_combination_list_of_instance_scores)
                                       for single_combination_list_of_instance_scores
                                       in combination_wise_instance_scores]
    return combination_wise_dataset_scores

if False: #creating the file (not needed if pickle is already available. Set to `if False` to avoid running again)
    dataset_parsed_no_masks = []
    for i in range(len(test_dataset_explanations)):
        dataset_parsed_no_masks.append(get_parsed_instance(explanations=test_dataset_explanations[i],topk=4,
                                                           rm_special_tokens=True,
                                                           leaveout_idx_list=[],
                                                           human_aggreg_values=test_dataset_human_aggregate_values[i],
                                                           dynamic=False))
        print(i, "DONE")
    with open("./esnli/parses/dataset_parsed_no_masks.pickle", "wb") as file:
        pickle.dump(dataset_parsed_no_masks, file)
else: #if pickled file is available
    with open("./esnli/parses/dataset_parsed_no_masks.pickle", "rb") as file:
        dataset_parsed_no_masks = pickle.load(file)

"""
statistics on chunks
"""
n_chunks_dataset = [len(x[1][2]) for x in dataset_parsed_no_masks]
print(np.mean(n_chunks_dataset), np.min(n_chunks_dataset), np.max(n_chunks_dataset))
print(np.std(n_chunks_dataset))

n_tokens_dataset = [len(x[0]) for x in dataset_parsed_no_masks]
print(np.mean(n_tokens_dataset), np.min(n_tokens_dataset), np.max(n_tokens_dataset))
print(np.std(n_tokens_dataset))

n_proportion_chunks_tokens_dataset = [n_chunks/n_tokens for n_tokens,n_chunks in zip(n_tokens_dataset,n_chunks_dataset)]
print(np.mean(n_proportion_chunks_tokens_dataset), np.min(n_proportion_chunks_tokens_dataset), np.max(n_proportion_chunks_tokens_dataset))

chunk_lens = [len(c) for ex in [x[1][0] for x in dataset_parsed_no_masks] for c in ex]
print(np.mean(chunk_lens), np.min(chunk_lens), np.max(chunk_lens))

# k = 4
dataset_parsed_keq4 = [(parses, chunks, masks)
                       for (parses, chunks), masks
                       in zip(dataset_parsed_no_masks,
                              [get_list_of_masks_instance(explanations=test_dataset_explanations[i], topk=4,
                                                          rm_special_tokens=True, leaveout_idx_list=[],
                                                          human_aggreg_values=test_dataset_human_aggregate_values[i],
                                                          dynamic=False)
                               for i in range(len(test_dataset_explanations))])]

# prepare for visuals (horizontal barplots with punct/isstop/POS stats)
parsed_dicts_keq4 = [
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]},
    {"pos":[],"dep":[],"is_stop":[]}
]
for parsed_tokens, _, list_of_masks in dataset_parsed_keq4: # for example in dataset...
    for i in range(7):
        topk_parsedtokens = [parsedtoken for parsedtoken,mask in list(zip(parsed_tokens,list_of_masks[i])) if mask==1]
        parsed_dicts_keq4[i]["pos"] += [x[0] for x in topk_parsedtokens]
        parsed_dicts_keq4[i]["dep"] += [x[1] for x in topk_parsedtokens]
        parsed_dicts_keq4[i]["is_stop"] += [x[2] for x in topk_parsedtokens]

# rel frequencies POS
pos_dict_keq4 = {0:{},1:{},2:{},3:{},4:{},5:{},6:{}}
for i in range(7):
    # l = parsed_dicts_keq4[i]["pos"]
    # count_d = Counter(l)
    # for item, count in count_d.items():
    #     relative_frequency = count / len(l)
    #     print(f"Item {item}: Relative Frequency = {relative_frequency:.2f}")
    # print()
    l = parsed_dicts_keq4[i]["pos"]
    count_d = Counter(l)
    for pos in ["NOUN", "VERB", "ADJ", "ADP", "DET"]:  # for k=4: 0.47, 0.20, 0.08, 0.07, 0.06
        pos_dict_keq4[i][pos] = round(count_d[pos] / len(l), 2)

# rel frequencies DEP
for i in range(7):
    l = parsed_dicts_keq4[i]["dep"]
    count_d = Counter(l)
    for item, count in count_d.items():
        relative_frequency = count / len(l)
        print(f"Item {item}: Relative Frequency = {relative_frequency:.2f}")
    print()

# rel frequencies IS_STOP
isstop_dict_keq4 = {0:{},1:{},2:{},3:{},4:{},5:{},6:{}}
for i in range(7):
    l = parsed_dicts_keq4[i]["is_stop"]
    count_d = Counter(l)
    for item, count in count_d.items():
        relative_frequency = round(count / len(l),2)
        if item == True:
            isstop_dict_keq4[i]["stop words"] = relative_frequency
        else:
            isstop_dict_keq4[i]["other"] = relative_frequency

# rel frequencies is_PUNCTUATION
ispunct_dict_keq4 = {0:{},1:{},2:{},3:{},4:{},5:{},6:{}}
for i in range(7):
    l = parsed_dicts_keq4[i]["pos"]
    count_d = Counter(l)
    ispunct_dict_keq4[i]["punctuation"] = round(count_d["PUNCT"] / len(l),2)
    ispunct_dict_keq4[i]["other"] = round(sum([c for pos,c in count_d.items() if pos != "PUNCT"]) / len(l),2)

"""
figure 2: horizontal barplots
"""
# increase font in plots
plt.rcParams.update({'font.size': 15})

def generate_horizontal_barplot_2_classes(count_dict, key, title, classes):
    categories = [key[i] for i in count_dict.keys()]
    percentages_0 = [count_dict[key][classes[0]] for key in count_dict.keys()]
    percentages_1 = [count_dict[key][classes[1]] for key in count_dict.keys()]

    # Reverse the order of categories and percentages
    categories = categories[::-1]
    percentages_0 = percentages_0[::-1]
    percentages_1 = percentages_1[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_positions = np.arange(len(categories))
    bar_width = 0.55

    # Define custom colors and hatch patterns
    custom_colors = ['#6497b1','#ffc2cd']
    hatch_patterns = ['/', None]

    # Create bars for the first class with a custom color and hatch
    ax.barh(bar_positions, percentages_0, bar_width, color=custom_colors[0], hatch=hatch_patterns[0], label=classes[0])

    # Create bars for the second class with a custom color and hatch, adjusting the left position
    ax.barh(bar_positions, percentages_1, bar_width, left=percentages_0, color=custom_colors[1], hatch=hatch_patterns[1], label=classes[1])

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Ratio')
    #ax.set_title(title, fontweight='bold')

    legend = ax.legend(handleheight=2, loc='upper center', bbox_to_anchor=(0.96, 0.62))
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_edgecolor('black')  # set legend box edge color to black
    #legend.get_frame().set_facecolor('none')  # set legend box face color to none / #d3d3d3

    # plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    plt.show()

def generate_horizontal_barplot_multiple_classes(count_dict, key, title, classes):
    categories = [key[i] for i in count_dict.keys()]

    # Initialize data lists for each class
    class_percentages = {class_name: [] for class_name in classes}

    # Extract data from the dictionary for each class
    for i, entry in count_dict.items():
        for class_name in classes:
            class_percentages[class_name].append(entry[class_name])

    # Reverse the order of categories and class percentages
    categories = categories[::-1]
    for class_name in classes:
        class_percentages[class_name] = class_percentages[class_name][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.55  # Width of each bar
    bar_positions = np.arange(len(categories))
    bottom = np.zeros(len(categories))  # Initialize the bottom for stacking bars

    # Define hatch patterns for each class (you can customize these)
    hatch_patterns = [None, '-', '/', '.', '\\']

    # Define hardcoded custom colors for each class
    custom_colors = ['#ffc2cd', '#ff93ac', '#ff6289', '#6497b1', '#b3cde0']

    # Create bars for each class with distinct hatch patterns and custom colors
    for i, class_name in enumerate(classes):
        percentages = class_percentages[class_name]
        hatch = hatch_patterns[i % len(hatch_patterns)]  # Cycle through hatch patterns
        color = custom_colors[i % len(custom_colors)]  # Cycle through custom colors
        ax.barh(bar_positions, percentages, bar_width, label=class_name, left=bottom, hatch=hatch, color=color)
        bottom += percentages  # Update the bottom for stacking

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Ratio')
    #ax.set_title(title, fontweight='bold')

    legend = ax.legend(handleheight=2, loc='upper center', bbox_to_anchor=(0.99, 0.75))
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_edgecolor('black')  # set legend box edge color to black
    #legend.get_frame().set_facecolor('none')  # set legend box face color to none / #d3d3d3

    #plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    plt.show()

names = {0:"PartSHAP",1:"LIME",2:"VanGrad",3:"Grad×I",4:"IntGrad",5:"IntGrad×I",6:"Human"}

generate_horizontal_barplot_2_classes(isstop_dict_keq4, names, "Relative frequency for relevant stop words, k=4",
                                      ["stop words","other"])
generate_horizontal_barplot_2_classes(ispunct_dict_keq4, names, "Relative frequency for relevant punctuation, k=4",
                                      ["punctuation","other"])
generate_horizontal_barplot_multiple_classes(pos_dict_keq4,names,"Relative frequency for relevant POS (human top-5), k=4",
                                             ["NOUN","VERB","ADJ","ADP","DET"])

"""
Statistical tests: chi-squared test for testing similarity token type preferences
"""
def print_chi_results_nicely(chi2_stat, p_val, dof, expected_table):
    print(f"Comparing classes {pair[0]} and {pair[1]}:")
    print("Chi-squared statistic:", chi2_stat)
    print("P-value:", p_val)
    print("Degrees of freedom:", dof)
    print("\nExpected frequencies:")
    print(expected_table)
    print("\n" + "=" * 50 + "\n")

def combine_p_values(list_of_p_values):
    z_scores = np.sqrt(chi2.ppf(1 - np.array(list_of_p_values), df=1))
    chi_squared_sum = np.sum(z_scores**2)
    combined_p_value = 1 - chi2.cdf(chi_squared_sum, df=2 * len(list_of_p_values))
    return combined_p_value

# Assuming pos_observed_classes is a list of numpy arrays containing observed frequencies
pos_p_values = []
pos_observed_classes = [np.array([[v*100 for v in list(pos_dict_keq4[i].values())]]) for i in range(7)]
for pair in combinations(range(len(pos_observed_classes)), 2):
    observed_table = np.stack((pos_observed_classes[pair[0]], pos_observed_classes[pair[1]]))
    chi2_stat, p_val, dof, expected_table = chi2_contingency(observed_table)
    print_chi_results_nicely(chi2_stat, p_val, dof, expected_table)
    pos_p_values.append(p_val)
    combine_p_values(pos_p_values)

isstop_p_values = []
isstop_dict_keq4[0] = {"other":0.79, "stop words":0.21} #inverted the two values in nested dictionary, needed for the test
isstop_observed_classes = [np.array([[v*100 for v in list(isstop_dict_keq4[i].values())]]) for i in range(7)]
for pair in combinations(range(len(isstop_observed_classes)), 2):
    observed_table = np.stack((isstop_observed_classes[pair[0]], isstop_observed_classes[pair[1]]))
    chi2_stat, p_val, dof, expected_table = chi2_contingency(observed_table)
    print_chi_results_nicely(chi2_stat, p_val, dof, expected_table)
    isstop_p_values.append(p_val)
    combine_p_values(isstop_p_values)

ispunct_p_values = []
ispunct_observed_classes = [np.array([[v * 100 for v in list(ispunct_dict_keq4[i].values())]]) for i in range(7)]
for pair in combinations(range(len(ispunct_observed_classes)), 2):
    observed_table = np.stack((ispunct_observed_classes[pair[0]], ispunct_observed_classes[pair[1]]))
    chi2_stat, p_val, dof, expected_table = chi2_contingency(observed_table)
    print_chi_results_nicely(chi2_stat, p_val, dof, expected_table)
    ispunct_p_values.append(p_val)
    combine_p_values(ispunct_p_values)

"""
chunk agreement k = 4
"""
def make_confusion_matrix(combinations, scores, title="<NULL TITLE>", plot_half=False):
    # Extract unique classes from the combinations
    classes = sorted(set([item for sublist in combinations for item in sublist]))

    # Generate all possible combinations of classes
    all_combinations = list(product(classes, repeat=2))

    # Create an empty confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)))

    # Fill the confusion matrix with the scores
    for combination, score in zip(combinations, scores):
        actual_class = combination[0]
        predicted_class = combination[1]
        actual_index = classes.index(actual_class)
        predicted_index = classes.index(predicted_class)
        confusion_matrix[actual_index, predicted_index] = score

    # Set the missing combinations to the same score as their reverse combinations
    for combination in all_combinations:
        if combination not in combinations and combination[::-1] in combinations:
            reverse_index = combinations.index(combination[::-1])
            score = scores[reverse_index]
            actual_class = combination[0]
            predicted_class = combination[1]
            actual_index = classes.index(actual_class)
            predicted_index = classes.index(predicted_class)
            confusion_matrix[actual_index, predicted_index] = score

    # Set the special cases where combination is (x, x) with score 1.0
    for i, class_name in enumerate(classes):
        class_index = classes.index(class_name)
        confusion_matrix[class_index, class_index] = 1.0

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a color map for the heatmap
    cmap = colors.LinearSegmentedColormap.from_list('custom',
                                                    ['#440154', '#0A5D67', '#FAAA05', '#FFDA80', '#A6A6A6'],
                                                    N=1000)
    cmap.set_under('#000000')
    cmap.set_over('#000000')
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    min_value = np.min(confusion_matrix)
    max_value = np.max(confusion_matrix)

    # Create a mask for the upper triangular region
    mask = np.triu(np.ones_like(confusion_matrix, dtype=bool), k=1)

    # Plot the confusion matrix as a heatmap with the mask
    if plot_half:
        sns.heatmap(confusion_matrix, annot=True, cmap=cmap, fmt=".2f", cbar=True,
                    xticklabels=classes, yticklabels=classes, ax=ax, norm=norm, vmin=min_value, vmax=max_value,
                    mask=mask)
    else:
        sns.heatmap(confusion_matrix, annot=True, cmap=cmap, fmt=".2f", cbar=True,
                    xticklabels=classes, yticklabels=classes, ax=ax, norm=norm, vmin=min_value, vmax=max_value)

    # Set the axis labels and title
    # ax.set_xlabel('Attrib. Method or Human')
    # ax.set_ylabel('Attrib. Method or Human')
    ax.set_title(title, fontstyle='italic')

    # Rotate y-axis tick labels
    plt.setp(ax.get_yticklabels(), rotation=30, ha='right')
    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()

    # Show the plot
    plt.show()

pairwise_dataset_chunks_agreement_scores_keq4 = get_pairwise_dataset_chunks_agreement_scores(dataset_parsed_keq4)

names = {0:"PartSHAP",1:"LIME",2:"VanGrad",3:"Grad×I",4:"IntGrad",5:"IntGrad×I",6:"Human"}
make_confusion_matrix(combinations=[(names[i],names[j]) for (i,j) in get_pairwise_combinations(range(7))],
                      scores=pairwise_dataset_chunks_agreement_scores_keq4,
                      title=None,#"Chunks: Mean Agreement @ k = 4",
                      plot_half=True)

# count targeted spans per sentence k = 4
n_chunks_targeted_dataset = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
for _, (list_of_chunks_indices,__,___), list_of_tokenmasks in dataset_parsed_keq4:
    for i in range(7):
        n_chunks_targeted_dataset[i].append(sum(tokenmask_to_chunkmask(list_of_tokenmasks[i],list_of_chunks_indices)))
names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}
for i in range(7):
    print(names[i], np.mean(n_chunks_targeted_dataset[i]))

"""
chunk agreement at k = dynamic
"""
dataset_parsed_keqdyn = [(parses, chunks, masks)
                         for (parses, chunks), masks
                         in zip(dataset_parsed_no_masks,
                                [get_list_of_masks_instance(explanations=test_dataset_explanations[i], topk=-1,
                                                            rm_special_tokens=True, leaveout_idx_list=[],
                                                            human_aggreg_values=test_dataset_human_aggregate_values[i],
                                                            dynamic=True, dyn_threshold="mean")
                                 for i in range(len(test_dataset_explanations))])]

names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}
pairwise_dataset_chunks_agreement_scores = get_pairwise_dataset_chunks_agreement_scores(dataset_parsed=dataset_parsed_keqdyn)
make_confusion_matrix(combinations=[(names[i], names[j]) for (i, j) in get_pairwise_combinations(range(7))],
                      scores=pairwise_dataset_chunks_agreement_scores,
                      title=None,#"Chunks: Mean Agreement @ k = dyn, threshold: ["+threshold_name+"]",
                      plot_half=True)

# count targeted spans per sentence k = dyn, for threshold = mean
n_chunks_targeted_dataset = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
for _, (list_of_chunks_indices,__,___), list_of_tokenmasks in dataset_parsed_keqdyn:
    for i in range(7):
        n_chunks_targeted_dataset[i].append(sum(tokenmask_to_chunkmask(list_of_tokenmasks[i],list_of_chunks_indices)))
names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}

for i in range(7):
    print(names[i], round(np.mean(n_chunks_targeted_dataset[i]),2), "+-", round(np.std(n_chunks_targeted_dataset[i]),2))

"""
baseline token agreement and baseline span agreement for random binary masks with 16% / 23% 1s
"""
def get_instance_baseline_agr_score(mask_attribution_profile, mask_baseline):
    tupled_tokenwise = list(zip(mask_attribution_profile, mask_baseline))
    agreements_tokenwise = [get_token_agr_score(tup=tup) for tup in tupled_tokenwise]
    return np.nanmean(agreements_tokenwise)

baselines_token_agr = []
for i in range(1000):
    list_a = [1] * 16 + [0] * 84
    list_b = [1] * 16 + [0] * 84
    random.shuffle(list_a)
    random.shuffle(list_b)
    baselines_token_agr.append(get_instance_baseline_agr_score(list_a,list_b))
print(np.nanmean(baselines_token_agr))

baselines_span_agr = []
for i in range(1000):
    list_a = [1] * 23 + [0] * 67
    list_b = [1] * 23 + [0] * 67
    random.shuffle(list_a)
    random.shuffle(list_b)
    baselines_span_agr.append(get_instance_baseline_agr_score(list_a,list_b))
print(np.nanmean(baselines_span_agr))

"""
baselines tokens and baselines spans for different thresholds. Change dyn_threshold="mean" to desired threshold value.
"""
def get_scores_nocls(dataset_explanations, method_i, dataset_human_aggreg_values, strictness_y=-1):
    """
    :param dataset_explanations:
    :param method_i: index from range [0,5]; each index corresponds to a different attribution method
    :param dataset_human_aggreg_values:
    :param strictness_y: if == -1, only returns the list of attrib scores and no gold topk label
    :return: list of tuples (instance scores without CLS/SEP, gold topk from annotations based on strictness)
    """
    dataset_noCLS_scores_method_i = []
    dataset_noCLS_gold_topks = []
    for instance in dataset_explanations:
        instance_noCLS_scores_method_i = [s for s, t in zip(instance[method_i].scores, instance[method_i].tokens)
                                         if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
        dataset_noCLS_scores_method_i.append(instance_noCLS_scores_method_i)

    if strictness_y == -1:
        return dataset_noCLS_scores_method_i

    else:
        for instance in dataset_human_aggreg_values:
            instance_noCLS_gold_topk = len([s for s in instance if s >= strictness_y])
            dataset_noCLS_gold_topks.append(instance_noCLS_gold_topk)

        return list(zip(dataset_noCLS_scores_method_i, dataset_noCLS_gold_topks))

def instance_random_shuffle(attribution_profile):
    shuffled_attribution_profile = random.sample(attribution_profile, len(attribution_profile))
    return shuffled_attribution_profile

def create_dyn_topk_mask(attribs, dyn_threshold="mean", peaks=True):

    attribs2, local_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs,
                                                                   dyn_threshold=dyn_threshold,
                                                                   plot=False,
                                                                   peaks=peaks)
    dynamic_topk_mask = [0 for _ in attribs2]
    for loc_max_i in local_maxima_indices:
        dynamic_topk_mask[loc_max_i] = 1
    return dynamic_topk_mask

def get_dataset_baseline_agr_score(dataset_mask_attribution_profiles, dataset_mask_baselines):
    dataset_baseline_agr_scores = []
    for mask_ap, mask_base in zip(dataset_mask_attribution_profiles, dataset_mask_baselines):
        dataset_baseline_agr_scores.append(get_instance_baseline_agr_score(mask_ap, mask_base))
    return np.nanmean(dataset_baseline_agr_scores)

def get_instance_baseline_agr_score_chunks(mask_attribution_profile, mask_baseline, list_of_chunks_indices):
    chunkmask_a = tokenmask_to_chunkmask(mask_attribution_profile, list_of_chunks_indices)
    chunkmask_b = tokenmask_to_chunkmask(mask_baseline, list_of_chunks_indices)
    tupled_chunkwise = list(zip(chunkmask_a, chunkmask_b))
    # aligning chunkmasks, not tokenmasks, but same principle:
    agreements_chunkwise = [get_token_agr_score(tup, class_1_agreement=True) for tup in tupled_chunkwise]
    return np.nanmean(agreements_chunkwise)

def get_dataset_baseline_agr_score_chunks(dataset_mask_attribution_profiles, dataset_mask_baselines, dataset_parsed):
    dataset_baseline_agr_scores_chunks = []
    for i, (_, (list_of_chunks_indices, __, ___)) in enumerate(dataset_parsed):
        dataset_baseline_agr_scores_chunks.append(
            get_instance_baseline_agr_score_chunks(dataset_mask_attribution_profiles[i],
                                                   dataset_mask_baselines[i],
                                                   list_of_chunks_indices=list_of_chunks_indices))
    return np.nanmean(dataset_baseline_agr_scores_chunks)

for i in range(6):
    vvv = get_scores_nocls(dataset_explanations=test_dataset_explanations,method_i=i,
                           dataset_human_aggreg_values=-1,strictness_y=-1)

    vvv_shuffled = [instance_random_shuffle(profile) for profile in vvv]

    vvv_dyn_masks = [create_dyn_topk_mask(profile) for profile in vvv]
    vvv_shuffled_dyn_masks = [create_dyn_topk_mask(shuffled_profile) for shuffled_profile in vvv_shuffled]

    agreement_with_baseline = get_dataset_baseline_agr_score(vvv_dyn_masks,vvv_shuffled_dyn_masks)
    agreement_with_baseline_chunks = get_dataset_baseline_agr_score_chunks(vvv_dyn_masks,vvv_shuffled_dyn_masks,dataset_parsed_no_masks)
    print(round(agreement_with_baseline,2), round(agreement_with_baseline_chunks,2))

""" 
viz figure 1
"""
def colorcode_strings_machine_agreement_new(lists, names, phrase_indices, phrase_tags):
    """
    :param lists: list of (6) tupled agreement masks (token, mask) for a single instance
    :param names: list of names corresponding to the methods, e.g., ["PartSHAP", "LIME", ...]
    :param phrase_indices: list of lists containing indices of tokens corresponding to each phrase
    :param phrase_tags: list of phrase tags corresponding to each token
    :return: html object containing colorcoded strings in table format
    """
    # Define CSS styles
    css = """
    <style>
    .container {
        display: flex;
        flex-direction: row;
    }
    table {
        border-collapse: collapse;
        margin-right: 20px;
    }
    td {
        padding: 6px 3px; /* Adjust padding to 5px vertically and 3px horizontally */
        text-align: center;
        font-family: sans-serif;
    }
    .bg-black {
        background-color: black;
        color: white;
    }
    .bg-white {
        background-color: white;
        color: black;
    }
    .bg-first-column {
        background-color: #ffc2cd; /* Set the background color for the first column */
        color: black;
    }
    .phrase {
        font-weight: bold;
    }
    .line {
        border-top: 1px solid black;
        width: 80%; /* Adjust the width of the line */
        margin: auto; /* Center the line */
    }
    </style>
    """

    # Create table rows
    rows = ""
    for lst, name in zip(lists, names):
        row = "<tr>"
        row += f"<td class='bg-first-column'>{name}</td>"  # Add the name in the first column with the specified background color
        for item in lst:
            string, integer = item
            if integer == 1:
                row += f"<td class='bg-white'>{string}</td>"
            elif integer == 0:
                row += f"<td class='bg-black'>{string}</td>"
        row += "</tr>"
        rows += row

    # Create phrase tags row
    phrase_tag_row = "<tr>"
    phrase_tag_row += "<td class='bg-white'></td>"  # Add an empty cell in the first column
    for indices, tag in zip(phrase_indices, phrase_tags):
        start_token = indices[0]
        end_token = indices[-1]
        # Redefine tag to "." if it's '<PAD_punct_tag>'
        tag = '.' if tag == '<PAD_punct_tag>' else tag
        line_width = (end_token - start_token + 1) * (50 if tag != '.' else 20)  # Adjust the width of the line dynamically
        # Visualize regular phrase tags with a horizontal line
        phrase_tag_row += f"<td colspan='{end_token - start_token + 1}' class='phrase'><hr class='line' style='width: {line_width}px;' />{tag}</td>"
    phrase_tag_row += "</tr>"

    table = f"<table>{rows}{phrase_tag_row}</table>"  # Combine rows and phrase tags row into an HTML table
    output = f"{css}{table}"  # Combine table and CSS into the final HTML output
    return HTML(output)  # Return an HTML object

def save_and_viz_new(dataset_explanations, dataset_human_aggregate_values, index_dataset_instance, range_topks, filename,
                     names, dataset_parsed_no_masks):
    """
    :param dataset_explanations: list of explanation objects, one for each instance in the dataset
    :param index_dataset_instance: the index of the instance in the dataset we want to visualize topk machine agr.
    :param range_topks: range of topk values we want to visualize; e.g. top=4 -> [1,2,3,4]
    :param filename: filename in .html format to save output to.
    :param names: list of names corresponding to the methods, e.g., ["PartSHAP", "LIME", ...]
    :return: nothing. Saves html to filename and opens it in a new tab of your webbrowser
    """
    instance_explanations = dataset_explanations[index_dataset_instance]
    instance_human_aggregate_values = dataset_human_aggregate_values[index_dataset_instance]

    html_list = []
    for topk in range_topks:
        mean_instance_agr_score, \
        list_of_masks, \
        agreements_tokenwise, \
        tokens = get_instance_agr_score(instance_explanations, topk=topk)
        list_of_tupled_token_masks = [list(zip(tokens, lm)) for lm in list_of_masks]

        human_topk_mask = create_topk_mask(attribs=instance_human_aggregate_values,
                                           tokens=["dummy" for _ in instance_human_aggregate_values],
                                           topk=topk,
                                           ranked=False,
                                           rm_special_tokens=True,
                                           dynamic=False)
        list_of_tupled_token_masks.append(list(zip(tokens, human_topk_mask)))
        print(list_of_tupled_token_masks)
        phrase_indices, _, phrase_tags = dataset_parsed_no_masks[index_dataset_instance][1]
        html_list.append(colorcode_strings_machine_agreement_new(lists=list_of_tupled_token_masks,
                                                                 names=names,
                                                                 phrase_indices=phrase_indices,
                                                                 phrase_tags=phrase_tags))
    html_string = "<html><body>" + ("<p></p>".join(html.data for html in html_list)) + "</body></html>"  # concat html objects
    with open(filename, "w") as file:  # save to file
        file.write(html_string)
    webbrowser.open_new_tab('file://' + os.path.abspath(filename))  # finds absolute file path and opens in the browser
    print(filename)

save_and_viz_new(dataset_explanations=test_dataset_explanations,
                     dataset_human_aggregate_values=test_dataset_human_aggregate_values,
                     index_dataset_instance=1409, #random.randint(0, len(test_dataset_explanations)), or #9489
                     range_topks=range(4,5),
                     filename="./esnli/explanations/agreement_machine_viz_"+str(1409)+".html",
                     names=["PartSHAP","LIME","VanGrad","Grad×I","IntGrad","IntGrad×I","Human"],
                     dataset_parsed_no_masks=dataset_parsed_no_masks)

"""
§3.4: Head vs modifier preference
"""
def get_groupedtokenmask(tokenmask, chunks_indices):
    """
    get_groupedtokenmask([0,0,0,0,1,0,1,0,0,1,0], [[0,1,2],[3],[4],[5,6],[7],[8,9,10]])
    Out[828]: [[0, 0, 0], [0], [1], [0, 1], [0], [0, 1, 0]]
    """
    groupedtokenmask = []
    for chunk in chunks_indices:
        groupedtokens = [tokenmask[token_idx] for token_idx in chunk]
        groupedtokenmask.append(groupedtokens)
    return groupedtokenmask

targeted_NPs_VanGrad = []
targeted_NPs_GradXI = []
targeted_NPs_POS = []

for x in dataset_parsed_keq4:
    tokenmask_PartSHAP = x[2][0]
    tokenmask_LIME = x[2][1]
    tokenmask_VanGrad = x[2][2]
    tokenmask_GradXI = x[2][3]
    chunks_indices = x[1][0]
    chunkmask_PartSHAP = tokenmask_to_chunkmask(tokenmask_PartSHAP, chunks_indices)
    chunkmask_LIME = tokenmask_to_chunkmask(tokenmask_LIME, chunks_indices)
    chunkmask_VanGrad = tokenmask_to_chunkmask(tokenmask_VanGrad, chunks_indices)
    chunkmask_GradXI = tokenmask_to_chunkmask(tokenmask_GradXI, chunks_indices)
    grouped_tokenmask_VanGrad = get_groupedtokenmask(tokenmask_VanGrad, chunks_indices)
    grouped_tokenmask_GradXI = get_groupedtokenmask(tokenmask_GradXI, chunks_indices)
    XPs = x[1][2]
    POS_sentence = [POS for POS,_,__ in x[0]]
    grouped_POS_sent = get_groupedtokenmask(POS_sentence, chunks_indices)
    for XP,cm_0,cm_1,cm_2,cm_3,gtm_2,gtm_3,gPOSs in zip(XPs,
                                                  chunkmask_PartSHAP,chunkmask_LIME,chunkmask_VanGrad,chunkmask_GradXI,
                                                  grouped_tokenmask_VanGrad,grouped_tokenmask_GradXI,grouped_POS_sent):
        if len(gtm_2) >= 2: #NP should be at least of length 2
            if XP == "NP" and sum([cm_0, cm_1, cm_2, cm_3]) == 4: # sum == 4 if all four chunkmasks are 1s
                targeted_NPs_VanGrad.append(gtm_2)
                targeted_NPs_GradXI.append(gtm_3)
                targeted_NPs_POS.append(gPOSs)

#[[1, 0, 0],[1, 0],[1, 0, 1, 0],[1, 0],[1, 1],[0, 1, 1],[0, 1, 0, 0],[1, 0],[0, 1, 1],[0, 1]]
#[[0, 1, 1],[0, 1],[0, 0, 1, 1],[0, 1],[0, 1],[0, 0, 1],[0, 1, 1, 0],[0, 1],[0, 0, 1],[0, 1]]
#[['DET', 'NOUN', 'NOUN'],['DET', 'NOUN'],['DET', 'ADV', 'ADJ', 'NOUN'],['DET', 'NOUN'],['DET', 'NOUN'],['DET', 'ADJ', 'NOUN'],['ADJ', 'ADJ', 'NOUN', 'NOUN'],['DET', 'NOUN'],['NUM', 'ADJ', 'NOUN'],['DET', 'NOUN']]

flattened_POS_list = [item for sublist in targeted_NPs_POS for item in sublist if isinstance(sublist, list)]
frequencies_POS_list = Counter(flattened_POS_list)
print(frequencies_POS_list["DET"]/len(flattened_POS_list)) #28%
print(frequencies_POS_list["NOUN"]/len(flattened_POS_list)) #47%
print(frequencies_POS_list["ADJ"]/len(flattened_POS_list)) #11%

print(frequencies_POS_list["PROPN"]/len(flattened_POS_list)) #3%
print(frequencies_POS_list["INTJ"]/len(flattened_POS_list)) # 0.06%

targeted_NPs_POS_len2 = [pos_ch for pos_ch in targeted_NPs_POS if len(pos_ch) == 2]
print(len(targeted_NPs_POS_len2)) #2702
print(len([POS_chunk for POS_chunk in targeted_NPs_POS_len2 if POS_chunk == ["DET","NOUN"]])) #1963
print(len([POS_chunk for POS_chunk in targeted_NPs_POS_len2 if POS_chunk == ["DET","NOUN"]])/len(targeted_NPs_POS_len2)) #73%

count_noun_targeted_VanGrad = 0
count_noun_mismatch = 0
for mask_VanGrad, mask_GradXI, POS_chunk in zip(targeted_NPs_VanGrad,targeted_NPs_GradXI,targeted_NPs_POS):
    if POS_chunk == ["DET","NOUN"]:
        if mask_VanGrad[1] == 1:
            count_noun_targeted_VanGrad += 1
            if mask_GradXI[1] == 0:
                count_noun_mismatch += 1
print(count_noun_targeted_VanGrad)
print(count_noun_mismatch)
print(count_noun_mismatch/count_noun_targeted_VanGrad)

# span concentration for NPs
flattened_targeted_NPs_VanGrad = [item for sublist in targeted_NPs_VanGrad for item in sublist if isinstance(sublist, list)]
flattened_targeted_NPs_GradXI = [item for sublist in targeted_NPs_GradXI for item in sublist if isinstance(sublist, list)]
print(sum(flattened_targeted_NPs_VanGrad)/len(flattened_targeted_NPs_VanGrad))
print(sum(flattened_targeted_NPs_GradXI)/len(flattened_targeted_NPs_GradXI))

# span concentration for [DET NOUN] spans
flattened_targeted_NPs_VanGrad_DETNOUN = []
flattened_targeted_NPs_GradXI_DETNOUN = []
for mask_VanGrad, mask_GradXI, POS_chunk in zip(targeted_NPs_VanGrad,targeted_NPs_GradXI,targeted_NPs_POS):
    if POS_chunk == ["DET","NOUN"]:
        flattened_targeted_NPs_VanGrad_DETNOUN += mask_VanGrad
        flattened_targeted_NPs_GradXI_DETNOUN += mask_GradXI
print(sum(flattened_targeted_NPs_VanGrad_DETNOUN) / len(flattened_targeted_NPs_VanGrad_DETNOUN))
print(sum(flattened_targeted_NPs_GradXI_DETNOUN) / len(flattened_targeted_NPs_GradXI_DETNOUN))

# span concentration: insight jump Integrated Gradient versus GradientXInput
n_chunks_targeted_dataset = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
for _, (list_of_chunks_indices,__,___), list_of_tokenmasks in dataset_parsed_keq4:
    for i in range(7):
        n_chunks_targeted_dataset[i].append(sum(tokenmask_to_chunkmask(list_of_tokenmasks[i],list_of_chunks_indices)))
names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}
for i in range(7):
    print(names[i], np.mean(n_chunks_targeted_dataset[i]))

n_chunks_targeted_dataset = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
for _, (list_of_chunks_indices, __, ___), list_of_tokenmasks in dataset_parsed_keqdyn:
    for i in range(7):
        n_chunks_targeted_dataset[i].append(sum(tokenmask_to_chunkmask(list_of_tokenmasks[i], list_of_chunks_indices)))
names = {0: "PartSHAP", 1: "LIME", 2: "VanGrad", 3: "Grad×I", 4: "IntGrad", 5: "IntGrad×I", 6: "Human"}
for i in range(7):
    print(names[i], np.mean(n_chunks_targeted_dataset[i]))

"""
Counting punctuation
"""
def is_punctuation_string_with_regex(input_string):
    """
    #Example usage:
    text = "]"
    result = is_punctuation_string_with_regex(text)
    print(result)  # This will print 1
    """
    # Define a regex pattern to match punctuation marks
    pattern = r'^[' + re.escape(string.punctuation) + ']+$'
    return 1 if re.match(pattern, input_string) else 0

counts = []
for x in test_dataset_explanations:
    count = 0
    for t in x[0].tokens:
        count += is_punctuation_string_with_regex(t)
    counts.append(count)
print(counts)
print(np.mean(counts))
print(np.std(counts))