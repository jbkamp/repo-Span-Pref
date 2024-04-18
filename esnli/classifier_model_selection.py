import pickle
import numpy as np

def explpickle2matrix(explpickle_file):
    """
    Transforms the pickled explanations file to a padded matrix (see appendix of our paper for the details)
    """
    with open(explpickle_file, "rb") as file:
        explanations = pickle.load(file)
    explanations_matrix = []
    for instance in explanations:
        for x in instance:
            explanations_vector = [score for score,token in zip(x.scores,x.tokens) if token not in {"[CLS]","[SEP]"}]
            explanations_matrix.append(explanations_vector)
    max_len = max([len(vec) for vec in explanations_matrix])
    explanations_matrix_padded = [vec+[0 for n in range(max_len-len(vec))] for vec in explanations_matrix]
    return np.array(explanations_matrix_padded)

pickle_files_test_distilbert = ["./esnli/explanations/test_explanations/test_dataset_explanations_db_"+i+".pickle"
                               for i in ["01","02","03","04","05","06","07","08","09","10"]]

def print_rounded_APD(pickle_files):
    """
    print APD scores for all pickle files
    """
    explanations_matrices = [explpickle2matrix(pf) for pf in pickle_files]

    APDs = []
    for matrix1 in explanations_matrices:
        differences = [np.abs(matrix1 - matrix2) for matrix2 in explanations_matrices]
        avg_difference = np.mean(differences) #np.sum or np.mean yield same result
        APDs.append(round(avg_difference,5))

    for file,APD in zip(pickle_files,APDs):
        print("{}\t{}".format(file,APD))

print_rounded_APD(pickle_files_test_distilbert)