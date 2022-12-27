from src.utils import precedence_matrix_agreement
import numpy as np

def calc_consensus_accuracy(base_ranks, consensus):
    agree_count = 0
    n_voters, n_items = np.shape(base_ranks)
    precedence_mat = precedence_matrix_agreement(base_ranks)
    positions = len(consensus)
    for pos in range(positions):
        won = consensus[pos]
        lost = consensus[pos + 1: positions]
        for x in lost:
            agree_count += precedence_mat[won, x]

    print("agree count", agree_count)
    print("sum precedence_mat", np.sum(precedence_mat))
    result = agree_count/np.sum(precedence_mat)
    return result

# ranks = np.asarray([[0,1,2,3,5,4], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]])
# consensus = np.array([0,1,2,3,4,5])
#
# fit = calc_consensus_accuracy(ranks, consensus)
# print("fit", fit)

