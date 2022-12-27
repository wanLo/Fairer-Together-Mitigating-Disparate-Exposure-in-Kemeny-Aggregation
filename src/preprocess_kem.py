"""
This script corresponds to the baseline of pre-processing for fairness prior to finding the consensus.

"""
import numpy as np
from baselines.kemeny import kemeny
from src.epira import epiRA

def pre_proc_kem(base_ranks, item_ids, group_ids, bnd):
    """
    Function to pre-process input rankings to Kemeny to be fair.
    :param base_ranks: Assumes zero index
    :param item_ids: Assumes zero index
    :param group_ids: Assumes zero index
    :return: consensus: A numpy array
    """

    # perform exposure fairness on each ranking
    n_voters, n_items = np.shape(base_ranks)
    fair_base_ranks = np.zeros_like(base_ranks, dtype = int)
    for r in range(0,n_voters):
        base_rank = base_ranks[r,:]
        base_rank = np.reshape(base_rank, (1, len(base_rank)))
        fair_base_rank, _ = epiRA(base_rank, item_ids, group_ids, bnd, True, "Copeland")
        print("fair base rank", fair_base_rank)
        fair_base_ranks[r, :] = fair_base_rank

    print("fair base ranks in pre fair", fair_base_ranks)

    #perform kemeny
    result, ranking_group_ids = kemeny(fair_base_ranks, item_ids, group_ids)

    return np.asarray(result), ranking_group_ids

