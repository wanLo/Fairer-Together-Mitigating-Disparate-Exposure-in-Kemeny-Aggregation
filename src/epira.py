import numpy as np
from itertools import combinations, permutations
import gurobipy as gp
from gurobipy import GRB
from src.utils import *
from collections import Counter
from more_itertools import unique_everseen
from metrics.exposure_ratio import *
from baselines.kemeny import *

def epiRA(base_ranks, item_ids, group_ids, bnd, grporder, agg_method):
   """
   Function to perform fair exposure rank aggregation via post-processing a voting rule.
   :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
   :param item_ids: Assumes zero index. Numpy array of item ids.
   :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
   :param bnd: Desired minimum exposure ratio of consensus ranking
   :param grporder: True - re orders consensus ranking to preserve within group order. False does not preserve within group order.
   :param agg_method: String indicating which voting rule to use. 'Kemeny', 'Copeland', 'Schulze', 'Borda', 'Maximin'.
   :return: consensus: A numpy array of item ides representing the consensus ranking. ranking_group_ids: a numy array of
    group ids corresponding to the group membership of each item in the consensus.
   """

   num_voters, num_items = np.shape(base_ranks)
   if agg_method == "Copeland":
       consensus, consensus_group_ids, current_ranking, current_group_ids = copeland(base_ranks, item_ids, group_ids)

   if agg_method == "Kemeny":
       kemeny_r, kemeny_group_ids = kemeny(base_ranks, item_ids, group_ids)
       consensus = list(kemeny_r)
       current_ranking = np.asarray(consensus)
       consensus_group_ids = np.asarray(kemeny_group_ids)
       current_group_ids = consensus_group_ids

   if agg_method == "Borda":
       consensus, consensus_group_ids, current_ranking, current_group_ids = borda(base_ranks, item_ids, group_ids)

   if agg_method == "Schulze":
       consensus, consensus_group_ids, current_ranking, current_group_ids = schulze(base_ranks, item_ids, group_ids)
   if agg_method == "Maximin":
       consensus, consensus_group_ids, current_ranking, current_group_ids = maximin(base_ranks, item_ids, group_ids)



   cur_exp, avg_exps = calc_exposure_ratio(current_ranking, current_group_ids)
   exp_at_position = np.array([(1 / (np.log2(i + 1))) for i in range(1, num_items + 1)])
   repositions = 0


   swapped = np.full(len(current_ranking), False) #hold items that have been swapped
   while( cur_exp < bnd ):

       # Prevent infinite loops
       if repositions > ((num_items * (num_items - 1)) / 2):
           print("Try increasing the bound")
           return current_ranking, current_group_ids
           break

       max_avg_exp = np.max(avg_exps)
       grp_min_avg_exp = np.argmin(avg_exps) #group id of group with lowest avg exposure
       grp_max_avg_exp = np.argmax(avg_exps)  # group id of group with lowest avg exposure
       grp_min_size = np.sum(group_ids == grp_min_avg_exp)
       Gmin_positions = np.argwhere(current_group_ids == grp_min_avg_exp).flatten()
       Gmax_positions = np.argwhere(current_group_ids == grp_max_avg_exp).flatten()

       indx_highest_grp_min_item = np.min(Gmin_positions)
       valid_Gmax_items = Gmax_positions < indx_highest_grp_min_item

       if np.sum(valid_Gmax_items) == 0:
           Gmin_counter = 1
           while np.sum(valid_Gmax_items) == 0:
               next_highest_ranked_Gmin = np.min(Gmin_positions[Gmin_counter:, ])
               valid_Gmax_items = Gmax_positions < next_highest_ranked_Gmin
               Gmin_counter += 1
           indx_highest_grp_min_item = next_highest_ranked_Gmin
       if swapped[indx_highest_grp_min_item] == True: #swapping same item
           #valid_grp_min = np.argwhere(~swapped & current_group_ids == grp_min_avg_exp).flatten()
           valid_grp_min = np.intersect1d(np.argwhere(~swapped).flatten(),np.argwhere(current_group_ids == grp_min_avg_exp).flatten())
           if len(valid_grp_min) != 0: indx_highest_grp_min_item = np.min(valid_grp_min)  # index of highest ranked item that was not swapped
       highest_item_exp = exp_at_position[indx_highest_grp_min_item]
       exp_grp_min_without_highest = (np.min(avg_exps) * grp_min_size) - highest_item_exp

       boost = (grp_min_size*max_avg_exp*bnd) - exp_grp_min_without_highest

       exp = np.copy(exp_at_position) #deep copy
       exp[np.argwhere(current_group_ids == grp_min_avg_exp).flatten()] = np.Inf
       exp[indx_highest_grp_min_item] = np.Inf #added 11/21
       indx = (np.abs(exp - boost)).argmin() #find position with closest exposure to boost

       min_grp_item = current_ranking[indx_highest_grp_min_item]
       print("min_grp_item",min_grp_item)
       swapping_item = current_ranking[indx]
       print("swapping_item", swapping_item)
       #put swapping item in min_grp_item position
       current_ranking[indx_highest_grp_min_item] = swapping_item
       #put min_group_item at indx
       current_ranking[indx] = min_grp_item
       repositions += 1
       swapped[indx_highest_grp_min_item] = True
       swapped[indx] = True
       #update group ids
       current_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in current_ranking]
       #set up next loop
       cur_exp, avg_exps = calc_exposure_ratio(current_ranking, current_group_ids)
       print("exposure after swap:", cur_exp)


   if grporder == True: #Reorder to preserve consensus
       pass
       consensus = np.asarray(consensus)
       current_ranking = np.ones(num_items, dtype = int)
       current_group_ids = np.asarray(current_group_ids)
       for g in np.unique(group_ids).tolist():
           where_to_put_g = np.argwhere(current_group_ids == g).flatten()
           g_ordered = consensus[np.argwhere(consensus_group_ids == g).flatten()] #order in copeland
           current_ranking[where_to_put_g] = g_ordered
       current_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in current_ranking]
       return current_ranking, np.asarray(current_group_ids)



   return np.asarray(current_ranking), np.asarray(current_group_ids)



def copeland(base_ranks, item_ids, group_ids):
    """
    Function to perform copeland voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    copelandDict = {key: 0 for key in items_list}
    pair_agreements = precedence_matrix_agreement(base_ranks)
    for item in items_list:
        for comparison_item in items_list:
            if item != comparison_item:
                num_item_wins = pair_agreements[comparison_item, item]
                num_comparison_item_wins = pair_agreements[item, comparison_item]
                if num_item_wins < num_comparison_item_wins:
                    copelandDict[item] += 1

    items = list(copelandDict.keys())
    copeland_pairwon_cnt = list(copelandDict.values())
    zip_scores_items = zip(copeland_pairwon_cnt, items)
    sorted_pairs = sorted(zip_scores_items, reverse=True)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids


def borda(base_ranks, item_ids, group_ids):
    """
    Function to perform borda voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    item_list = list(item_ids)
    bordaDict = {key: 0 for key in item_list}
    num_rankings, num_items = base_ranks.shape
    points_per_pos_legend = list(range(num_items - 1, -1, -1))

    for ranking in range(0, num_rankings):
        for item_pos in range(0, num_items):
            item = base_ranks[ranking, item_pos]
            bordaDict[item] += points_per_pos_legend[item_pos]

    candidates = list(bordaDict.keys())
    borda_scores = list(bordaDict.values())
    zip_scores_items = zip(borda_scores, candidates)
    sorted_pairs = sorted(zip_scores_items, reverse=True)
    consensus = [element for _, element in sorted_pairs] #borda
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids


def maximin(base_ranks, item_ids, group_ids):
    """
    Function to perform borda voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    maximinDict = {key: 0 for key in items_list}
    pair_agreements = precedence_matrix_agreement(base_ranks)
    for item in items_list:
        max_item_wins = 0
        for comparison_item in items_list:
           if item != comparison_item:
              num_item_wins = pair_agreements[comparison_item, item]
              max_item_wins = max(max_item_wins, num_item_wins)

        maximinDict[item] += max_item_wins

    items = list(maximinDict.keys())
    maximin_score = list(maximinDict.values())
    zip_scores_items = zip(maximin_score, items)
    sorted_pairs = sorted(zip_scores_items, reverse=False)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids



def schulze(base_ranks, item_ids, group_ids):
    """
    Function to perform schulze voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    Qmat = precedence_matrix_agreement(base_ranks)
    Pmat = np.zeros_like(Qmat)

    for i in items_list:
       for j in items_list:
           if i != j:
              if Qmat[j, i] > Qmat[i, j]:
               Pmat[j, i] = Qmat[j, i]
              else:
               Pmat[j, i] = 0

    for i in items_list:
       for j in items_list:
           if i != j:
              for k in items_list:
                if i != k and j != k:
                  Pmat[k, j] = np.maximum(Pmat[k, j], np.minimum(Pmat[i, j], Pmat[k, i]))

    wins_candidate_has_over_others = np.sum(Pmat, axis=0)
    zip_scores_items = zip(wins_candidate_has_over_others, items_list)
    sorted_pairs = sorted(zip_scores_items, reverse=False)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids

