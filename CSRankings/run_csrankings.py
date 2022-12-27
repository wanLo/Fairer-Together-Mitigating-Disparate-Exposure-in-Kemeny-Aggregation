"""
Code for CSRankings experiment. Dataset downloaded from https://github.com/KCachel/MANI-Rank/tree/main/experiments/csranking_casestudy


"""
from src import *
from metrics import *
from baselines import *
import numpy as np
import pandas as pd

def make_intersectional_attribute(groups, printgrps):
    """groups = attributes X candidates numpy, first row - items, then each subsequent is an attribute
    """
    groups = groups[1:,]
    combos = np.unique(groups, axis = 1)
    num_candidates = groups.shape[1]
    num_intersectional_groups = combos.shape[1]
    if printgrps:
        for i in range(0,num_intersectional_groups):
            print("intersectional group ", i, " represents ", list(combos[:,i]))
    intersectional = [np.where((combos.T == list(groups[:,i])).all(axis=1))[0][0]    for i in range(0,num_candidates)]
    return np.asarray(intersectional)



def printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
    data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
    west_public, south_private, south_public):
    # dictionary of lists

    dict = {'data_name': data_name,
            'method': method,
            'consensus_accuracy': consensus_accuracy,
            'exposure_ratio': exposure_ratio,
            'consensus_ranking': consensus_ranking,
            'northeast_private': northeast_private,
            'northeast_public': northeast_public,
            'midwest_private': midwest_private,
            'midwest_public': midwest_public,
            'west_private': west_private,
            'west_public': west_public,
            'south_private': south_private,
            'south_public': south_public}

    results = pd.DataFrame(dict)
    print(results)
    results.to_csv(output_file, index=False)

def CSRANKING(output_file):
    base_ranks = np.load('cs_baseranks.npy').astype(int)
    base_ranks = base_ranks[11:21, :]
    group_file = 'cs_groups.csv'

    group_info = np.genfromtxt(group_file, delimiter=',',
                                   dtype=int).T  # transpose to be of attributes x candidates

    intersectional = make_intersectional_attribute(group_info, True)
    groups_info_with_inter = np.row_stack((group_info, intersectional))

    method = []
    consensus_accuracy = []
    exposure_ratio = []
    consensus_ranking = []
    northeast_private = [] #0, 00
    northeast_public = [] #1, 01
    midwest_private = [] #2, 10
    midwest_public = [] #3, 11
    west_private = [] #4, 20
    west_public = [] #5, 21
    south_private = [] #6, 30
    south_public = [] #7, 31
    data_name = []


    bnd = .8
    fk_t = [.1]
    dataset = 'CSRankings'

    item_ids = groups_info_with_inter[0,:]
    group_ids = groups_info_with_inter[3,:]
    # kemeny
    print("starting KEMENY........")
    kem_r, kem_group_ids = kemeny(base_ranks, item_ids, group_ids)
    method.append('KEMENY')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, kem_r))
    exp, G = calc_exposure_ratio(kem_r, kem_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(kem_r)
    northeast_private.append(G[0])
    northeast_public.append(G[1])
    midwest_private.append(G[2])
    midwest_public.append(G[3])
    west_private.append(G[4])
    west_public.append(G[5])
    south_private.append(G[6])
    south_public.append(G[7])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
             data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
             west_public, south_private, south_public)

    # epik
    print("starting EPIK........")
    #epik_r, epik_group_ids = epik(base_ranks, item_ids, group_ids, bnd)
    epik_r = np.asarray([17, 10,  2,  3,  5,  7, 28,  0,  9,  6,  4,  1,  8, 13, 11, 12, 16, 15, 19, 20, 14, 27, 21, 30,
     25, 22, 18, 24, 23, 46, 26, 29, 34, 45, 39, 36, 38, 32, 33, 35, 40, 43, 41, 55, 44, 51, 37, 31,
     54, 42, 48, 49, 50, 47, 56, 57, 62, 53, 64, 60, 58, 63, 59, 52, 61])
    epik_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in epik_r])
    method.append('EPIK')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, epik_r))
    exp, G = calc_exposure_ratio(epik_r, epik_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(epik_r)
    northeast_private.append(G[0])
    northeast_public.append(G[1])
    midwest_private.append(G[2])
    midwest_public.append(G[3])
    west_private.append(G[4])
    west_public.append(G[5])
    south_private.append(G[6])
    south_public.append(G[7])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
             data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
             west_public, south_private, south_public)

    # Fair-Kemeny Cachel et al. ICDE
    print("starting FK........")
    fk_r, fk_group_ids = aggregate_rankings_fair_ilp(base_ranks, np.row_stack((item_ids, group_ids)), fk_t,
                                                     True)
    method.append('FAIRKEMENY')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, fk_r))
    exp, G = calc_exposure_ratio(fk_r, fk_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(fk_r)
    northeast_private.append(G[0])
    northeast_public.append(G[1])
    midwest_private.append(G[2])
    midwest_public.append(G[3])
    west_private.append(G[4])
    west_public.append(G[5])
    south_private.append(G[6])
    south_public.append(G[7])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
             data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
             west_public, south_private, south_public)

    # EPIRA Voting
    for v in ['Kemeny', 'Copeland', 'Schulze', 'Borda', 'Maximin']:
        epira_r, epira_group_ids = epiRA(base_ranks, item_ids, group_ids, bnd, True, v)
        method.append('EPIRA+' + v)
        consensus_accuracy.append(calc_consensus_accuracy(base_ranks, epira_r))
        exp, G = calc_exposure_ratio(epira_r, epira_group_ids)
        exposure_ratio.append(exp)
        consensus_ranking.append(epira_r)
        northeast_private.append(G[0])
        northeast_public.append(G[1])
        midwest_private.append(G[2])
        midwest_public.append(G[3])
        west_private.append(G[4])
        west_public.append(G[5])
        south_private.append(G[6])
        south_public.append(G[7])
        data_name.append(dataset)

        # save results
        printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
                 data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
                 west_public, south_private, south_public)

    # RAPF
    print("starting RAPF")
    seed = 10  # for reproducibility
    rapf_r, rapf_group_ids = RAPF(base_ranks, item_ids, group_ids, seed)
    method.append('RAPF')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, rapf_r))
    exp, G = calc_exposure_ratio(rapf_r, rapf_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(rapf_r)
    northeast_private.append(G[0])
    northeast_public.append(G[1])
    midwest_private.append(G[2])
    midwest_public.append(G[3])
    west_private.append(G[4])
    west_public.append(G[5])
    south_private.append(G[6])
    south_public.append(G[7])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
             data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
             west_public, south_private, south_public)

    # PRE-PROC
    print("starting pre- EPI")
    pe_r, pe_group_ids = pre_proc_kem(base_ranks, item_ids, group_ids, bnd)
    method.append('PRE-FE')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, pe_r))
    exp, G = calc_exposure_ratio(pe_r, pe_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(pe_r)
    northeast_private.append(G[0])
    northeast_public.append(G[1])
    midwest_private.append(G[2])
    midwest_public.append(G[3])
    west_private.append(G[4])
    west_public.append(G[5])
    south_private.append(G[6])
    south_public.append(G[7])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking,
             data_name, northeast_private, northeast_public, midwest_private, midwest_public, west_private,
             west_public, south_private, south_public)





CSRANKING("CSRankings_results.csv")