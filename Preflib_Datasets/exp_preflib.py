import numpy as np
import pandas as pd
import time
from src import *
from baselines import *
from metrics import *

def printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
    data_name, group_0_avg_exp, group_1_avg_exp):
    # dictionary of lists

    dict = {'data_name': data_name,
            'method': method,
            'consensus_accuracy': consensus_accuracy,
            'exposure_ratio': exposure_ratio,
            'consensus_ranking': consensus_ranking,
            'group_0': group_0,
            'group_1': group_1,
            'group_0_avg_exp': group_0_avg_exp,
            'group_1_avg_exp': group_1_avg_exp}

    results = pd.DataFrame(dict)
    print(results)
    results.to_csv(output_file, index=False)

def execute(dataset, output_file):

    if dataset == "AGH2003":
        group_0_str = '012345-majority'
        group_1_str = '678-minority'
        filename = "agh2003.csv"
        data = np.genfromtxt(filename, delimiter=',', dtype=int)
        num_voters = np.sum(data[:, 0])
        num_items = 9
        base_ranks = np.full((num_voters, num_items), 0, dtype=int)
        # fill base ranks per SOC file standards
        v = 0
        for r in range(0, np.shape(data)[0]):
            num = data[r, 0]
            for i in range(0, num):
                base_ranks[v, :] = data[r, 1:10]
                v += 1

        base_ranks = base_ranks - 1
        item_ids = np.arange(0, 9, 1)
        group_ids = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1])

    if dataset == "AGH2004":
        group_0_str = '023-minority'
        group_1_str = '1456-majority'
        filename = "agh2004.csv"
        data = np.genfromtxt(filename, delimiter=',', dtype=int)
        num_voters = np.sum(data[:, 0])
        num_items = 7
        base_ranks = np.full((num_voters, num_items), 0, dtype=int)
        # fill base ranks per SOC file standards
        v = 0
        for r in range(0, np.shape(data)[0]):
            num = data[r, 0]
            for i in range(0, num):
                base_ranks[v, :] = data[r, 1:8]
                v += 1

        base_ranks = base_ranks - 1

        item_ids = np.arange(0, 7, 1)
        group_ids = np.asarray([1, 0, 1, 1, 0, 0, 0])

    if dataset == "Meath":
        group_0_str = 'male'
        group_1_str = 'female'
        filename = "meath.csv"
        data = np.genfromtxt(filename, delimiter=',', dtype=int)
        num_voters = np.sum(data[:, 0])
        num_items = 14
        base_ranks = np.full((num_voters, num_items), 0, dtype=int)
        # fill base ranks per SOC file standards
        v = 0
        for r in range(0, np.shape(data)[0]):
            num = data[r, 0]
            for i in range(0, num):
                base_ranks[v, :] = data[r, 1:num_items + 1]
                v += 1

        base_ranks = base_ranks - 1

        item_ids = np.arange(0, num_items, 1)
        group_ids = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    if dataset == "DublinNorth":
        group_0_str = 'male'
        group_1_str = 'female'
        filename = "dublinnorth.csv"
        data = np.genfromtxt(filename, delimiter=',', dtype=int)
        num_voters = np.sum(data[:, 0])
        num_items = 12
        base_ranks = np.full((num_voters, num_items), 0, dtype=int)
        # fill base ranks per SOC file standards
        v = 0
        for r in range(0, np.shape(data)[0]):
            num = data[r, 0]
            for i in range(0, num):
                base_ranks[v, :] = data[r, 1:num_items + 1]
                v += 1

        base_ranks = base_ranks - 1

        item_ids = np.arange(0, 12, 1)
        group_ids = np.asarray([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])


    if dataset == "DublinWest":
        group_0_str = 'male'
        group_1_str = 'female'
        filename = "dublinwest.csv"
        data = np.genfromtxt(filename, delimiter=',', dtype=int)
        num_voters = np.sum(data[:, 0])
        num_items = 9
        base_ranks = np.full((num_voters, num_items), 0, dtype=int)
        # fill base ranks per SOC file standards
        v = 0
        for r in range(0, np.shape(data)[0]):
            num = data[r, 0]
            for i in range(0, num):
                base_ranks[v, :] = data[r, 1:num_items + 1]
                v += 1

        base_ranks = base_ranks - 1

        item_ids = np.arange(0, num_items, 1)
        group_ids = np.asarray([0, 1, 1, 0, 0, 1, 0, 0, 1])


    #initialize data collectors
    method = []
    consensus_accuracy = []
    exposure_ratio = []
    consensus_ranking = []
    group_0 = []
    group_1 = []
    group_0_avg_exp = []
    group_1_avg_exp = []
    data_name = []

    #initalize hyperparameters
    bnd = .95 #exposure ratio = .95
    fk_t = [.1] #ARP = .1

    #kemeny
    print("starting KEMENY........")
    kem_r, kem_group_ids = kemeny(base_ranks, item_ids, group_ids)
    method.append('KEMENY')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, kem_r))
    exp, G = calc_exposure_ratio(kem_r, kem_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(kem_r)
    group_0.append(group_0_str)
    group_1.append(group_1_str)
    group_0_avg_exp.append(G[0])
    group_1_avg_exp.append(G[1])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
             data_name, group_0_avg_exp, group_1_avg_exp)

    # epik
    print("starting EPIK........")
    epik_r, epik_group_ids = epik(base_ranks, item_ids, group_ids, bnd)
    method.append('EPIK')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, epik_r))
    exp, G = calc_exposure_ratio(epik_r, epik_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(epik_r)
    group_0.append(group_0_str)
    group_1.append(group_1_str)
    group_0_avg_exp.append(G[0])
    group_1_avg_exp.append(G[1])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
             data_name, group_0_avg_exp, group_1_avg_exp)

    # Fair-Kemeny Cachel et al. ICDE
    print("starting FK........")
    fk_r, fk_group_ids = aggregate_rankings_fair_ilp(base_ranks, np.row_stack((item_ids, group_ids)), fk_t,
                                                     True)
    method.append('FAIRKEMENY')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, fk_r))
    exp, G = calc_exposure_ratio(fk_r, fk_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(fk_r)
    group_0.append(group_0_str)
    group_1.append(group_1_str)
    group_0_avg_exp.append(G[0])
    group_1_avg_exp.append(G[1])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
             data_name, group_0_avg_exp, group_1_avg_exp)

    # EPIRA Voting
    for v in ['Kemeny', 'Copeland', 'Schulze', 'Borda', 'Maximin']:
        epira_r, epira_group_ids = epiRA(base_ranks, item_ids, group_ids, bnd, True, v)
        method.append('EPIRA+'+ v)
        consensus_accuracy.append(calc_consensus_accuracy(base_ranks, epira_r))
        exp, G = calc_exposure_ratio(epira_r, epira_group_ids)
        exposure_ratio.append(exp)
        consensus_ranking.append(epira_r)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_avg_exp.append(G[0])
        group_1_avg_exp.append(G[1])
        data_name.append(dataset)

        # save results
        printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
                 data_name, group_0_avg_exp, group_1_avg_exp)


    #RAPF
    print("starting RAPF")
    seed = 10 #for reproducibility
    rapf_r, rapf_group_ids = RAPF(base_ranks, item_ids, group_ids, seed)
    method.append('RAPF')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, rapf_r))
    exp, G = calc_exposure_ratio(rapf_r, rapf_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(rapf_r)
    group_0.append(group_0_str)
    group_1.append(group_1_str)
    group_0_avg_exp.append(G[0])
    group_1_avg_exp.append(G[1])
    data_name.append(dataset)

    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
             data_name, group_0_avg_exp, group_1_avg_exp)

    #PRE-PROC
    print("starting pre- EPI")
    pe_r, pe_group_ids = pre_proc_kem(base_ranks, item_ids, group_ids, bnd)
    method.append('PRE-FE')
    consensus_accuracy.append(calc_consensus_accuracy(base_ranks, pe_r))
    exp, G = calc_exposure_ratio(pe_r, pe_group_ids)
    exposure_ratio.append(exp)
    consensus_ranking.append(pe_r)
    group_0.append(group_0_str)
    group_1.append(group_1_str)
    group_0_avg_exp.append(G[0])
    group_1_avg_exp.append(G[1])
    data_name.append(dataset)
    # save results
    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
             data_name, group_0_avg_exp, group_1_avg_exp)

execute("DublinWest", "blah")
execute("AGH2003", "agh2003_results.csv")
execute("AGH2004", "agh2004_results.csv")
execute("DublinNorth", "dublinnorth_results.csv")
execute("DublinWest", "dublinwest_results.csv")
execute("Meath", "meath_results.csv")
