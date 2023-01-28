import numpy as np
import pandas as pd
import time
from src import *
from baselines import *
from metrics import *

def printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
    data_name, group_0_avg_exp, group_1_avg_exp, mallows_dispersion, reference_ranking, bnd):
    # dictionary of lists

    dict = {'data_name': data_name,
            'method': method,
            'mallows_dispersion': mallows_dispersion,
            'reference_ranking': reference_ranking,
            'consensus_accuracy': consensus_accuracy,
            'exposure_ratio': exposure_ratio,
            'consensus_ranking': consensus_ranking,
            'group_0': group_0,
            'group_1': group_1,
            'group_0_avg_exp': group_0_avg_exp,
            'group_1_avg_exp': group_1_avg_exp,
            'gamma': bnd}

    results = pd.DataFrame(dict)
    print(results)
    results.to_csv(output_file, index=False)

def execute(dataset, output_file):




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
    mallows_dispersion = []
    reference_ranking = []
    lambdas = []

    #initalize hyperparameters
    bnd = .9
    fk_t = [.1] #ARP = .1
    #GP is 1, GNP is 0
    group_0_str = 'grp0-012345'
    group_1_str = 'grp1-678910111213141516171819'

    item_ids = np.arange(0, 20, 1)
    group_ids = np.concatenate((np.tile(1, 6), np.tile(0, 14)))

    for bnd in [.6, .7, .8, .9]:
        for disp in [0, .2, .4, .6, .8, 1]:
            for ref_r in [ 'a', 'b', 'c', 'd', 'e']:
                filename = "R_disp_"+str(disp)+"_fairp_"+ref_r+"_.csv"
                base_ranks = np.genfromtxt(filename, delimiter=',', dtype=int)

                print("dispersion param........", disp)
                print("ref rank...........", ref_r)


                # save results
                printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
                         data_name, group_0_avg_exp, group_1_avg_exp, mallows_dispersion, reference_ranking, bnd)

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
                mallows_dispersion.append(disp)
                reference_ranking.append(ref_r)
                lambdas.append(bnd)

                # save results
                printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
                         data_name, group_0_avg_exp, group_1_avg_exp, mallows_dispersion, reference_ranking, bnd)

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
                    mallows_dispersion.append(disp)
                    reference_ranking.append(ref_r)
                    lambdas.append(bnd)

                    # save results
                    printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0, group_1,
                             data_name, group_0_avg_exp, group_1_avg_exp, mallows_dispersion, reference_ranking, bnd)

                    if v == 'Copeland': #also do no WiG
                        epira_r, epira_group_ids = epiRA(base_ranks, item_ids, group_ids, bnd, False, v)
                        method.append('EPIRA+' + v+'_noWiG')
                        consensus_accuracy.append(calc_consensus_accuracy(base_ranks, epira_r))
                        exp, G = calc_exposure_ratio(epira_r, epira_group_ids)
                        exposure_ratio.append(exp)
                        consensus_ranking.append(epira_r)
                        group_0.append(group_0_str)
                        group_1.append(group_1_str)
                        group_0_avg_exp.append(G[0])
                        group_1_avg_exp.append(G[1])
                        data_name.append(dataset)
                        mallows_dispersion.append(disp)
                        reference_ranking.append(ref_r)
                        lambdas.append(bnd)

                        # save results
                        printoff(output_file, method, consensus_accuracy, exposure_ratio, consensus_ranking, group_0,
                                 group_1,
                                 data_name, group_0_avg_exp, group_1_avg_exp, mallows_dispersion, reference_ranking, bnd)




#execute('Mallows', 'mallows_gamma_results.csv')
