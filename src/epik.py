import numpy as np
from itertools import combinations, permutations
import gurobipy as gp
from gurobipy import GRB
from src.utils import *
from collections import Counter
from more_itertools import unique_everseen


def epik(base_ranks, item_ids, group_ids, bnd):
    """
    Function perform fair exposure kemeny rank aggregation.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :param bnd: Desired minimum exposure ratio of consensus ranking
    :return: consensus: A numpy array of item ides representing the consensus ranking. ranking_group_ids: a numy array of
    group ids corresponding to the group membership of each item in the consensus.
    """

    # Declare and initialize model
    m = gp.Model('EPiK')

    # Create decision variable (every pair)
    items = np.unique(base_ranks[0]).tolist()
    num_voters, num_items = np.shape(base_ranks)
    item_strings = [str(var) for var in item_ids]
    group_strings = [str(var) for var in group_ids]
    item_grpid_combo_strings = list(zip(item_strings, group_strings))
    pair_combinations = [(i, j) for i in item_strings for j in item_strings] #all possible
    x = m.addVars(pair_combinations, vtype = GRB.BINARY,  name="pair")
    x = m.addVars(pair_combinations, name="pair", vtype=gp.GRB.BINARY)
    m.addConstrs((x[r,r] == 0 for r in item_strings), name="zeroselfpairs")


    # Ordering constraint Xab +Xba = 1
    print("making strict ordering.....")
    unique_pairs = [(str(i), str(j)) for i, j in combinations(range(np.shape(base_ranks)[1]), 2)] #count 01, and 10 only once
    m.addConstrs((x[a,b]+x[b,a] == 1 for a,b in unique_pairs if a != b), name='strict_order')

    # Prevent cycles constraint
    print("making cycle prevention.....")
    m.addConstrs((x[a, b] + x[b, c] + x[c,a] <= 2 for a, b in unique_pairs if a != b for c in item_strings if c != a and c != b), name='stopcycles')

    # Objective function
    print("starting objective function.....")
    pair_agreements = precedence_matrix_agreement(base_ranks)
    pair_agreement_list = pair_agreements.ravel()

    print("making objective function dictionary.....")
    #Make dictionary for Gurobi
    pair_weights = {}
    iter = 0
    for (i, j) in pair_combinations:
        pair_weights[(i, j)] = pair_agreement_list[iter]
        iter += 1
    pair_combinations, scores = gp.multidict(pair_weights)

    print("setting objective function.....")
    m.setObjective(x.prod(scores), GRB.MAXIMIZE)


    #Group Fairness
    unique_grp_ids, size_grp = np.unique(group_ids, return_counts = True)
    num_groups = len(unique_grp_ids)


    posofitem = m.addVars([i for i in item_strings], name="posofitem-id")
    m.addConstrs(((num_items - x.sum(r, '*')) + 1 == posofitem[r] for r in item_strings ),
                 name='pair2pos')  # add one for the log term
    l = m.addVars([i for i in item_strings], name="logofposforitem-id")
    for r in item_strings:
        m.addGenConstrLogA(posofitem[r], l[r], 2, "logarithm" + str(r))


    #make exposure variables e[item, grp]
    e = m.addVars(item_grpid_combo_strings, name="expofitem-id-grp") #E[item][group]

    m.addConstrs((l[r]*e[r,grp] == 1 for r, grp in item_grpid_combo_strings),
                  name='exposure')

    g = m.addVars([str(grp) for grp in unique_grp_ids], name = "groupexp-grp")


    m.addConstrs((e.sum('*', str(grp)) == g[str(grp)] for grp in unique_grp_ids),
                     name='sumgrpexposure')

    ag = m.addVars([str(grp) for grp in unique_grp_ids], name="avggroupexp-grp")

    m.addConstrs((g[str(grp)] / size_grp[np.argwhere(unique_grp_ids == grp).flatten()[0]] == ag[str(grp)] for grp in
                  unique_grp_ids), name="avgexpofgroup")


    group_tuples = list(permutations([str(g) for g in unique_grp_ids], 2))
    g_ratio = m.addVars(group_tuples, name = "ratioavgexpgrps-grp-grp")

    m.addConstrs((ag[j]*g_ratio[i,j] == ag[i] for i, j in group_tuples), name = 'ratio-avg-grp-exps')

    m.addConstrs((g_ratio[i,j] >= bnd for i, j in group_tuples), name= "lowerb-groupexpratio")
    m.addConstrs((g_ratio[i, j] <= (1/bnd) for i, j in group_tuples), name="upperb-groupexpratio")

    # m.write('epikCS.lp')
    print("starting optimization.....")
    # Run optimization engine
    m.params.NonConvex = 2
    m.optimize()

    # Uncoment to display optimal values of decision variables
    # print("Printing variables....")
    # for v in m.getVars():
    #     if v.x > 1e-6:
    #         print(v.varName, v.x)

    #extract solution
    rank_pairs = [var.varName for var in m.getVars() if var.x == 1 and var.varName.startswith('pair')]
    winning_items = [(var.split(',')[0]).split('[')[1] for var in rank_pairs]
    result = [item for items, c in Counter(winning_items).most_common()
              for item in [items] * c]
    consensus = list(unique_everseen(result))
    consensus = list(map(int, consensus))
    bottom_candidate = [item for item in range(0, num_items) if item not in consensus]
    consensus.append(bottom_candidate[0])

    ranking_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus]
    return np.asarray(consensus), ranking_group_ids


