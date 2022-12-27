"""
References:https://github.com/KCachel/MANI-Rank/blob/main/multi_fair

Code from utily.py & fair_kemeny.py

"""


import numpy as np
import pulp as pl
from itertools import combinations
from collections import Counter
from more_itertools import unique_everseen


path_to_cplex = r'C:\Program Files\IBM\ILOG\CPLEX_Studio201\cplex\bin\x64_win64\cplex.exe'

def find_solution(prob, num_candidates):
    """   Compute solution (list of candidates from PuLP binary variables.
                :param prob: A PuLP problem variable
                :param num_candidates: int number of candidates in problem.
                :return solution: python list ranking over candidates"""

    #collect variables that are true
    true_vars = [v.name for v in prob.variables() if v.varValue == 1 and v.name != 'Y']
    top_of_pair_candidate = [var.split('_')[1] for var in true_vars]
    # sorting on basis of frequency of elements
    result = [item for items, c in Counter(top_of_pair_candidate).most_common()
              for item in [items] * c]
    #get unique elements while preserving order
    solution = list(unique_everseen(result))
    #convert string candidates to ints
    solution = list(map(int, solution))
    #append last candidate
    bottom_candidate = [item for item in range(0,num_candidates) if item not in solution]
    solution.append(bottom_candidate[0])

    return np.asarray(solution)

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

def determine_group_identity(candidates, grp_mem):
    """Create dictionary with key = group id and value = candidate ids"""
    group_id_dict = {}
    for var in np.unique(grp_mem):
        idx = np.where(grp_mem == var)
        group_id_dict[(var)] = [str(item) for item in candidates[idx].tolist()]  # make it a list of str
    return group_id_dict


def all_pair_precedence(ranks):
    """construct precedence matrix"""
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))

    pwin_cand = np.unique(ranks[0]).tolist()
    plose_cand = np.unique(ranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    #for i, j in combinations(range(n_candidates), 2):
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(n_voters):
            #if list(ranks[r]).index(i) > list(ranks[r]).index(j):
            if np.argwhere(ranks[r] == i)[0][0] > np.argwhere(ranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        edge_weights[i, j] = h_ij
        edge_weights[j, i] = h_ji
        np.fill_diagonal(edge_weights, 0)
    return edge_weights  # index [i,j] shows how many prefer j over i


def generate_mixed_pairs_per_item(attribute_dict, n_candidates):
    mpair_dict = {}
    keys_list = list(attribute_dict) #in order to index attribute_dict
    for aval in range(len(attribute_dict)):
        items = attribute_dict[keys_list[aval]]
        mpair_cnt = len(items)*(n_candidates - len(items))
        for it in items:
            mpair_dict[it] = mpair_cnt
    return mpair_dict


def aggregate_rankings_fair_ilp(ranks, groups, thres_vals, inter_given):
    global attribute_dict
    if not inter_given:
        intersectional = make_intersectional_attribute(groups, True)
        groups = np.row_stack((groups, intersectional))
    n_voters, n_candidates = ranks.shape
    # construct
    pwin_cand = np.unique(ranks[0]).tolist()
    plose_cand = np.unique(ranks[0]).tolist()
    # convert pairwise wining/losing candidate index to string to index our variable
    plose_cand = [str(var) for var in plose_cand]
    pwin_cand = [str(var) for var in pwin_cand]
    cand = plose_cand

    # create a list of tuples containing all possible win row candidates and lose column candidates
    combos = [(i, j) for i in pwin_cand for j in plose_cand]

    # create a list of the precedence matrix representing the number of base rankings where a = row lost to b = col
    precedence_mat = all_pair_precedence(ranks)
    precedence_mat = precedence_mat.ravel()

    # create a dictionary to hold the weight for cand pair a and b, where cand a and cand b are keys and the #rankers put b above a is value (precedence mat)
    weight_dict = {}
    dur_iter = 0
    for (a, b) in combos:
        weight_dict[(a, b)] = precedence_mat[dur_iter]
        dur_iter = dur_iter + 1
    # Create the 'prob' variable to contain the problem data
    prob = pl.LpProblem("rank_agg", pl.LpMinimize)

    # Create the Xab variable
    #X = pl.LpVariable.dicts("X", (pwin_cand, plose_cand), 0, 1, pl.LpInteger)
    X = pl.LpVariable.dicts("X", (pwin_cand, plose_cand), 0, 1, cat= 'Integer')
    # Add the objective function
    prob += pl.lpSum(X[a][b] * weight_dict[(a, b)] for (a,b) in combos)

    # The strict ordering constraint
    for (a, b) in combos:
        if a != b:
            prob += pl.lpSum(X[a][b] + X[b][a]) == 1

    # The transitivity constraint
    for (a, b) in combos:
        if a != b:
            for c in cand:
                if c != a and c != b:
                    prob += pl.lpSum(X[a][b] + X[b][c] + X[c][a]) <= 2

    # Group Parity
    num_attributes = groups.shape[0] - 1

    for atr in range(0, num_attributes):

        thres = thres_vals[atr]
        atr = atr + 1
        attribute_dict = determine_group_identity(groups[0], groups[atr])
        mpair_dict = generate_mixed_pairs_per_item(attribute_dict, n_candidates)
        # in order to index dictionary
        keys_list = list(attribute_dict)

        #binary case
        if len(attribute_dict) == 2:
            mixed_pairs = [(i, j) for i in attribute_dict[keys_list[0]] for j in attribute_dict[keys_list[1]]]
            # add constraint
            prob += pl.lpSum(((1/mpair_dict[a])*X[a][b] - (1/mpair_dict[b])*X[b][a]) for (a, b) in mixed_pairs) <= thres
            prob += pl.lpSum((-(1/mpair_dict[a])*X[a][b] + (1/mpair_dict[b])*X[b][a]) for (a, b) in mixed_pairs) <= thres
        #multiclass case
        if len(attribute_dict) > 2:
            #get all size 2 combination of the groups
            combos = list(combinations(list(np.unique(groups[atr])), 2))
            for combo in combos:
                # get !cur_grp vals
                keys = [item for item in np.unique(groups[atr])] #key = group label
                groupa = [attribute_dict.get(key) for key in keys if key == combo[0]] #first group in combo
                not_groupa = [attribute_dict.get(key) for key in keys if key != combo[0]] #not first group in combo
                groupb = [attribute_dict.get(key) for key in keys if key == combo[1]] #second group in combo
                not_groupb = [attribute_dict.get(key) for key in keys if key != combo[1]]  # not second group in combo
                # flatten the lists of list
                groupa = [y for x in groupa for y in x]
                not_groupa = [y for x in not_groupa for y in x]
                groupb = [y for x in groupb for y in x]
                not_groupb = [y for x in not_groupb for y in x]
                # create mixed pairs of cur_group and other_grps
                groupa_pairs = [(i, j) for i in groupa for j in not_groupa]
                groupb_pairs = [(i, j) for i in groupb for j in not_groupb]


                # add constraint
                prob += (pl.lpSum((1/mpair_dict[a])*X[a][b] for (a, b) in groupa_pairs) - pl.lpSum(
                    (1/mpair_dict[c])*X[c][d] for (c,d) in groupb_pairs)) <= thres
                prob += (pl.lpSum(-(1 / mpair_dict[a]) * X[a][b] for (a, b) in groupa_pairs) + pl.lpSum(
                    (1 / mpair_dict[c]) * X[c][d] for (c, d) in groupb_pairs)) <= thres


    solver = pl.CPLEX_CMD(path = path_to_cplex, mip = True, options=['set mip tolerances integrality 0'])
    prob.solve(solver)
    prob.roundSolution()
    print("Status:", pl.LpStatus[prob.status])

    result = find_solution(prob, n_candidates)
    group_ids = groups[1,:]
    ranking_group_ids = [group_ids[np.argwhere(groups[0,:] == i)[0][0]] for i in result]

    return np.asarray(result), ranking_group_ids


