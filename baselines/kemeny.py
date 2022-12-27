import numpy as np
from itertools import combinations, permutations
import gurobipy as gp
from gurobipy import GRB
from src.utils import *
from collections import Counter
from more_itertools import unique_everseen

def kemeny(base_ranks, item_ids, group_ids):


    # Declare and initialize model
    m = gp.Model('KemenyConsensus')

    # Create decision variable (every pair)
    num_voters, num_items = np.shape(base_ranks)
    items = np.unique(base_ranks[0]).tolist()
    item_strings = [str(var) for var in items]

    pair_combinations = [(i, j) for i in item_strings for j in item_strings] #all possible
    x = m.addVars(pair_combinations, name="pair", vtype=gp.GRB.BINARY)
    m.addConstrs((x[r, r] == 0 for r in item_strings), name="zeroselfpairs")


    # Ordering constraint Xab +Xba = 1
    print("making strict ordering.....")
    unique_pairs = [(str(i), str(j)) for i, j in combinations(range(np.shape(base_ranks)[1]), 2)] #count 01, and 10 only once
    m.addConstrs((x[a,b]+x[b,a] == 1 for a,b in unique_pairs if a != b), name='strict_order')
    m.addConstrs((x[a, b] + x[b, a] == 1 for a, b in pair_combinations if a != b), name='strict_order')
    # Prevent cycles constraint
    print("making cycle prevention.....")
    m.addConstrs((x[a, b] + x[b, c] + x[c,a] <= 2 for a, b in unique_pairs if a != b for c in item_strings if
                  c != a and c != b), name='stopcycles')

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




    # Save model for inspection
    #m.write('Kemeny.lp')
    m.setParam('MIPGAP',.015)
    print("starting optimization.....")
    # Run optimization engine
    m.optimize()

    # Display optimal values of decision variables
    # for v in m.getVars():
    #     if v.x > 1e-6:
    #         print(v.varName, v.x)

    rank_pairs = [var.varName for var in m.getVars() if var.x == 1 and var.varName.startswith('pair')]
    winning_items = [(var.split(',')[0]).split('[')[1] for var in rank_pairs]
    result = [item for items, c in Counter(winning_items).most_common()
              for item in [items] * c]
    kemeny = list(unique_everseen(result))
    kemeny = list(map(int, kemeny))
    bottom_candidate = [item for item in range(0, num_items) if item not in kemeny]
    kemeny.append(bottom_candidate[0])
    ranking_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in kemeny]

    return np.asarray(kemeny), ranking_group_ids

