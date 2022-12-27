import numpy as np

def precedence_matrix_disagreement(baseranks):
    """
    :param baseranks: num_rankers x num_items
    :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # disagreements with i over j
    """
    num_rankers, num_items = baseranks.shape


    weight = np.zeros((num_items, num_items))

    pwin_cand = np.unique(baseranks[0]).tolist()
    plose_cand = np.unique(baseranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(num_rankers):
            if np.argwhere(baseranks[r] == i)[0][0] > np.argwhere(baseranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        weight[i, j] = h_ij
        weight[j, i] = h_ji
        np.fill_diagonal(weight, 0)
    return weight



def precedence_matrix_agreement(baseranks):
    """
    :param baseranks: num_rankers x num_items
    :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # agreements with i over j
    """
    num_rankers, num_items = baseranks.shape


    weight = np.zeros((num_items, num_items))

    pwin_cand = np.unique(baseranks[0]).tolist()
    plose_cand = np.unique(baseranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(num_rankers):
            if np.argwhere(baseranks[r] == i)[0][0] < np.argwhere(baseranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        weight[i, j] = h_ij
        weight[j, i] = h_ji
        np.fill_diagonal(weight, 0)
    return weight