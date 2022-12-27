"""
Ref: Singh et al. 2018
Author: Kcachel
"""
import numpy as np

def calc_exposure_ratio(ranking, group_ids):

    unique_grps, grp_count_items = np.unique(group_ids, return_counts=True)
    num_items = len(ranking)
    exp_vals = exp_at_position_array(num_items)
    grp_exposures = np.zeros_like(unique_grps, dtype=np.float64)
    for i in range(0,num_items):
        grp_of_item = group_ids[i]
        exp_of_item = exp_vals[i]
        #update total group exp
        grp_exposures[grp_of_item] += exp_of_item

    avg_exp_grp = grp_exposures / grp_count_items
    #expdp = np.max(avg_exp_grp) - np.min(avg_exp_grp)
    expdpp = np.min(avg_exp_grp)/np.max(avg_exp_grp) #ratio based
    #print("un-normalized expdp: ", expdp)
    #norm_result = expdp / normalizer
    return expdpp, avg_exp_grp

def exp_at_position_array(num_items):
    return np.array([(1/(np.log2(i+1))) for i in range(1,num_items+1)])


#
# #Testing
# ranking = np.asarray([0,1, 2, 3, 4, 5, 6, 7])
# group_ids = np.asarray([0,0,0,0, 0, 0,1,1])
# # ranking = np.asarray([0,1, 2, 3, 4, 5, 6, 7])
# # group_ids = np.asarray([1, 1, 0,0,0,0, 0, 0])
#
# expdp = calculate_expdp_singh(ranking, group_ids)
#
# print("expdp for ranking, with group ids ", group_ids, "is value: ", expdp )
#
# #paper example
# # men = np.sum(exp_list[0:3])/3
# # men
# # Out[3]: 0.7103099178571526
# # female = np.sum(exp_list[3:6])/3
# # female
# # Out[5]: 0.3912455174719856
# # men - female
# # Out[6]: 0.319064400385167