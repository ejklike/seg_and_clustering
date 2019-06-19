import os
import utils
from itertools import product

import numpy as np

# ws_list = [1, 2, 3]
# ld_list = [0.0005, 0.005, 0.05, 0.5]
# bt_list = [0, 10, 50, 100, 200]
# nC_list = range(3, 10+1)

ws_list = [3]
ld_list = [0.005]
bt_list = [200]
nC_list = [6]

def print_solution(ws, ld, bt, nC):
    print('{},{},{},{},'.format(ws, ld, bt, nC), end='')
    
    solution_path = ('output_folder/ws={}ld={}bt={:0}/nC={}/solution.pkl'.format(ws, ld, bt, nC))
    print(solution_path)
    if os.path.exists(solution_path):
        # load solution
        # *_, score_dict = utils.load_solution(solution_path)
        # print('{:.1f},{:.1f}'.format(-score_dict['nll'], score_dict['bic']))
        ticc, *_ = utils.load_solution(solution_path)
        for k, theta in ticc.theta_dict.items():
            print(k, '----')
            print(np.abs(theta[0:7, 7:14]) > 2e-5)
    else:
        print('na,na')

for ws, ld, bt, nC in product(ws_list, ld_list, bt_list, nC_list):
    print_solution(ws, ld, bt, nC)


# target_vin = '5NMS33AD0KH034994'
# ws = 1
# ld = 5e-2
# bt = 200
# nC = 7


# # prefix_string: ./output_folder/vin/global/
# basedir = 'output_folder/{}/global/'.format(target_vin)
# prefix_string = basedir + 'ws={}ld={}bt={:0}'.format(ws, ld, bt)

# # ./output_folder/vin/global/ld=#bt=#ws=#nc=#/solution.pkl
# this_prefix_string = prefix_string + "nC=" + str(nC)
# this_solution_path = this_prefix_string + "/solution.pkl"

# print('target file: ', this_solution_path)
# print('')

# ### if: solution is already obtained ==> load solution
# if os.path.exists(this_solution_path):
#     # load solution
#     ticc, output_dict, score_dict = utils.load_solution(this_solution_path)
    
#     print('''ws={}, ld={}, bt={}, nc={}
#     ll  = {:.1f}
#     BIC = {:.1f}
#     '''.format(ws, ld, bt, nC, -score_dict['nll'], score_dict['bic']))

# else:
#     print('No result.')