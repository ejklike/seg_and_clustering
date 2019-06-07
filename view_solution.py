import os
import utils

target_vin = '5NMS33AD0KH034994'
ws = 1
ld = 5e-3
bt = 400
nC = 10


# prefix_string: ./output_folder/vin/global/
basedir = 'output_folder/{}/global/'.format(target_vin)
prefix_string = basedir + 'ws={}ld={}bt={:0}'.format(ws, ld, bt)

# ./output_folder/vin/global/ld=#bt=#ws=#nc=#/solution.pkl
this_prefix_string = prefix_string + "nC=" + str(nC)
this_solution_path = this_prefix_string + "/solution.pkl"

print(this_solution_path)

### if: solution is already obtained ==> load solution
if os.path.exists(this_solution_path):
    # load solution
    _, ticc, iters, bic, cluster_MRFs, cluster_assignment_list = \
        utils.load_solution(this_solution_path)

    print('iters', iters)
    print('bic', bic)
    print('cluster_MRFs', cluster_MRFs)

else:
    print('No result.')