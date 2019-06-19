"""

# train.py
---
main module for M-TICC


## USAGE STATEMENT

'''
>> python train.py -h

usage: train.py [-h] [--verbose [VERBOSE]] [--num_proc NUM_PROC] [--ld LD]
                [--bt BT] [--ws WS] [--maxiter MAXITER] [--threshold THRESHOLD]
                [--min_nc MIN_NC] [--max_nc MAX_NC] [--test-size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --verbose [VERBOSE]   verbose for TICC
  --num_proc NUM_PROC   the number of threads
  --ld LD               lambda (sparsity in Toplitz matrix)
  --bt BT               beta (segmentation penalty)
  --ws WS               window size
  --maxiter MAXITER     maxiter
  --threshold THRESHOLD
                        threshold
  --min_nc MIN_NC       min_num_cluster
  --max_nc MAX_NC       max_num_cluster
  --test-size TEST_SIZE
                        test data size
'''

## USAGE EXAMPLES

python train.py --min_nc 3 --max_nc 3 --ws 1
python train.py --verbose --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
python train.py --verbose --test_size 0 --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
"""

import time
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import utils
import dataio
from viz import draw_bic_plot
from TICC.MTICC_solver import MTICC


# PERFORMANCE MEASURE
VALIDATION_KEY = 'bic'
assert VALIDATION_KEY in ['nll', 'aic', 'bic']
INIT_SCORE_DICT = {
    'T': None,
    'aic': 1e10,
    'bic': 1e10,
    'nll':  1e10,
    'n_params': None}


def run_ticc(data_for_modeling, number_of_clusters, prefix_string, args):
    print('run a new TICC model ...', end='')
    start = time.time()
    ticc = MTICC(window_size=args.ws, 
                 number_of_clusters=number_of_clusters, 
                 lambda_parameter=args.ld, 
                 beta=args.bt, 
                 maxIters=args.maxiter, 
                 threshold=args.threshold,
                 prefix_string=prefix_string, 
                 num_proc=args.num_proc,
                 verbose=args.verbose)
    output_dict, score_dict = ticc.fit(data_for_modeling)
    end = time.time()
    print(' ---> elapsed: {:.1f}min'.format((end - start)/60))
    return ticc, output_dict, score_dict


def loop_ticc_modeling(trn_data_list, tst_data_list, args):
    """
    - loop for all number_of_clusters to find the best
    - return ticc and training_output
    """
    trn_bic_list = []
    tst_bic_list = []

    # find the best num of clusters based on bic score
    for number_of_clusters in range(args.min_nc, args.max_nc+1):

        # ./output_folder/vin/global/ld=#bt=#ws=#/nc=#/solution.pkl
        prefix_string = args.basedir + "nC=" + str(number_of_clusters)
        utils.maybe_exist(prefix_string)
        this_solution_path = prefix_string + "/solution.pkl"

        print('nc={}, '.format(number_of_clusters), end='')
        if os.path.exists(this_solution_path):
            print('load solution stored in local ...')
            ticc, output_dict_trn, score_dict_trn = \
                utils.load_solution(this_solution_path)
        else:
            ticc, output_dict_trn, score_dict_trn = \
                run_ticc(trn_data_list, number_of_clusters, prefix_string, args)
            utils.dump_solution(ticc, 
                                output_dict_trn, 
                                score_dict_trn, 
                                this_solution_path)

        # test
        _, score_dict_tst = ticc.test(tst_data_list)
        trn_bic_list.append(score_dict_trn)
        tst_bic_list.append(score_dict_tst)
        exist_solution = (
            score_dict_trn[VALIDATION_KEY] is not None and 
            score_dict_tst[VALIDATION_KEY] is not None)
        if exist_solution:
            print(' ---> iter={} | {}: trn {:.1f}, tst {:.1f}'
                .format(ticc.iters, 
                        VALIDATION_KEY,
                        score_dict_trn[VALIDATION_KEY], 
                        score_dict_tst[VALIDATION_KEY]), end='')
        else:
            print(' ---> iter={} | No solution.'.format(ticc.iters, end=''))
        print('')

    print('')
    print('Training finished.')
    if args.max_nc - args.min_nc > 0:
        draw_bic_plot(trn_bic_list, tst_bic_list, args)


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--verbose', 
        type=bool, 
        nargs='?',
        default=False, #default
        const=True, #if the arg is given
        help='verbose for TICC')
    parser.add_argument(
        '--num_proc', 
        type=int, 
        default=4,
        help='the number of threads')

    parser.add_argument(
        '--ld', 
        type=float, 
        default=5e-3,
        help='lambda (sparsity in Toplitz matrix)')
    parser.add_argument(
        '--bt', 
        type=int, 
        default=200,
        help='beta (segmentation penalty)')
    parser.add_argument(
        '--ws', 
        type=int, 
        default=1,
        help='window size')

    parser.add_argument(
        '--maxiter', 
        type=int, 
        default=100,
        help='maxiter')
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=2e-5,
        help='threshold')
    parser.add_argument(
        '--min_nc', 
        type=int, 
        default=3,
        help='min_num_cluster')
    parser.add_argument( # max 20
        '--max_nc', 
        type=int, 
        default=10,
        help='max_num_cluster')

    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.3,
        help='test data size')

    # Parse input arguments
    args, unparsed = parser.parse_known_args()

    print('')
    print("lambda (sparsity penalty): ", args.ld)
    print("beta (switching penalty):  ", args.bt)
    print("w (window size):           ", args.ws)
    print("K (number of cluster):      {} ~ {}".format(args.min_nc, 
                                                       args.max_nc))

    # basedir: ./output_folder/vin/parameters/
    args.basedir = utils.get_basedir(args)
    utils.maybe_exist(args.basedir)
    
    print('')
    print('All results will be saved to:', args.basedir)
    print('')

    ################################################
    # - Load rawdata (containing all trips)
    # - Parse the data into 'ign_on_time' unit
    # - Split the set of unit data into training/test sets
    ################################################
    # If you want to CHANGE the dataio logics, 
    # all you have to do is to MODIFY "dataio.py".
    ################################################
    dataloader = dataio.DataLoader()
    df = dataloader.load_rawdata_containing_all_trips()

    # In this data, each trip can be 
    # distinguished by 'ign_on_time' field.
    ign_on_time_list = list(df.ign_on_time.unique())
    
    # For test, we will choose only 100 trip samples.
    # If you do not want to test, please comment this line.
    ign_on_time_list = ign_on_time_list[:100] #<>#

    # split all trips into trn/tst sets
    trn_ign_on_time_list, tst_ign_on_time_list = \
        dataio.get_data_split_keys(ign_on_time_list, 
                                   test_size=args.test_size)
    
    # get trn/tst trip data
    trn_data_list, tst_data_list, _, _ = \
        dataio.get_data_and_path_list_from_split(df, 
                                                 trn_ign_on_time_list, 
                                                 tst_ign_on_time_list)

    #################
    # FIND SOLUTION
    #################

    # Given lambda, beta, and window size,
    # we loop model training from min_nc to max_nc,
    # and all the models are stored in 'basedir' directory.
    print('> Find soultion ...')
    loop_ticc_modeling(trn_data_list, tst_data_list, args)