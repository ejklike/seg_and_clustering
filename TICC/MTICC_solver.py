import numpy as np
import pandas as pd
import math, time, collections, os, errno, sys, code, random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
from multiprocessing import Pool

from .src.TICC_helper import upperToFull, updateClusters
from .src.admm_solver import ADMMSolver


class MTICC:
    def __init__(self, 
                 window_size=10, 
                 number_of_clusters=5, 
                 lambda_parameter=11e-2,
                 beta=400, 
                 maxIters=1000, 
                 threshold=2e-5, 
                 verbose=True,
                 prefix_string="", 
                 num_proc=1):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - prefix_string: output directory if necessary
        """
        self.w = window_size
        self.K = number_of_clusters
        self.ld = lambda_parameter
        self.bt = beta

        self.maxIters = maxIters
        self.threshold = threshold
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.verbose = verbose
        self.cluster_reassignment_limit = 20
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

        # The basic folder to be created
        self.str_NULL = self.prefix_string + '/'

    def prepare_data(self, input_data_list):
        D_list = []
        for input_data in input_data_list:
            if isinstance(input_data, pd.core.frame.DataFrame):
                input_data = input_data.values

            # Stack the training data
            if self.w == 1:
                D = input_data
            elif self.w > 1:
                D_list_for_concatenate = []
                for i in range(self.w):
                    j = len(input_data) - self.w + 1 + i
                    D_list_for_concatenate.append(input_data[i:j, :])
                D = np.concatenate(D_list_for_concatenate, axis=1)
            else:
                raise ValueError('Window size must be greater or equal to 1.')
            D_list.append(D)

        # flat data
        D_flat = np.concatenate(D_list, axis=0)
        return D_list, D_flat


    def get_num_of_nonzero_params(self):
        if self.trained:
            nonzero_params = 0
            for k in range(self.K):
                nonzero_params += np.sum(np.abs(self.theta_dict[k]) > self.threshold)
        else:
            nonzero_params = None
        return nonzero_params


    def get_sum_of_nll(self, S_dict, P_dict=None):
        if self.trained:
            sum_nll = 0
            for k in range(self.K):
                log_det_theta_k = np.log(np.linalg.det(self.theta_dict[k]))
                tr_S_theta_k = np.trace(np.dot(S_dict[k], self.theta_dict[k]))
                if P_dict is not None:
                    sum_nll += len(P_dict[k]) * (tr_S_theta_k - log_det_theta_k)
                else:
                    sum_nll += tr_S_theta_k - log_det_theta_k
            return sum_nll / self.K
        else:
            return None
        # return sum_nll / np.sum([len(P_dict[k]) for k in range(self.K)]) # too small


    def compute_AIC(self, S_dict, P_dict=None):
         return (2 * self.get_num_of_nonzero_params() + 
                 2 * self.get_sum_of_nll(S_dict, P_dict=P_dict))


    def compute_BIC(self, T, S_dict, P_dict=None):
         return (np.log(T) * self.get_num_of_nonzero_params() + 
                 2 * self.get_sum_of_nll(S_dict, P_dict=P_dict))


    def get_score_dict(self, D_flat, cluster_assignment_flat, nll_flat):
        _, S_dict = self.get_mu_S_dict(D_flat, cluster_assignment_flat)
        P_dict = self.get_P_dict(cluster_assignment_flat)

        try:
            score_dict = {
                'T': len(D_flat),
                'aic': self.compute_AIC(S_dict, P_dict=P_dict),
                'bic': self.compute_BIC(len(D_flat), S_dict, P_dict=P_dict),
                'nll':  self.get_sum_of_nll(S_dict, P_dict=P_dict),
                'n_params': self.get_num_of_nonzero_params()}
        except:
            score_dict = {
                'T': None,
                'aic': None,
                'bic': None,
                'nll':  None,
                'n_params': self.get_num_of_nonzero_params()}

        return score_dict


    def get_nll_by_cluster_assignment(self, nll_array, cluster_assignment):
        return nll_array[np.arange(len(cluster_assignment)), cluster_assignment]


    def get_P_dict(self, cluster_assignment):
        """
        args: cluster_assignment list
        return: {cluster_number: [point indices]}
        """
        # {cluster: [point indices]}
        cluster_indices_dict = collections.defaultdict(list)  
        for k in range(self.K):
            cluster_indices_dict[k] = \
                [i for i, c in enumerate(cluster_assignment) 
                 if cluster_assignment[i]==k]
        return cluster_indices_dict


    def get_mu_S_dict(self, D, cluster_assignment):
        P_dict = self.get_P_dict(cluster_assignment)
        mu_dict, S_dict = {}, {}
        for k in range(self.K):
            indices = P_dict[k]
            D_k = D[indices, :]
            mu = np.mean(D_k, axis=0)
            S = np.cov(np.transpose(D_k))
            mu_dict[k] = mu
            S_dict[k] = S
        return mu_dict, S_dict


    def fit(self, input_data_list):
        """
        Main method for TICC solver.
        Parameters:
            - input_data: list of numpy array
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()
        self.prepare_out_directory()

        self.ncol = input_data_list[0].shape[1]
        self.probSize = self.w * self.ncol
        
        # data preparation according to window_size
        D_train_list, D_train_flat = self.prepare_data(input_data_list)

        # Initialization
        # Gaussian Mixture
        if self.verbose:
            print('#########################################')
            print('# Initialization: Gaussian Mixture --> P')
            print('#########################################')

        gmm = mixture.GaussianMixture(n_components=self.K, 
                                      covariance_type="full", random_state=102)
        cluster_assignment_flat = gmm.fit_predict(D_train_flat)
        P_flat_dict = self.get_P_dict(cluster_assignment_flat)
        self.mu_dict, self.S_dict = \
            self.get_mu_S_dict(D_train_flat, cluster_assignment_flat)
        self.theta_dict = None
        self.log_det_theta_dict = None

        if self.verbose:
            for k in range(self.K):
                print("length of the cluster ", k, "--->", len(P_flat_dict[k]))

        old_cluster_assignment_flat = None  # points from last iteration

        # PERFORM TRAINING ITERATIONS
        # print('start multi-threading')
        pool = Pool(processes=self.num_proc)  # multi-threading
        try:
            output_dict = {'cluster_assignment': None,
                           'nll_vector': None}
            score_dict = {'T': None,
                          'aic': None,
                          'bic': None,
                          'nll':  None,
                          'n_params': None}

            for iters in range(self.maxIters):
                
                self.iters = iters
                if self.verbose:
                    print("\n\n\nITERATION ###", iters)

                if self.verbose:
                    print('#############################################')
                    print('# M-step: Update cluster parameters --> theta')
                    print('#############################################')

                # calculate theta(inv_upper_cov), emp_cov, emp_mean
                upper_theta_dict = self.train_clusters(D_train_flat, P_flat_dict, pool)

                # calculate theta and its inverse
                self.theta_dict, self.log_det_theta_dict = \
                    self.update_theta(upper_theta_dict)

                if self.verbose:
                    print('##########################################')
                    print('# E-step: Assign points to clusters --> P ')
                    print('##########################################')
            
                cluster_assignment_list, nll_array_list, nll_vector_list = \
                    self.get_pointwise_prediction(D_train_list)
                cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
                nll_flat = np.concatenate(nll_array_list, axis=0)
                P_flat_dict = self.get_P_dict(cluster_assignment_flat)

                if self.verbose:
                    for k in range(self.K):
                        print("length of the cluster ", k, "--->", len(P_flat_dict[k]))

                need_adjustment = any([len(P) == 0 for P in P_flat_dict.values()])
                if need_adjustment: 
                    if self.verbose:
                        print('-------- cluster reassignment for the empty cluster')
                        print('')
                    cluster_assignment_list = \
                        self.adjust_cluster_assignment(cluster_assignment_list, 
                                                       P_flat_dict)
                    cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
                    P_flat_dict = self.get_P_dict(cluster_assignment_flat)

                    if self.verbose:
                        for k in range(self.K):
                            print("length of the cluster ", k, "--->", len(P_flat_dict[k]))


                self.mu_dict, self.S_dict = \
                    self.get_mu_S_dict(D_train_flat, cluster_assignment_flat)

                self.write_plot(cluster_assignment_flat, iters)

                if np.array_equal(old_cluster_assignment_flat, cluster_assignment_flat):
                    if self.verbose:
                        print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!\n\n\n\n")
                    break

                old_cluster_assignment_flat = cluster_assignment_flat.copy()
            
            output_dict = {'cluster_assignment': cluster_assignment_list,
                           'nll_vector': nll_vector_list}
            score_dict = self.get_score_dict(D_train_flat, 
                                            cluster_assignment_flat, 
                                            nll_flat)

        except np.linalg.LinAlgError as err:
            if self.verbose:
                print('')
                print('--> np.linalg.LinAlgError occurred ({})! '
                      'terminate at iteration {}.'.format(err, iters))
                print('')
        except KeyError as err:
            if self.verbose:
                print('')
                print('--> There is an empty cluster ({})! '
                      'Error occurred at iteration {}.'.format(err, iters))
                print('')

        if pool is not None:
            pool.close()
            pool.join()

        return output_dict, score_dict


    def adjust_cluster_assignment(self, cluster_assignment_list, P_flat_dict):

        cov_dict = {k: np.linalg.inv(theta) for k, theta in self.theta_dict.items()}
        cluster_norms = [(np.linalg.norm(cov_dict[k]), k) for k in range(self.K)]
        norms_sorted = sorted(cluster_norms, reverse=True)

        # clusters that are not 0 as sorted by norm
        valid_clusters = [k for cov, k in norms_sorted if len(P_flat_dict[k]) != 0]
        invalid_clusters = [k for k in range(self.K) if k not in valid_clusters]
        if self.verbose:
            # print('cluster_norms', cluster_norms)
            # print('norms_sorted', norms_sorted)
            print('valid_clusters', valid_clusters)
            print('invalid_clusters', invalid_clusters)

        # Add a point to the empty clusters
        # assuming more non empty clusters than empty ones
        if len(invalid_clusters) > 0:
            # index_list = [np.ones_like(ca) * i for i, ca in 
            #               enumerate(cluster_assignment_list)]
            # index_flat = np.concatenate(index_list, axis=0)
            P_dict_list = [self.get_P_dict(ca) for ca in cluster_assignment_list]
            # P_valid_dict_list = [{k: P_k for k, P_k in P_dict.items() if len(P_k) > 0}
            #                      for P_dict in P_dict_list]
            P_valid_dict = {
                k: [(i, P_dict[k]) for i, P_dict in enumerate(P_dict_list) 
                    if len(P_dict[k]) > 0] for k in range(self.K)}
            # for k in range(self.K):
            #     print(k, len(P_valid_dict[k]))

            counter = 0
            for k_invalid in invalid_clusters:
                # select a cluster that is not len 0
                k_selected = valid_clusters[counter]
                counter = (counter + 1) % len(valid_clusters)
                if self.verbose:
                    print("empty cluster:", k_invalid, 
                          "--> selected cluster instead is:", k_selected)

                # choose P randomly from that cluster
                n_segments_of_selected_cluster = len(P_valid_dict[k_selected])
                if n_segments_of_selected_cluster == 1:
                    target_P_index = 0
                    index, P = P_valid_dict[k_selected][target_P_index]
                    if len(P) > 1:
                        P = P[:len(P)//2]
                    else:
                        return cluster_assignment_list

                    print('--> the chosen P (len={}) is from the {}-th time-series'
                            .format(len(P), index))
                    
                else:
                    target_P_index = np.random.choice(n_segments_of_selected_cluster)
                    index, P = P_valid_dict[k_selected].pop(target_P_index)
                    print('--> the chosen P (len={}) is from the {}-th time-series'
                            .format(len(P), index))
                cluster_assignment_list[index][P] = k_invalid
        
        return cluster_assignment_list

    def get_pointwise_nll_array(self, D_train):
        # For each point compute the LLE
        # return array with size [T, K]
        T = len(D_train)
        NLL = np.zeros([T, self.K])
        for k in range(self.K):
            mu_k = self.mu_dict[k]
            theta_k = self.theta_dict[k]
            log_det_theta_k = self.log_det_theta_dict[k]
            for t in range(T):
                X2 = (D_train[t, :] - mu_k).reshape([self.probSize, 1])
                NLL[t, k] = np.dot(X2.T, np.dot(theta_k, X2)) - log_det_theta_k
        return NLL


    def get_pointwise_prediction(self, D_list):
        cluster_assignment_list = []
        nll_array_list = []
        nll_vector_list = []
        for D in D_list:
            nll_array = self.get_pointwise_nll_array(D)
            nll_array_list.append(nll_array)

            cluster_assignment = updateClusters(nll_array, beta=self.bt)
            cluster_assignment_list.append(cluster_assignment)

            nll_vector = self.get_nll_by_cluster_assignment(nll_array, 
                                                            cluster_assignment)
            nll_vector_list.append(nll_vector)
        
        return cluster_assignment_list, nll_array_list, nll_vector_list

    @property
    def trained(self):
        return (
            self.theta_dict is not None and 
            len(self.theta_dict) == self.K)


    def test(self, input_data_list):
        assert self.ncol == input_data_list[0].shape[1]
        
        verbose_backup = self.verbose
        self.verbose = False

        if self.trained:

            # data preparation according to window_size
            D_test_list, D_test_flat = self.prepare_data(input_data_list)

            cluster_assignment_list, nll_array_list, nll_vector_list = \
                self.get_pointwise_prediction(D_test_list)

            output_dict = {'cluster_assignment': cluster_assignment_list,
                           'nll_vector': nll_vector_list}

            cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
            nll_flat = np.concatenate(nll_array_list, axis=0)

            score_dict = self.get_score_dict(D_test_flat, 
                                             cluster_assignment_flat, 
                                             nll_flat)
            if self.verbose:
                for k in range(self.K):
                    print("length of cluster #", k, "-------->", 
                            sum([x == k for x in cluster_assignment_flat]))

            self.write_plot(cluster_assignment_flat, 'test')

        else:
            output_dict = {'cluster_assignment': None,
                           'nll_vector': None}
            score_dict = {'T': None,
                          'aic': None,
                          'bic': None,
                          'nll':  None,
                          'n_params': None}

        self.verbose = verbose_backup
        return output_dict, score_dict


    def write_plot(self, cluster_assignment, iters):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(cluster_assignment, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.K + 0.5))
        plt.xlabel('time index')
        plt.ylabel('cluster label')
        if isinstance(iters, int):
            fname_fig = self.str_NULL + "TRAINING_EM_iter={:000}.jpg".format(iters)
        if isinstance(iters, str):
            fname_fig = self.str_NULL + "TRAINING_EM_iter={}.jpg".format(iters)
        if self.verbose:
            print(fname_fig)
        plt.savefig(fname_fig)
        plt.close()
        if self.verbose:
            print("Done writing the figure")


    def update_theta(self, upper_theta_dict):
        theta_dict = {}
        log_det_theta_dict = {}

        for k in range(self.K):
            if upper_theta_dict[k] == None:
                continue
            # from multiprocessing.pool.ApplyResult to its value
            upper_theta = upper_theta_dict[k].get()
            if self.verbose:
                print("OPTIMIZATION for Cluster #", k, "DONE!!!")
            # THIS IS THE SOLUTION
            theta = upperToFull(upper_theta, 0) # full theta (inv covariance)
            log_det_theta = np.log(np.linalg.det(theta))
            # update value
            theta_dict[k] = theta
            log_det_theta_dict[k] = log_det_theta
        return theta_dict, log_det_theta_dict


    def train_clusters(self, D_train, P_dict, pool):
        """
        return:
            - empirical covariance
            - empirical mean
            - upper_theta_dict
        """
        upper_theta_dict = {k:None for k in range(self.K)}

        for k in range(self.K):
            P_k = P_dict[k]
            if len(P_k) != 0:
                # training data for this cluster
                D_train_c = D_train[P_k, :]
                
                ##Fit a model - OPTIMIZATION
                lamb = self.ld * np.ones((self.probSize, self.probSize))
                S = np.cov(np.transpose(D_train_c))
                rho = 1
                solver = ADMMSolver(lamb, self.w, self.ncol, rho, S)
                # apply to process pool (args: maxIters, eps_abs, eps_rel, verbose)
                pool_result = pool.apply_async(solver, (1000, 1e-6, 1e-6, False))
                upper_theta_dict[k] = pool_result
        return upper_theta_dict


    def prepare_out_directory(self):
        if not os.path.exists(os.path.dirname(self.str_NULL)):
            try:
                os.makedirs(os.path.dirname(self.str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

    def log_parameters(self):
        if self.verbose:
            print('\n\n')
            print("lambda_sparse", self.ld)
            print("switch_penalty", self.bt)
            print("num_cluster", self.K)
            print("window_size", self.w)