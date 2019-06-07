import numpy as np
import math, time, collections, os, errno, sys, code, random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from multiprocessing import Pool

from .src.TICC_helper import upperToFull, updateClusters
from .src.admm_solver import ADMMSolver

criterion = None


class GlobalTICC:
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
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

        # The basic folder to be created
        self.str_NULL = self.prefix_string + '/'


    def prepare_data(self, input_data_list):
        D_list = []
        for input_data in input_data_list:
            # Stack the training data
            if self.w == 1:
                D = input_data
            if self.w == 2:
                D = np.concatenate([input_data[:-1], input_data[1:]], axis=1)

            D_list.append(D)
        
        # flat data
        D_flat = np.concatenate(D_list, axis=0)
        return D_list, D_flat


    def get_num_of_nonzero_params(self):
        nonzero_params = 0
        for k in range(self.K):
            nonzero_params += np.sum(np.abs(self.theta_dict[k]) > self.threshold)
        return nonzero_params


    def get_sum_of_log_likelihood(self, S_dict):
        mod_lle = 0
        for k in range(self.K):
            log_det_theta_k = np.log(np.linalg.det(self.theta_dict[k]))
            tr_S_theta_k = np.trace(np.dot(S_dict[k], self.theta_dict[k]))
            mod_lle += log_det_theta_k - tr_S_theta_k
        return mod_lle


    def compute_AIC(self, S_dict):
         return (2 * self.get_num_of_nonzero_params() - 
                 2 * self.get_sum_of_log_likelihood(S_dict))


    def compute_BIC(self, T, S_dict):
         return (np.log(T) * self.get_num_of_nonzero_params() - 
                 2 * self.get_sum_of_log_likelihood(S_dict))


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

        ############
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
        self.mu_dict, self.S_dict = \
            self.get_mu_S_dict(D_train_flat, cluster_assignment_flat)

        P_flat_dict = self.get_P_dict(cluster_assignment_flat)
        if self.verbose:
            for k in range(self.K):
                print("length of the cluster ", k, "--->", len(P_flat_dict[k]))

        old_cluster_assignment_flat = None  # points from last iteration

        # PERFORM TRAINING ITERATIONS
        # print('start multi-threading')
        pool = Pool(processes=self.num_proc)  # multi-threading
        try:
            for iters in range(self.maxIters):
                if self.verbose:
                    print("\n\n\nITERATION ###", iters)
                else:
                    print("\r### nC={} ... ITERATION {:000} ..."
                          .format(self.K, iters), end='')

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

                cluster_assignment_list = []
                for D_train in D_train_list:
                    # Update cluster points
                    log_likelihood_array = self.get_pointwise_log_likelihood(D_train)
                    cluster_assignment = updateClusters(log_likelihood_array, 
                                                        beta=self.bt)
                    cluster_assignment_list.append(cluster_assignment)
                
                cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
                self.mu_dict, self.S_dict = \
                    self.get_mu_S_dict(D_train_flat, cluster_assignment_flat)

                P_flat_dict = self.get_P_dict(cluster_assignment_flat)
                if self.verbose:
                    for k in range(self.K):
                        print("length of the cluster ", k, "--->", len(P_flat_dict[k]))

                self.write_plot(cluster_assignment_flat, iters)

                if np.array_equal(old_cluster_assignment_flat, cluster_assignment_flat):
                    if self.verbose:
                        print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                    break

                old_cluster_assignment_flat = cluster_assignment_flat.copy()
            
        except np.linalg.LinAlgError as err:
            if pool is not None:
                pool.close()
                pool.join()

            if not self.verbose:
                print('')
            
            cluster_assignment_list, theta_dict = None, None

            score_dict = {
                'T': len(D_train_flat),
                'aic': None,
                'bic': None,
                'll':  None,
                'n_params': None}

            return cluster_assignment_list, theta_dict, iters, score_dict
            
        if pool is not None:
            pool.close()
            pool.join()
            
            if not self.verbose:
                print('')

        score_dict = {
            'T': len(D_train_flat),
            'aic': self.compute_AIC(self.S_dict),
            'bic': self.compute_BIC(len(D_train_flat), self.S_dict),
            'll':  self.get_sum_of_log_likelihood(self.S_dict),
            'n_params': self.get_num_of_nonzero_params()}

        return cluster_assignment_list, self.theta_dict, iters, score_dict

    def test(self, input_data_list):

        assert self.ncol == input_data_list[0].shape[1]
        
        verbose_backup = self.verbose
        self.verbose = False

        if self.theta_dict is not None:

            # data preparation according to window_size
            D_test_list, D_test_flat = self.prepare_data(input_data_list)

            cluster_assignment_list = []
            for D_test in D_test_list:
                log_likelihood_array = self.get_pointwise_log_likelihood(D_test)
                cluster_assignment = updateClusters(log_likelihood_array, 
                                                    beta=self.bt)
                cluster_assignment_list.append(cluster_assignment)

            cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)

            D_test_flat = np.concatenate(D_test_list, axis=0)
            _, test_S_dict = self.get_mu_S_dict(D_test_flat, cluster_assignment_flat)
            
            score_dict = {
                'T': len(D_test_flat),
                'aic': self.compute_AIC(test_S_dict),
                'bic': self.compute_BIC(len(D_test_flat), test_S_dict),
                'll':  self.get_sum_of_log_likelihood(test_S_dict),
                'n_params': self.get_num_of_nonzero_params()}
            
            if self.verbose:
                for k in range(self.K):
                    print("length of cluster #", k, "-------->", 
                            sum([x == k for x in cluster_assignment_flat]))

            self.write_plot(cluster_assignment_flat, 'test')

        else:
            cluster_assignment_list = None
            score_dict = {
                'T': len(D_test_flat),
                'aic': None,
                'bic': None,
                'll':  None,
                'n_params': None}

        self.verbose = verbose_backup
        return cluster_assignment_list, score_dict


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
            print("lambda_sparse", self.ld)
            print("switch_penalty", self.bt)
            print("num_cluster", self.K)
            print("window_size", self.w)


    def get_pointwise_log_likelihood(self, D_train):
        # For each point compute the LLE
        # return array with size [T, K]
        T = len(D_train)
        log_likelihood = np.zeros([T, self.K])
        for t in range(T - self.w + 1):
            for k in range(self.K):
                mu_k = self.mu_dict[k]
                theta_k = self.theta_dict[k]
                log_det_theta_k = self.log_det_theta_dict[k]
                
                X2 = (D_train[t, :] - mu_k).reshape([self.probSize, 1])
                lle = log_det_theta_k - np.dot(X2.T, np.dot(theta_k, X2))
                log_likelihood[t, k] = lle
        return log_likelihood


    # def predict_clusters(self, test_data):
    #     '''
    #     Given the current trained model, predict clusters.  
    #     If the cluster segmentation has not been optimized yet,
    #     than this will be part of the interative process.

    #     Args:
    #         numpy array of data for which to predict clusters.  
    #         Columns are dimensions of the data, each row is
    #         a different timestamp

    #     Returns:
    #         vector of predicted cluster for the points
    #     '''
    #     if not isinstance(test_data, np.ndarray):
    #         raise TypeError("input must be a numpy array!")

    #     # SMOOTHENING
    #     log_likelihood_array = self.get_pointwise_log_likelihood(test_data)

    #     # Update cluster points - using NEW smoothening
    #     cluster_assignment = updateClusters(log_likelihood_array, beta=self.bt)
    #     log_likelihood = \
    #         [log_likelihood_array[i, c] for i, c in enumerate(cluster_assignment)]
    #     return cluster_assignment, log_likelihood
