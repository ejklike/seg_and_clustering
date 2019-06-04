import numpy as np
import math, time, collections, os, errno, sys, code, random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from multiprocessing import Pool

from .src.TICC_helper import upperToFull, computeBIC, updateClusters
from .src.admm_solver import ADMMSolver


class TICC:
    def __init__(self, 
                 window_size=10, 
                 number_of_clusters=5, 
                 lambda_parameter=11e-2,
                 beta=400, 
                 maxIters=1000, 
                 threshold=2e-5, 
                 write_out_file=False, 
                 verbose=True,
                 prefix_string="", 
                 num_proc=1, 
                 compute_BIC=False, 
                 cluster_reassignment=20):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
        """
        self.w = window_size
        self.K = number_of_clusters
        self.ld = lambda_parameter
        self.bt = beta

        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.verbose = verbose
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

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

    def fit(self, input_data):
        """
        Main method for TICC solver.
        Parameters:
            - input_data: numpy array
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()
        self.str_NULL = str_NULL

        # Get data into proper format
        self.nrow, self.ncol = input_data.shape
        self.probSize = self.w * self.ncol

        # Train test split
        training_indices = np.arange(self.nrow - self.w + 1)

        # Stack the training data
        if self.w == 1:
            D_train = input_data
        if self.w == 2:
            D_train = np.concatenate([input_data[:-1], input_data[1:]], axis=1)
            # D_train[-1, 8:] = 0.

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.K, 
                                      covariance_type="full", random_state=102)
        cluster_assignment = gmm.fit_predict(D_train)
        old_cluster_assignment = None  # points from last iteration

        # print(cluster_assignment)
        # P_dict = self.get_P_dict(cluster_assignment)
        # for k in range(self.K):
        #     print("length of the cluster ", k, "--->", len(P_dict[k]))
        # return None

        self.mu_dict = {}
        self.S_dict = {}
        self.theta_dict = {}
        self.log_det_theta_dict = {}
        self.S_est_dict = {}

        # PERFORM TRAINING ITERATIONS
        # print('start multi-threading')
        pool = Pool(processes=self.num_proc)  # multi-threading
        try:
            for iters in range(self.maxIters):
                if self.verbose:
                    print("\n\n\nITERATION ###", iters)

                P_dict = self.get_P_dict(cluster_assignment)

                if self.verbose:
                    print('#############################################')
                    print('# M-step: Update cluster parameters --> theta')
                    print('#############################################')

                # calculate theta(inv_upper_cov), emp_cov, emp_mean
                upper_theta_dict = self.train_clusters(D_train, P_dict, pool)

                # calculate theta and its inverse
                self.optimize_clusters(upper_theta_dict)

                if self.verbose:
                    for k in range(self.K):
                        print("length of the cluster ", k, "--->", len(P_dict[k]))

                if self.verbose:
                    print('##########################################')
                    print('# E-step: Assign points to clusters --> P ')
                    print('##########################################')

                cluster_assignment = self.predict_clusters(D_train)
                new_P_dict = self.get_P_dict(cluster_assignment)

                ##########################################

                # adjust empty clusters
                # before_empty_cluster_assign = cluster_assignment.copy()

                if iters != 0:
                    cluster_norms = \
                        [(np.linalg.norm(self.S_est_dict[k]), k) for k in range(self.K)]
                    norms_sorted = sorted(cluster_norms, reverse=True)
                    print('cluster_norms', cluster_norms)
                    print('norms_sorted', norms_sorted)
                    # clusters that are not 0 as sorted by norm
                    valid_clusters = [k for n, k in norms_sorted if len(new_P_dict[k]) != 0]
                    empty_clusters = [k for k in range(self.K) if len(new_P_dict[k]) == 0]
                    print('valid_clusters', valid_clusters)
                    print('empty_clusters', empty_clusters)

                    # Add a point to the empty clusters
                    # assuming more non empty clusters than empty ones
                    counter = 0
                    for k_empty in empty_clusters:
                        k_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        if self.verbose:
                            print("cluster that is zero is:", k_empty, 
                                  "selected cluster instead is:", k_selected)
                        
                        # random point number from that cluster
                        t_start = np.random.choice(new_P_dict[k_selected])
                        t_end = np.minimum(t_start + self.cluster_reassignment - 1, 
                                           len(cluster_assignment) - 1)
                        t_reassignment = np.arange(t_start, t_end + 1)
                        print(t_reassignment)
                        cluster_assignment[t_reassignment] = k_empty
                        self.S_est_dict[k_empty] = self.S_est_dict[k_selected]
                        self.mu_dict[k_empty] = np.mean(D_train[t_reassignment, :], axis=0)

                if self.verbose:
                    for k in range(self.K):
                        print("length of cluster #", k, "-------->", 
                              sum([x == k for x in cluster_assignment]))

                self.write_plot(cluster_assignment, str_NULL, training_indices, iters)

                if np.array_equal(old_cluster_assignment, cluster_assignment):
                    if self.verbose:
                        print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                    break
                # old_cluster_assignment = before_empty_cluster_assign
                old_cluster_assignment = cluster_assignment.copy()
                # end of training
        except np.linalg.LinAlgError as err:
            if pool is not None:
                pool.close()
                pool.join()
            cluster_assignment, theta_dict, bic = None, None, -1e10
            return cluster_assignment, theta_dict, bic
            
        if pool is not None:
            pool.close()
            pool.join()

        # return
        if self.compute_BIC:
            bic = computeBIC(self.K, 
                             self.nrow, 
                             cluster_assignment, 
                             self.theta_dict,
                             self.S_dict)
            return cluster_assignment, self.theta_dict, bic
        else:
            return cluster_assignment, self.theta_dict


    def write_plot(self, cluster_assignment, str_NULL, training_indices, iters):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(cluster_assignment, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.K + 0.5))
        plt.xlabel('time index')
        plt.ylabel('cluster label')
        fname_fig = (str_NULL + "TRAINING_EM_iter={:000}.jpg".format(iters))
        if self.verbose:
            print(fname_fig)
        if self.write_out_file: plt.savefig(fname_fig)
        plt.close()
        if self.verbose:
            print("Done writing the figure")


    def optimize_clusters(self, upper_theta_dict):
        for k in range(self.K):
            if upper_theta_dict[k] == None:
                continue
            # from multiprocessing.pool.ApplyResult to its value
            upper_theta = upper_theta_dict[k].get()
            if self.verbose:
                print("OPTIMIZATION for Cluster #", k, "DONE!!!")
            # THIS IS THE SOLUTION
            theta = upperToFull(upper_theta, 0) # full theta (inv covariance)
            log_det_theta = np.log(np.linalg.det(theta))  # log(det(sigma^-2))
            covariance = np.linalg.inv(theta)
            # Store the covariance, inverse-covariance, log_det_theta
            self.theta_dict[k] = theta
            self.S_est_dict[k] = covariance
            self.log_det_theta_dict[k] = log_det_theta


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
                # # from multiprocessing.pool.ApplyResult to its value
                # upper_theta_dict[k] = pool_result.get()
                upper_theta_dict[k] = pool_result
                # save empirical mean and covariance
                self.mu_dict[k] = np.mean(D_train_c, axis=0)
                self.S_dict[k] = S
        return upper_theta_dict


    def prepare_out_directory(self):
        str_NULL = self.prefix_string + '/'
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise
        return str_NULL


    def log_parameters(self):
        if self.verbose:
            print("lam_sparse", self.ld)
            print("switch_penalty", self.bt)
            print("num_cluster", self.K)
            print("num stacked", self.w)


    def predict_clusters(self, test_data):
        '''
        Given the current trained model, predict clusters.  
        If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  
            Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if not isinstance(test_data, np.ndarray):
            raise TypeError("input must be a numpy array!")

        # SMOOTHENING
        LLE_given_theta = \
            self.smoothen_clusters(test_data)

        # Update cluster points - using NEW smoothening
        cluster_assignment = updateClusters(LLE_given_theta, beta=self.bt)
        return cluster_assignment


    def smoothen_clusters(self, D_train):
        # For each point compute the LLE
        if self.verbose:
            print("beginning the smoothening ALGORITHM")
        
        T = self.nrow - self.w + 1
        assert len(D_train) == T

        LLE_given_theta = np.zeros([T, self.K])
        for t in range(T):
            if t + self.w - 1 < D_train.shape[0]:
                for k in range(self.K):
                    mu_k = self.mu_dict[k]
                    theta_k = self.theta_dict[k]
                    log_det_theta_k = self.log_det_theta_dict[k]
                    
                    X2 = (D_train[t, :] - mu_k).reshape([self.probSize, 1])
                    lle = np.dot(X2.T, np.dot(theta_k, X2)) - log_det_theta_k
                    LLE_given_theta[t, k] = lle
        return LLE_given_theta