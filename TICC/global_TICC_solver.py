import numpy as np
import math, time, collections, os, errno, sys, code, random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from multiprocessing import Pool

from .src.TICC_helper import upperToFull, compute_BIC_pointwise, updateClusters
from .src.admm_solver import ADMMSolver


class GlobalTICC:
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

        ############
        # The basic folder to be created
        self.str_NULL = self.prefix_string + '/'


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

    def get_S_dict(self, D, cluster_assignment):
        P_dict = self.get_P_dict(cluster_assignment)
        S_dict = {}
        for k in range(self.K):
            indices = P_dict[k]
            D_k = D[indices, :]
            S = np.cov(np.transpose(D_k))
            S_dict[k] = S
        return S_dict


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
        D_train_list = []
        for input_data in input_data_list:
            # Stack the training data
            if self.w == 1:
                D_train = input_data
            if self.w == 2:
                D_train = np.concatenate([input_data[:-1], input_data[1:]], axis=1)

            D_train_list.append(D_train)

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.K, 
                                      covariance_type="full", random_state=102)

        D_train_flat = np.concatenate(D_train_list, axis=0)
        cluster_assignment_flat = gmm.fit_predict(D_train_flat)
        old_cluster_assignment_flat = None  # points from last iteration

        # data statistics
        self.mu_dict = {}
        self.S_dict = {}
        # model parameters
        self.theta_dict = {}
        self.log_det_theta_dict = {}
        # self.S_est_dict = {}
        cluster_assignment_list = []

        # PERFORM TRAINING ITERATIONS
        # print('start multi-threading')
        pool = Pool(processes=self.num_proc)  # multi-threading
        try:
            for iters in range(self.maxIters):
                if self.verbose:
                    print("\n\n\nITERATION ###", iters)

                P_flat_dict = self.get_P_dict(cluster_assignment_flat)

                if self.verbose:
                    print('#############################################')
                    print('# M-step: Update cluster parameters --> theta')
                    print('#############################################')

                # calculate theta(inv_upper_cov), emp_cov, emp_mean
                upper_theta_dict = self.train_clusters(D_train_flat, P_flat_dict, pool)

                # calculate theta and its inverse
                self.optimize_clusters(upper_theta_dict)

                if self.verbose:
                    for k in range(self.K):
                        print("length of the cluster ", k, "--->", len(P_flat_dict[k]))

                if self.verbose:
                    print('##########################################')
                    print('# E-step: Assign points to clusters --> P ')
                    print('##########################################')

                cluster_assignment_list = []
                new_P_dict_list = []
                for D_train in D_train_list:
                    cluster_assignment, log_likelihood = self.predict_clusters(D_train)
                    new_P_dict = self.get_P_dict(cluster_assignment)
                    cluster_assignment_list.append(cluster_assignment)
                    new_P_dict_list.append(new_P_dict)

                # update S


                # ##########################################

                # # adjust empty clusters
                # if iters != 0:
                #     data_order = [[i]*len(cluster_assignment) 
                #         for i, cluster_assignment in enumerate(cluster_assignment_list)]
                #     data_order_flat = np.concatenate(data_order, axis=0)
                #     cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
                #     for i in range(len(cluster_assignment_list)):
                #         new_P_dict = new_P_dict_list[i] 
                #         cluster_assignment = cluster_assignment_list[i]

                #         cluster_norms = \
                #             [(np.linalg.norm(self.S_est_dict[k]), k) for k in range(self.K)]
                #         norms_sorted = sorted(cluster_norms, reverse=True)
                #         # clusters that are not 0 as sorted by norm
                #         valid_clusters = [k for n, k in norms_sorted if len(new_P_dict[k]) != 0]
                #         empty_clusters = [k for k in range(self.K) if len(new_P_dict[k]) == 0]

                #         # Add a point to the empty clusters
                #         # assuming more non empty clusters than empty ones
                #         counter = 0
                #         for k_empty in empty_clusters:
                #             k_selected = valid_clusters[counter]  # a cluster that is not len 0
                #             counter = (counter + 1) % len(valid_clusters)
                #             if self.verbose:
                #                 print("cluster that is zero is:", k_empty, 
                #                       "selected cluster instead is:", k_selected)
                            
                #             # random point number from that cluster
                #             t_start = np.random.choice(new_P_dict[k_selected])
                #             t_end = np.minimum(t_start + self.cluster_reassignment - 1, 
                #                                len(cluster_assignment) - 1)
                #             t_reassignment = np.arange(t_start, t_end + 1)
                #             cluster_assignment[t_reassignment] = k_empty
                #             cluster_assignment_list[i] = cluster_assignment
                            
                #             # self.S_est_dict[k_empty] = self.S_est_dict[k_selected]
                #             # self.mu_dict[k_empty] = np.mean(D_train[t_reassignment, :], axis=0)

                
                cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
                if self.verbose:
                    for k in range(self.K):
                        print("length of cluster #", k, "-------->", 
                              sum([x == k for x in cluster_assignment_flat]))

                self.write_plot(cluster_assignment_flat, iters)

                if np.array_equal(old_cluster_assignment_flat, cluster_assignment_flat):
                    if self.verbose:
                        print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                    break
                # old_cluster_assignment = before_empty_cluster_assign
                old_cluster_assignment_flat = cluster_assignment_flat.copy()
                # end of training
        except np.linalg.LinAlgError as err:
            if pool is not None:
                pool.close()
                pool.join()
            cluster_assignment_list, theta_dict, bic = None, None, 1e10
            return cluster_assignment_list, theta_dict, bic
            
        if pool is not None:
            pool.close()
            pool.join()

        T = np.sum([len(c) for c in cluster_assignment_list])
        bic = compute_BIC_pointwise(T, self.K, self.theta_dict, self.S_dict, 
            lle_list=log_likelihood, P_dict=new_P_dict)

        return cluster_assignment_list, self.theta_dict, bic, iters

    def test(self, input_data_list):

        assert self.ncol == input_data_list[0].shape[1]
        
        verbose_backup = self.verbose
        self.verbose = False

        # data preparation according to window_size
        cluster_assignment_list = []
        log_likelihood_list = []
        D_test_list = []
        for input_data in input_data_list:
            # Stack the test data
            if self.w == 1:
                D_test = input_data
            if self.w == 2:
                D_test = np.concatenate([input_data[:-1], input_data[1:]], axis=1)

            cluster_assignment, log_likelihood = self.predict_clusters(D_test)
            cluster_assignment_list.append(cluster_assignment)
            D_test_list.append(D_test)
            log_likelihood_list.append(log_likelihood)

        cluster_assignment_flat = np.concatenate(cluster_assignment_list, axis=0)
        log_likelihood_flat = np.concatenate(log_likelihood_list, axis=0)
        P_dict_flat = self.get_P_dict(cluster_assignment_flat)

        D_test_flat = np.concatenate(D_test_list, axis=0)
        test_S_dict = self.get_S_dict(D_test_flat, cluster_assignment_flat)
        T = len(cluster_assignment_flat)
        bic = compute_BIC_pointwise(
            T, self.K, self.theta_dict, test_S_dict, 
            lle_list=log_likelihood_flat, 
            P_dict=P_dict_flat)

        
        if self.verbose:
            for k in range(self.K):
                print("length of cluster #", k, "-------->", 
                        sum([x == k for x in cluster_assignment_flat]))

        self.write_plot(cluster_assignment_flat, 'test')
        self.verbose = verbose_backup
        return cluster_assignment_list, bic


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
            # covariance = np.linalg.inv(theta)
            # Store the covariance, inverse-covariance, log_det_theta
            self.theta_dict[k] = theta
            # self.S_est_dict[k] = covariance
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
        if not os.path.exists(os.path.dirname(self.str_NULL)):
            try:
                os.makedirs(os.path.dirname(self.str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

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
        log_likelihood_array = self.get_log_likelihood(test_data)

        # Update cluster points - using NEW smoothening
        cluster_assignment = updateClusters(log_likelihood_array, beta=self.bt)
        log_likelihood = \
            [log_likelihood_array[i, c] for i, c in enumerate(cluster_assignment)]
        return cluster_assignment, log_likelihood


    def get_log_likelihood(self, D_train):
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