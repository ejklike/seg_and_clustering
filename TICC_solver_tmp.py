import numpy as np
import math, time, collections, os, errno, sys, code, random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
import pandas as pd
from multiprocessing import Pool

from .src.TICC_helper import *
from .src.admm_solver import ADMMSolver


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5, lambda_parameter=11e-2,
                 beta=400, maxIters=1000, threshold=2e-5, write_out_file=False, verbose=True,
                 prefix_string="", num_proc=1, compute_BIC=False, cluster_reassignment=20):
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
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.verbose = verbose
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

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
        nrow, ncol = input_data.shape

        # Train test split
        training_indices = np.arange(nrow - self.window_size + 1)

        # Stack the training data
        if self.window_size == 1:
            D_train = input_data
        if self.window_size == 2:
            D_train = np.concatenate([input_data[:-1], input_data[1:]], 
                                     axis=1)

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, 
                                      covariance_type="full")
        gmm.fit(D_train)
        cluster_assignment = gmm.predict(D_train)

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_cluster_assignment = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        # print('start multi-threading')
        pool = Pool(processes=self.num_proc)  # multi-threading
        try:
            for iters in range(self.maxIters):
                if self.verbose:
                    print("\n\n\nITERATION ###", iters)
                # Get the train and test points
                train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
                for point, cluster_num in enumerate(cluster_assignment): # idx, gmm_labels
                    train_clusters_arr[cluster_num].append(point)

                len_cluster_dict = Counter(cluster_assignment)

                # train_clusters holds the indices in complete_D_train
                # for each of the clusters
                opt_res = self.train_clusters(cluster_mean_info, 
                                              cluster_mean_stacked_info, 
                                              D_train,
                                              empirical_covariances, 
                                              len_cluster_dict, 
                                              ncol, 
                                              pool,
                                              train_clusters_arr)

                self.optimize_clusters(computed_covariance, 
                                       len_cluster_dict, 
                                       log_det_values, 
                                       opt_res,
                                       train_cluster_inverse)

                # update old computed covariance
                old_computed_covariance = computed_covariance
                if self.verbose:
                    print("UPDATED THE OLD COVARIANCE")

                # based on self.trained model, predict cluster points
                self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                      'computed_covariance': computed_covariance,
                                      'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                      'D_train': D_train,
                                      'time_series_col_size': ncol}
                cluster_assignment = self.predict_clusters()

                # recalculate lengths
                new_train_clusters = collections.defaultdict(list) # {cluster: [point indices]}
                for point, cluster in enumerate(cluster_assignment):
                    new_train_clusters[cluster].append(point)

                len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

                before_empty_cluster_assign = cluster_assignment.copy()

                if iters != 0:
                    cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                    range(self.number_of_clusters)]
                    norms_sorted = sorted(cluster_norms, reverse=True)
                    # clusters that are not 0 as sorted by norm
                    valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                    # Add a point to the empty clusters
                    # assuming more non empty clusters than empty ones
                    counter = 0
                    for cluster_num in range(self.number_of_clusters):
                        if len_new_train_clusters[cluster_num] == 0:
                            cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                            counter = (counter + 1) % len(valid_clusters)
                            if self.verbose:
                                print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                            start_point = np.random.choice(
                                new_train_clusters[cluster_selected])  # random point number from that cluster
                            for i in range(0, self.cluster_reassignment):
                                # put cluster_reassignment points from point_num in this cluster
                                point_to_move = start_point + i
                                if point_to_move >= len(cluster_assignment):
                                    break
                                cluster_assignment[point_to_move] = cluster_num
                                computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                    self.number_of_clusters, cluster_selected]
                                cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = D_train[
                                                                                                point_to_move, :]
                                cluster_mean_info[self.number_of_clusters, cluster_num] \
                                    = D_train[point_to_move, :][
                                    (self.window_size - 1) * ncol:self.window_size * ncol]

                for cluster_num in range(self.number_of_clusters):
                    if self.verbose:
                        print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

                self.write_plot(cluster_assignment, str_NULL, training_indices, iters)

                if np.array_equal(old_cluster_assignment, cluster_assignment):
                    if self.verbose:
                        print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                    break
                old_cluster_assignment = before_empty_cluster_assign
                # end of training
        except np.linalg.LinAlgError as err:
            if pool is not None:
                pool.close()
                pool.join()
            cluster_assignment, train_cluster_inverse, bic = None, None, -1
            return cluster_assignment, train_cluster_inverse, bic
            
        if pool is not None:
            pool.close()
            pool.join()

        if self.compute_BIC:
            bic = computeBIC(self.number_of_clusters, 
                             nrow, 
                             cluster_assignment, 
                             train_cluster_inverse,
                             empirical_covariances)
            return cluster_assignment, train_cluster_inverse, bic

        return cluster_assignment, train_cluster_inverse


    def write_plot(self, cluster_assignment, str_NULL, training_indices, iters):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(cluster_assignment, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.number_of_clusters + 0.5))
        plt.xlabel('time index')
        plt.ylabel('cluster label')
        fname_fig = (str_NULL + "TRAINING_EM_" + 
            # "n_cluster=" + str(self.number_of_clusters) + 
            # "lam_sparse=" + str(self.lambda_parameter) + 
            # "switch_penalty = " + str(self.switch_penalty) + 
            "iter={:000}.jpg".format(iters))
        if self.verbose:
            print(fname_fig)
        if self.write_out_file: plt.savefig(fname_fig)
        plt.close("all")
        if self.verbose:
            print("Done writing the figure")


    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, D_train, n):
        cluster_assignment_len = len(D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(self.num_blocks - 1) * n,
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        if self.verbose:
            print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([cluster_assignment_len, self.number_of_clusters])
        for point in range(cluster_assignment_len):
            if point + self.window_size - 1 < D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean = cluster_mean_info[self.number_of_clusters, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters


    def optimize_clusters(self, computed_covariance, len_cluster_dict, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.number_of_clusters):
            if optRes[cluster] == None:
                continue
            val = optRes[cluster].get()
            if self.verbose:
                print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.number_of_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.number_of_clusters):
            if self.verbose:
                print("length of the cluster ", cluster, "------>", len_cluster_dict[cluster])


    def train_clusters(self, 
                       cluster_mean_info,         # {}
                       cluster_mean_stacked_info, # {}
                       D_train,                   # input_data
                       empirical_covariances,     # {}
                       len_cluster_dict,          # {cluster: len(point indices)]}
                       ncol,                         # col_size
                       pool,                      # multi-threading
                       train_clusters_arr):       # {cluster: [point indices]}
        optRes = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len_cluster_dict[cluster]
            if cluster_length != 0:
                indices = train_clusters_arr[cluster]
                D_train_c = np.zeros([cluster_length, self.window_size * ncol])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train_c[i, :] = D_train[point, :]

                cluster_mean_info[self.number_of_clusters, cluster] = \
                    np.mean(D_train_c, axis=0)[(self.window_size-1)*ncol:self.window_size*ncol].reshape([1, ncol])
                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)
                ##Fit a model - OPTIMIZATION
                probSize = self.window_size * ncol
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train_c))
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, self.window_size, ncol, rho, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        return optRes


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
            print("lam_sparse", self.lambda_parameter)
            print("switch_penalty", self.switch_penalty)
            print("num_cluster", self.number_of_clusters)
            print("num stacked", self.window_size)


    def predict_clusters(self, test_data = None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['ncol'])

        # Update cluster points - using NEW smoothening
        cluster_assignment = updateClusters(lle_all_points_clusters, 
                                            switch_penalty=self.switch_penalty)

        return cluster_assignment