import numpy as np


def upperToFull(a, eps=0):
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1 + np.sqrt(1 + 8*a.shape[0]))/2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def updateClusters(LLE, beta=1):
    """
    Takes in LLE_node_vals matrix and computes the path that minimizes
    the total cost over the path
    Note the LLE's are negative of the true LLE's actually!!!!!

    Note: switch penalty (=beta) > 0
    """
    T, K = LLE.shape

    # compute future costs
    FutureCost = np.zeros(LLE.shape)
    for t in range(T-2, -1, -1): # for t in [T-2, 0]
        for k in range(K):
            total_cost = FutureCost[t+1, :] + LLE[t+1, :] + beta
            total_cost[k] -= beta # no switching, no penalty.
            FutureCost[t, k] = np.min(total_cost)

    # compute the best path
    path = np.zeros(T)
    # the first location (no switching)
    path[0] = np.argmin(FutureCost[0, :] + LLE[0, :])
    # compute the path
    for t in range(T-1):
        prev_location = int(path[t])
        total_cost = FutureCost[t+1, :] + LLE[t+1, :] + beta
        total_cost[prev_location] -= beta
        path[t+1] = np.argmin(total_cost)

    # return the computed path
    return path


def computeBIC(K, T, cluster_assignment, theta_dict, S_dict, threshold=2e-5):
    '''
    G. Schwarz (1978) proposed the “Bayes information criterion (BIC)”
    ---
    BIC = 2 * MLL - d * log(n)

    - MLL: maximum log-likelihood
    - d  : the number of free parameters to be estimated
    - n  : the sample size

    ---    
    When fitting models, it is possible to increase the likelihood 
    by adding parameters, but doing so may result in overfitting. 
    The BIC resolves this problem by introducing a penalty term for 
    the number of parameters in the model. 
    The penalty term is larger in BIC than in AIC.

    ---
    empirical covariance and inverse_covariance should be dicts
    K is num clusters
    T is num samples
    '''
    mod_lle = 0
    degree_dict = {}
    for k in range(K):
        log_det_theta_k = np.log(np.linalg.det(theta_dict[k]))
        tr_S_theta_k = np.trace(np.dot(S_dict[k], theta_dict[k]))
        mod_lle += log_det_theta_k - tr_S_theta_k
        degree_dict[k] = np.sum(np.abs(theta_dict[k]) > threshold)
    
    curr_val = -1
    non_zero_params = 0
    for c in cluster_assignment:
        if c != curr_val:
            non_zero_params += degree_dict[c]
            curr_val = c
    return 2*mod_lle - non_zero_params * np.log(T)
