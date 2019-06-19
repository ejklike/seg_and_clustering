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


def updateClusters(NLL, beta=1):
    """
    Takes in log_likelihood matrix and computes the path that minimizes
    the total cost over the path

    Note: switch penalty (=beta) > 0
    """
    T, K = NLL.shape

    # compute future costs
    FutureCost = np.zeros(NLL.shape)
    for t in range(T-2, -1, -1): # for t in [T-2, 0]
        for k in range(K):
            total_cost = FutureCost[t+1, :] + NLL[t+1, :] + beta
            total_cost[k] -= beta # no switching, no penalty.
            FutureCost[t, k] = np.min(total_cost)

    # compute the best path
    path = np.zeros(T, dtype=np.int)
    # the first location (no switching)
    path[0] = np.argmin(FutureCost[0, :] + NLL[0, :])
    # compute the path
    for t in range(T-1):
        prev_location = int(path[t])
        total_cost = FutureCost[t+1, :] + NLL[t+1, :] + beta
        total_cost[prev_location] -= beta
        path[t+1] = np.argmin(total_cost)

    # return the computed path
    return path


# def computeBIC(K, T, cluster_assignment, theta_dict, S_dict, threshold=2e-5):
#     '''
#     G. Schwarz (1978) proposed the “Bayes information criterion (BIC)”
#     ---
#     BIC = -2 * MLL + d * log(n)

#     - MLL: maximum log-likelihood
#     - d  : the number of free parameters to be estimated
#     - n  : the sample size

#     ---    
#     When fitting models, it is possible to increase the likelihood 
#     by adding parameters, but doing so may result in overfitting. 
#     The BIC resolves this problem by introducing a penalty term for 
#     the number of parameters in the model. 
#     The penalty term is larger in BIC than in AIC.

#     ---
#     empirical covariance and inverse_covariance should be dicts
#     K is num clusters
#     T is num samples
#     '''
#     mod_lle = 0
#     degree_dict = {}
#     for k in range(K):
#         log_det_theta_k = np.log(np.linalg.det(theta_dict[k]))
#         tr_S_theta_k = np.trace(np.dot(S_dict[k], theta_dict[k]))
#         mod_lle += log_det_theta_k - tr_S_theta_k
#         degree_dict[k] = np.sum(np.abs(theta_dict[k]) > threshold)

#     curr_val = -1
#     non_zero_params = 0
#     for c in cluster_assignment:
#         if c != curr_val:
#             non_zero_params += degree_dict[c]
#             curr_val = c
#     return 2*mod_lle - non_zero_params * np.log(T)


# def compute_BIC_score(T, K, theta_dict, S_dict, threshold=2e-5):
#     """
#     compute BIC score for the clusters 
#     """

#     mod_lle = 0
#     nonzero_params = 0
#     for k in range(K):
#         log_det_theta_k = np.log(np.linalg.det(theta_dict[k]))
#         tr_S_theta_k = np.trace(np.dot(S_dict[k], theta_dict[k]))
#         mod_lle += log_det_theta_k - tr_S_theta_k
#         nonzero_params += np.sum(np.abs(theta_dict[k]) > threshold)
    
#     tot_lle = mod_lle / K
#     BIC = nonzero_params * np.log(T) - 2 * tot_lle

#     return BIC

def compute_BIC_pointwise(T, K, theta_dict, S_dict, criterion='B',
                          threshold=2e-5, lle_list=None, P_dict=None, verbose=False):
    """
    compute BIC score for the clusters 
    ---

    G. Schwarz (1978) proposed the “Bayes information criterion (BIC)”
    
    BIC = -2 * MLL + d * log(n)

    - MLL: maximum log-likelihood
    - d  : the number of free parameters to be estimated
    - n  : the sample size
        
    In our case, the MLL can be calculated as:
    
    MLL_k = log det (theta_k) - tr(S_k theta_k)

    """

    assert criterion in [None, 'A', 'B']

    mod_lle = 0
    nonzero_params = 0
    for k in range(K):
        log_det_theta_k = np.log(np.linalg.det(theta_dict[k]))
        tr_S_theta_k = np.trace(np.dot(S_dict[k], theta_dict[k]))
        mod_lle += log_det_theta_k - tr_S_theta_k
        nonzero_params += np.sum(np.abs(theta_dict[k]) > threshold)
    
    # if lle_list is not None:
    #     point_lle = 0
    #     lle_list = np.array(lle_list)
    #     for k, P_k in P_dict.items():
    #         point_lle += np.sum(lle_list[P_k]) / len(P_k)
    #     # point_lle = np.sum(lle_list)
    #     print(mod_lle, point_lle, nonzero_params)
    #     # tot_lle = mod_lle / K + point_lle
    # # else:


    tot_lle = mod_lle
    AIC = 2 * nonzero_params - 2 * tot_lle
    BIC = nonzero_params * np.log(T) - 2 * tot_lle

    if verbose:
        print('---AIC={:.1f}, BIC={:.1f} | log(T)={:.1f}, K={}, MLL={:.1f}'
            .format(AIC, BIC, np.log(T), nonzero_params, mod_lle))

    if criterion is None:
        return tot_lle#, nonzero_params
    if criterion == 'A':
        return AIC
    if criterion == 'B':
        return BIC