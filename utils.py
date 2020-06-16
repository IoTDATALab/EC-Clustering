import numpy as np
import scipy.sparse as sp
import copy
import warnings
import pandas as pd
import sys
import math

from sklearn.metrics.pairwise import euclidean_distances,pairwise_distances_argmin_min
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state, check_array,gen_batches
from sklearn.utils.extmath import stable_cumsum, row_norms,squared_norm
from sklearn.utils.validation import FLOAT_DTYPES, _num_samples
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
from collections import Counter
from joblib import Parallel, delayed, effective_n_jobs







def _k_estimation(X, n_clusters, x_squared_norms, random_state, n_local_trials=None, delta = 0.0001):
    ''' Optimize the cluster number in dataset X by containing (1-delta) information (modified from sklearn.cluster._kmeans._k_init)
    Parameters
    -----------------------
        X: dataset for cluster number estimation
        n_clusters: maximum cluster number 
        x_squared_norms
        random_state
        n_local_trials
        delta: threshold for D(x) cut down
    Retrun
    ------------------------
        estimate_cluster_num: estimate cluster number k
        centers[0:estimate_cluster_num]: seeds from original dataset X
    '''
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_estimate'
    random_state = check_random_state(random_state)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()
    max_diff_pot = 0
    form_pot = current_pot
    estimate_cluster_num = n_clusters
    
    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),rand_vals)

        distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        form_pot = current_pot
        current_pot = best_pot
        closest_dist_sq = best_dist_sq
        if c >= 2:
            if c == 2:
                max_diff_pot = form_pot - current_pot
            elif (form_pot - current_pot)/(max_diff_pot+0.0)<delta:
                estimate_cluster_num = c
                break
    return centers[0:estimate_cluster_num], estimate_cluster_num 

class k_means_center:
    ''' Structure for centers in k-means
        Parameter
        -----------------------
        X: points in cluster
        
        Attributions
        -----------------------
        center_ :mean of  X 
        size_:size of X
        pfunv_:sum of square distance of point in X and center_
    '''
    def __init__(self, **kargs):
        if 'X' in kargs:
            X = kargs.get('X')
            self.center_ = np.mean(X, axis = 0)
            self.size_ = np.size(X, axis = 0)
            self.pfunv_ = np.sum(np.linalg.norm(X - self.center_))
        if 'center' in kargs:
            self.center_=kargs.get('center')
        if 'size' in kargs:
            self.size_=kargs.get('size')
        if 'pfunv' in kargs:
            self.pfunv_=kargs.get('pfunv')

class edge_center(k_means_center):
    ''' Structure for center and its summary from edge
        Function
            add_centers: combine centers 
    '''
    def __init__(self, **kargs):
        if 'kcenter' in kargs:
            kcenter = kargs.get('kcenter')
            self.center_ = kcenter.center_
            self.size_ = kcenter.size_
            self.pfunv_ = kcenter.pfunv_
        if ('center' in kargs) and ('pfunv' in kargs):
            self.center_ = kargs.get('center')
            self.size_ = kargs.get('size')
            self.pfunv_ = kargs.get('pfunv')
    def copy(self):
        return edge_center(center = self.center_, size = self.size_,
            pfunv = self.pfunv_)
    def add_centers(self, centers):
        for c in centers:
            center = (c.center_*c.size_ + self.center_*self.size_)\
                /(self.size_+ c.size_+0.0)
            size = self.size_+ c.size_
            pfunv = c.pfunv_ + self.pfunv_\
                + self.size_*np.linalg.norm(center - self.center_)**2\
                + c.size_*np.linalg.norm(center - c.center_)**2
            self.center_ = center
            self.size_ = size
            self.pfunv_ = pfunv

def edge_summarize(X, n_clusters, delta, copy_data=False):
    ''' Run k-means at the edge server, including cluster number estimation, \
        k-means clustering and center calculation
    Parameters
    --------------------------
        X: dataset for clustering
        n_clusters: maximum cluster number for k-means
        delta: parameter for cluster number k estimation
        copy_data: Return original data(True) or labels(False) of cluster

    Return
    --------------------------
        centers: array of k_means_center
        datasets: depends on copy_data, original data(True) or labels(False) of clusters
    '''
    # cluster number estimation by _k_init with 1-delta information remaining
    Xnorm = row_norms(X, squared=True)
    centers,n_clusters = _k_estimation(X, n_clusters, x_squared_norms=Xnorm, random_state=True, delta=delta)

    # Note: using the comment if only recording communication size
    # k_means = KMeans(init=centers, n_clusters=n_clusters)
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    
    centers = []
    X_df = pd.DataFrame(X)
    X_df['label'] = k_means.labels_
    for c_i in range(0, n_clusters):
        items = X_df[X_df['label']==c_i]
        items = items.drop(columns = 'label')
        centers.append(k_means_center(X = np.array(items)))
    return np.array(centers), k_means.labels_


def _k_cluster_init(X, x_weights, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init weighted n_clusters seeds according to k-means++(modified from sklearn.cluster._kmeans._k_init)
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/_kmeans.py
    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    x_weights : size of each cluster
    n_clusters : integer
        The number of seeds to choose
    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    random_state = check_random_state(random_state)
    assert x_squared_norms is not None, 'x_squared_norms None in _k_cluster_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(x_weights.sum())
    cumsum = 0
    for cluster_i, size in enumerate(x_weights):
        if cumsum>center_id:
            center_id = cluster_i
            break
        else:
            cumsum+=size

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = (closest_dist_sq*x_weights).sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq*x_weights),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        #**# update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                out=distance_to_candidates)
        candidates_pot = (distance_to_candidates*x_weights).sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers

def cloud_clustering(centers, k):
    ''' Combine clusters from edges with k-means method
        Parameters
        -------------------------
            centers: array of edge_center
            k: number of cluster
        Return
        -------------------------
            new_centers: k centers after combination
    '''
    center_vs =np.array([c.center_ for c in centers])
    center_we = np.array([c.size_ for c in centers])
    Xnorm = row_norms(center_vs, squared=True)
    
    cluster_center = _k_cluster_init(X = center_vs, x_weights=center_we , n_clusters=k, 
        x_squared_norms=Xnorm, random_state=True, n_local_trials=None)

    kmeans = KMeans(n_clusters = k)
    kmeans.fit(center_vs, sample_weight=center_we)
    new_centers = []
    for l in set(kmeans.labels_):
        indexs = np.where(kmeans.labels_ == l)
        centerlist = indexs[0]
        
        # centerlist = list(set(labels[indexs[0]]))
        if len(centerlist) > 1:
            new_center = centers[centerlist[0]].copy()
            new_center.add_centers([centers[i] for i in centerlist[1:len(centerlist)]])
            new_centers.append(new_center)
        else:
            new_centers.append(centers[centerlist[0]].copy())

    return new_centers

def f1(results, labels):
    ''' Function to measure f1-sore of clustering results
    Parameter:
    -----------------------------------------
        results- (Nx1 array)clustering results
        labels - (Nx1 array)labels
    Return:
    ------------------------------------------
        f1-score 
        (print f1-score of each cluster)

    '''
    results = np.array(results)
    labels = np.array(labels)
    f1s =0
    for k_i in set(results):
        indexs = np.where(results == k_i)
        counter = Counter(labels[indexs[0]]).most_common(1)
        c_len = len(np.where(labels == counter[0][0])[0])
        kc_len = counter[0][1]
        k_len = len(indexs[0])
        p = kc_len/(k_len+0.0)
        r = kc_len/(c_len+0.0)
        f1 = 2*p*r/(p+r)
        f1s+= f1*k_len
        # print(f1, p,r)
    f1s = f1s/(len(results)+0.0)
    return f1s