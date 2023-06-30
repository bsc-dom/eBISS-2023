import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import validation

from dislib.data.array import Array

from pycompss.api.task import task
from pycompss.api.parameter import COLLECTION_IN

from .persistentfit import PersistentFitStructure


@task(returns=2, queries=COLLECTION_IN)
def _merge_kqueries(k, queries):
    # Reorganize and flatten
    dist, ind = zip(*queries)
    aggr_dist = np.hstack(dist)
    aggr_ind = np.hstack(ind)

    # Final indexes of the indexes (sic)
    final_ii = np.argsort(aggr_dist)[:,:k]

    # Final results
    final_dist = np.take_along_axis(aggr_dist, final_ii, 1)
    final_ind = np.take_along_axis(aggr_ind, final_ii, 1)

    return final_dist, final_ind


class DCKNNClassifier(BaseEstimator):
    """k-Nearest Neighbors classifier with dataClay."""
    _fit_data: list[PersistentFitStructure]
    n_neighbors: int

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, x, y):
        self.labels = y
        self._fit_data = list()

        if len(x._blocks[0]) != 1:
            raise ValueError("I only know how to work with dsarray of one column")

        offset = 0

        for row in x._iterator(axis=0):
            nn = PersistentFitStructure()
            nn.make_persistent()
            nn.fit(row._blocks, offset)
            self._fit_data.append(nn)
            # Carry the offset by counting samples
            offset += row.shape[0]
    
    def kneighbors(self, x, n_neighbors=None, return_distance=True):
        validation.check_is_fitted(self, '_fit_data')
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        queries = []

        for nn_fit_struct in self._fit_data:
            queries.append(nn_fit_struct.get_kneighbors(x, n_neighbors))

        distances, indices = _merge_kqueries(n_neighbors, queries)

        if return_distance:
            return distances, indices

        return indices

    def _choose_label_kernel(self, vector):
        labels = self.labels[vector]
        counts = np.bincount(labels)
        return np.argmax(counts)
    
    @task()
    def choose_label(self, indices):
        return np.apply_along_axis(self._choose_label_kernel, axis=1, arr=indices)

    def predict(self, x):
        neigh_ind = self.kneighbors(x, return_distance=False)
        return self.choose_label(neigh_ind)
