import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import validation

from dislib.data.array import Array

from pycompss.api.task import task
from pycompss.api.parameter import IN, Depth, Type, COLLECTION_IN, INOUT

from .persistentfit import PersistentFitStructure, AllFitStructures


@task(returns=2)
def _merge_kqueries(k, *queries):
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
        
        distances = []
        indices = []

        for q_row in x._iterator(axis=0):
            queries = []

            for nn_fit_struct in self._fit_data:
                queries.append(nn_fit_struct.get_kneighbors(q_row._blocks, n_neighbors))

            dist, ind = _merge_kqueries(n_neighbors, *queries)
            distances.append([dist])
            indices.append([ind])

        ind_arr = Array(blocks=indices,
                        top_left_shape=(x._top_left_shape[0], n_neighbors),
                        reg_shape=(x._reg_shape[0], n_neighbors),
                        shape=(x.shape[0], n_neighbors), sparse=False)

        if return_distance:
            dst_arr = Array(blocks=distances,
                            top_left_shape=(x._top_left_shape[0], n_neighbors),
                            reg_shape=(x._reg_shape[0], n_neighbors),
                            shape=(x.shape[0], n_neighbors), sparse=False)
            return dst_arr, ind_arr

        return ind_arr

    def _choose_label(self, vector):
        labels = self.labels[vector]
        counts = np.bincount(labels)
        return np.argmax(counts)

    def predict(self, x):
        neigh_ind = self.kneighbors(x, return_distance=False)
        pred = np.apply_along_axis(self._choose_label, axis=0, arr=neigh_ind)
        return pred

    def store_fit_data(self, alias: str):
        afs = AllFitStructures()
        afs.make_persistent(alias=alias)
        afs.nn = self._fit_data
    
    def load_fit_data(self, alias: str):
        afs = AllFitStructures.get_by_alias(alias)
        self._fit_data = afs.nn
