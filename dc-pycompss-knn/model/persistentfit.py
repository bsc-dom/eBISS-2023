from copy import copy

import numpy as np

from sklearn.neighbors import NearestNeighbors

from dataclay import DataClayObject, activemethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN
    Depth = None
    Type = None
    COLLECTION_IN = None


class PersistentFitStructure(DataClayObject):
    """Split that tracks the internal item index for each chunk."""
    _nn: NearestNeighbors
    _itemindexes: np.ndarray

    def __init__(self):
        self._nn = None

    @task(target_direction=IN)
    @activemethod
    def fit(self, blocks: list, offset: int):
        subdataset = np.block(blocks)

        self._nn = NearestNeighbors()
        self._nn.fit(subdataset)

        # Track properly the item indexes
        self._itemindexes = np.arange(offset, offset + len(subdataset))

    @task(target_direction=IN, returns=2)
    @activemethod
    def get_kneighbors(self, q: np.ndarray, n_neighbors: int) -> tuple:
        # Prepare a new structure for the tree walk
        # (due to the lack of readonly/concurrent implementation in the KDTree sklearn implementation)
        nn = copy(self._nn)
        nn._tree = copy(self._nn._tree)

        # Note that the merge requires distances, so we ask for them
        dist, ind = nn.kneighbors(X=q, n_neighbors=n_neighbors)

        return dist, self._itemindexes[ind]
        #            ^****** This converts the local indexes to global ones
