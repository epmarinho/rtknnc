import numpy as np
import pytest
from cannon_knn import KNNGraph

@pytest.fixture
def blob():
    return np.random.default_rng(42).standard_normal((100, 4))

def test_fit_shape(blob):
    assert KNNGraph(k=5).fit(blob).adjacency_.shape == (100, 100)

def test_outdegree_is_k(blob):
    assert np.all(KNNGraph(k=5).fit(blob).outdegree == 5)

def test_indegree_sum(blob):
    assert KNNGraph(k=5).fit(blob).indegree.sum() == 100 * 5

def test_roundtrip_edges_symmetric(blob):
    edges = KNNGraph(k=5).fit(blob).roundtrip_edges()
    assert edges.shape[1] == 2 and np.all(edges[:, 0] < edges[:, 1])

def test_roundtrip_adjacency_symmetric(blob):
    A = KNNGraph(k=5).fit(blob).roundtrip_adjacency()
    assert np.allclose((A - A.T).data, 0) if len((A - A.T).data) else True

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])