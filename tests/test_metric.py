import numpy as np
import pytest
from rtknnc import LocalMahalanobis
from rtknnc.metric import _local_covariance, _mahalanobis_knn

@pytest.fixture
def data():
    return np.random.default_rng(7).standard_normal((60, 3))

def test_local_covariance_pd(data):
    assert np.all(np.linalg.eigvalsh(_local_covariance(data[:15])) > 0)

def test_mahalanobis_knn_count(data):
    idx, dists = _mahalanobis_knn(data, data[0], np.eye(3), k=5)
    assert len(idx) == 5 and len(dists) == 5

def test_mahalanobis_knn_can_exclude_self(data):
    idx, dists = _mahalanobis_knn(data, data[0], np.eye(3), k=5, exclude_index=0)
    assert 0 not in idx
    assert len(idx) == 5 and len(dists) == 5

def test_fit_query_keys(data):
    result = LocalMahalanobis(k=5, max_iter=10).fit(data).fit_query(data[0])
    for key in ("indices", "distances", "cov", "metric", "n_iter"):
        assert key in result

def test_fit_query_index_count(data):
    assert len(LocalMahalanobis(k=5).fit(data).fit_query(data[10])["indices"]) == 5

def test_metric_positive_definite(data):
    M = LocalMahalanobis(k=8).fit(data).fit_query(data[5])["metric"]
    assert np.all(np.linalg.eigvalsh(M) > 0)

def test_ellipsoid_axes(data):
    axes = LocalMahalanobis(k=8).fit(data).ellipsoid_axes(data[0])
    assert axes["ratio"] >= 1.0 and len(axes["radii"]) == 3

def test_anisotropic_graph_excludes_self(data):
    A = LocalMahalanobis(k=5, max_iter=5).fit(data).anisotropic_knn_graph()
    assert np.all(A.diagonal() == 0)
    assert np.all(np.asarray(A.sum(axis=1)).ravel() == 5)

def test_anisotropic_graph_external_query_shape(data):
    queries = data[:7] + 0.01
    A = LocalMahalanobis(k=5, max_iter=5).fit(data).anisotropic_knn_graph(queries)
    assert A.shape == (len(queries), len(data))
    assert np.all(np.asarray(A.sum(axis=1)).ravel() == 5)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
