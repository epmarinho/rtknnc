import numpy as np
import pytest
from cannon_knn import LocalMahalanobis
from cannon_knn.metric import _local_covariance, _mahalanobis_knn

@pytest.fixture
def data():
    return np.random.default_rng(7).standard_normal((60, 3))

def test_local_covariance_pd(data):
    assert np.all(np.linalg.eigvalsh(_local_covariance(data[:15])) > 0)

def test_mahalanobis_knn_count(data):
    idx, dists = _mahalanobis_knn(data, data[0], np.eye(3), k=5)
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