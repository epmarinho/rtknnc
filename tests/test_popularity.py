import numpy as np
import pytest
from rtknnc import (KNNGraph, node_popularity, vote_propagation,
                        leadership_graph, cluster_seeds, popularity_pipeline)

@pytest.fixture
def graph():
    return KNNGraph(k=6).fit(np.random.default_rng(0).standard_normal((80, 3)))

def test_popularity_range(graph):
    pop = node_popularity(graph.adjacency_)
    assert pop.min() >= 0.0 and pop.max() <= 1.0

def test_vote_propagation_range(graph):
    s = vote_propagation(graph.adjacency_)
    assert s.min() >= 0.0 and s.max() <= 1.0

def test_leadership_direction(graph):
    scores = vote_propagation(graph.adjacency_)
    L = leadership_graph(graph.adjacency_, scores).tocoo()
    assert np.all(scores[L.col] > scores[L.row])

def test_cluster_seeds_min_size(graph):
    scores = vote_propagation(graph.adjacency_)
    for c in cluster_seeds(scores, graph.adjacency_, threshold=0.3):
        assert len(c) >= 3

def test_pipeline_keys(graph):
    result = popularity_pipeline(graph.adjacency_)
    for key in ("raw_popularity", "scores", "leadership", "clusters"):
        assert key in result

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
