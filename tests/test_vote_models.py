import numpy as np
import pytest

from rtknnc.vote_models import vote_weights, weighted_popularity


DISTANCES = np.array(
    [
        [1.0, 2.0, 4.0],
        [0.5, 1.5, 3.0],
    ]
)


@pytest.mark.parametrize("model", ["uniform", "inv_r", "inv_r2", "sigmoid"])
def test_every_kernel_conserves_the_historical_source_budget(model):
    weights = vote_weights(DISTANCES, model=model, sigmoid_alpha=8.0)
    historical_budget = np.sum(1.0 / np.square(DISTANCES), axis=1)

    np.testing.assert_allclose(weights.sum(axis=1), historical_budget)


def test_inverse_square_remains_exactly_historical():
    expected = 1.0 / np.square(DISTANCES)

    np.testing.assert_array_equal(vote_weights(DISTANCES, "inv_r2"), expected)


def test_uniform_redistributes_each_source_budget_equally():
    budget = np.sum(1.0 / np.square(DISTANCES), axis=1, keepdims=True)
    expected = np.broadcast_to(budget / DISTANCES.shape[1], DISTANCES.shape)

    np.testing.assert_allclose(vote_weights(DISTANCES, "uniform"), expected)


def test_inverse_linear_changes_only_the_within_source_allocation():
    shape = 1.0 / DISTANCES
    budget = np.sum(1.0 / np.square(DISTANCES), axis=1, keepdims=True)
    expected = budget * shape / shape.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(vote_weights(DISTANCES, "inv_r"), expected)


def test_sigmoid_is_budget_matched_but_not_uniform():
    radius = DISTANCES[:, -1][:, None]
    shape = 1.0 / (1.0 + np.exp(8.0 * (DISTANCES / radius - 1.0)))
    budget = np.sum(1.0 / np.square(DISTANCES), axis=1, keepdims=True)
    expected = budget * shape / shape.sum(axis=1, keepdims=True)
    actual = vote_weights(DISTANCES, "sigmoid", sigmoid_alpha=8.0)

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, vote_weights(DISTANCES, "uniform"))
    assert not np.allclose(actual, vote_weights(DISTANCES, "inv_r"))


@pytest.mark.parametrize("model", ["uniform", "inv_r", "inv_r2", "sigmoid"])
def test_global_coordinate_scaling_preserves_shape_and_inverse_square_units(model):
    scale = 7.5
    original = vote_weights(DISTANCES, model=model)
    scaled = vote_weights(scale * DISTANCES, model=model)

    np.testing.assert_allclose(scaled, original / scale**2)


@pytest.mark.parametrize("model", ["uniform", "inv_r", "inv_r2", "sigmoid"])
def test_all_budget_matched_kernels_reject_zero_distance(model):
    distances = DISTANCES.copy()
    distances[0, 0] = 0.0

    with pytest.raises(ValueError, match="require r > 0"):
        vote_weights(distances, model=model)


def test_weighted_popularity_uses_budget_matched_edge_votes():
    indices = np.array([[1, 2, 3], [0, 2, 3]])
    weights = vote_weights(DISTANCES, model="sigmoid")
    expected = np.zeros(4)
    np.add.at(expected, indices.ravel(), weights.ravel())

    actual = weighted_popularity(
        indices,
        DISTANCES,
        n_nodes=4,
        model="sigmoid",
    )
    np.testing.assert_allclose(actual, expected)
