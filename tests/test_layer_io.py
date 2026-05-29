import pytest

from rtknnc.layer_io import (
    LAYER_FORMAT_COMMENT,
    LAYER_HEADER_COMMENT,
    LayerRow,
    parse_cluster_token,
    parse_layer_line,
    read_layer_file,
)


def test_parse_cluster_token_accepts_historical_colon():
    assert parse_cluster_token("12:") == 12


def test_parse_layer_line_reads_labeled_rows():
    assert parse_layer_line("2: 7 1.5 -2.0 4") == LayerRow(
        cluster_id=2,
        point_id=7,
        coords=[1.5, -2.0],
        label=4.0,
    )


def test_parse_layer_line_reads_unlabeled_rows_when_requested():
    assert parse_layer_line("2: 7 1.5 -2.0", has_label=False) == LayerRow(
        cluster_id=2,
        point_id=7,
        coords=[1.5, -2.0],
        label=None,
    )


def test_read_layer_file_skips_comments_and_wraps_rows(tmp_path):
    layer_path = tmp_path / "sample.layer"
    layer_path.write_text(
        f"{LAYER_FORMAT_COMMENT}\n"
        f"{LAYER_HEADER_COMMENT}\n"
        "1: 1 0.0 1.0 3\n"
        "\n"
        "2: 2 2.0 3.0 4\n",
        encoding="utf-8",
    )

    rows = read_layer_file(layer_path)

    assert rows == [
        LayerRow(cluster_id=1, point_id=1, coords=[0.0, 1.0], label=3.0),
        LayerRow(cluster_id=2, point_id=2, coords=[2.0, 3.0], label=4.0),
    ]


def test_read_layer_file_keeps_legacy_unlabeled_2d_layers(tmp_path):
    layer_path = tmp_path / "legacy.layer"
    layer_path.write_text("1: 1 0.0 1.0\n", encoding="utf-8")

    assert read_layer_file(layer_path) == [
        LayerRow(cluster_id=1, point_id=1, coords=[0.0, 1.0], label=None)
    ]


def test_read_layer_file_reports_path_and_line(tmp_path):
    layer_path = tmp_path / "broken.layer"
    layer_path.write_text("1:\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.layer:1: expected"):
        read_layer_file(layer_path)
