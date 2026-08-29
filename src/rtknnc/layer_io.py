from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

LAYER_FORMAT_COMMENT = "# format: rtknnc layer"
LAYER_HEADER_COMMENT = "# columns: cluster_id: point_id x0 x1 ... original_label"


@dataclass
class LayerRow:
    cluster_id: int
    point_id: int
    coords: list[float]
    label: Optional[float] = None


def parse_cluster_token(token: str) -> int:
    """
    Parse a cluster token from rtknnc.layer files.

    The expected format is either:
        "3"
    or:
        "3:"

    Returns:
        int: cluster id.
    """
    return int(token.strip().rstrip(":"))


def parse_layer_line(line: str, *, has_label: bool = True) -> LayerRow:
    """
    Parse one data row from a rtknnc layer file.

    The historical labeled format is:
        <cluster_id>: <point_id> <x0> <x1> ... <original_label>

    Comment lines that document the columns, for example
    "# cluster_id: point_id x0 x1 ... original_label", are handled by
    read_layer_file().

    Some older fixture files contain only coordinates after point_id; pass
    has_label=False for that unlabeled variant.
    """
    parts = line.split()
    min_columns = 4 if has_label else 3

    if len(parts) < min_columns:
        raise ValueError(
            f"expected at least {min_columns} columns, got {len(parts)}: {line}"
        )

    cluster_id = parse_cluster_token(parts[0])
    point_id = int(parts[1])

    if has_label:
        coords = [float(value) for value in parts[2:-1]]
        label = float(parts[-1])
    else:
        coords = [float(value) for value in parts[2:]]
        label = None

    if not coords:
        raise ValueError(f"expected at least one coordinate: {line}")

    return LayerRow(
        cluster_id=cluster_id,
        point_id=point_id,
        coords=coords,
        label=label,
    )


def infer_layer_has_label(
    lines: list[str],
    *,
    column_header: Optional[str] = None,
) -> bool:
    """
    Infer whether layer rows include a final label column.

    A columns header ending in ``original_label`` is authoritative. A header
    containing only coordinate names (x0, x1, x2, ...) is authoritative in
    the other direction, which disambiguates unlabeled 3D data from labeled
    2D data. Files without a usable header retain the historical inference:
    five or more fields mean that the last field is a label.
    """
    if column_header is not None:
        columns = column_header.removeprefix("# columns:").split()
        value_columns = columns[2:]
        if value_columns and value_columns[-1] == "original_label":
            return True
        if value_columns and all(
            column.startswith("x") and column[1:].isdigit()
            for column in value_columns
        ):
            return False

    return any(len(line.split()) > 4 for line in lines)


def read_layer_file(
    path: Union[str, Path],
    *,
    has_label: Optional[bool] = None,
) -> list[LayerRow]:
    path = Path(path)
    rows = []
    data_lines = []
    column_header = None

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if line.startswith("# columns:"):
                column_header = line

            if not line or line.startswith("#"):
                continue

            data_lines.append((lineno, line))

    if has_label is None:
        has_label = infer_layer_has_label(
            [line for _, line in data_lines],
            column_header=column_header,
        )

    for lineno, line in data_lines:
        try:
            rows.append(parse_layer_line(line, has_label=has_label))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc

    if not rows:
        raise ValueError(f"{path}: no data rows found")

    dimension = len(rows[0].coords)
    for lineno, row in zip((lineno for lineno, _ in data_lines), rows):
        if len(row.coords) != dimension:
            raise ValueError(
                f"{path}:{lineno}: expected {dimension} coordinates, "
                f"got {len(row.coords)}"
            )

    return rows
