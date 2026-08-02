#!/usr/bin/env python3
"""
latex_british_english_safe.py

Convert natural-language text in a LaTeX manuscript from US to British spelling,
while preserving LaTeX commands, macros, labels, citation keys, filenames, and
bibliography commands.

Usage:
  python latex_british_english_safe.py input.tex output.tex

Example:
  python latex_british_english_safe.py \
    doc/round_trip_knn_nca_sn_draft_final_discussed.tex \
    doc/round_trip_knn_nca_sn_draft_final_discussed_enGB.tex
"""

from pathlib import Path
import re
import sys


PROTECTED_COMMANDS = {
    "label",
    "ref",
    "eqref",
    "cite",
    "citep",
    "citet",
    "bibliography",
    "bibliographystyle",
    "includegraphics",
    "graphicspath",
    "usepackage",
    "documentclass",
    "newcommand",
    "newtheorem",
    "email",
}


REPLACEMENTS = [
    ("nearest-neighbors", "nearest-neighbours"),
    ("nearest-neighbor", "nearest-neighbour"),
    ("reverse-neighbors", "reverse-neighbours"),
    ("reverse-neighbor", "reverse-neighbour"),
    ("shared-neighbors", "shared-neighbours"),
    ("shared-neighbor", "shared-neighbour"),
    ("forward-neighbors", "forward-neighbours"),
    ("forward-neighbor", "forward-neighbour"),
    ("neighbors", "neighbours"),
    ("Neighbors", "Neighbours"),
    ("neighbor", "neighbour"),
    ("Neighbor", "Neighbour"),
    ("neighborhoods", "neighbourhoods"),
    ("Neighborhoods", "Neighbourhoods"),
    ("neighborhood", "neighbourhood"),
    ("Neighborhood", "Neighbourhood"),
    ("analyzed", "analysed"),
    ("Analyzed", "Analysed"),
    ("analyzing", "analysing"),
    ("normalizations", "normalisations"),
    ("normalization", "normalisation"),
    ("minimization", "minimisation"),
    ("organization", "organisation"),
    ("Organization", "Organisation"),
    ("organized", "organised"),
    ("emphasized", "emphasised"),
    ("behavior", "behaviour"),
    ("Behavior", "Behaviour"),
]


def _consume_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> int:
    """Return position after a balanced {...} or [...] group starting at pos."""
    assert text[pos] == open_ch
    depth = 1
    pos += 1

    while pos < len(text) and depth:
        ch = text[pos]

        if ch == "\\":
            # Skip escaped character/control sequence content safely.
            pos += 2
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1

        pos += 1

    return pos


def mask_protected_regions(text: str):
    """Mask command arguments that must remain byte-for-byte unchanged."""
    masks = []
    out = []
    last = 0
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "\\":
            i += 1
            continue

        j = i + 1
        while j < n and (text[j].isalpha() or text[j] == "@"):
            j += 1

        command = text[i + 1:j]

        if command not in PROTECTED_COMMANDS:
            i = j
            continue

        k = j

        while k < n and text[k].isspace():
            k += 1

        # Protect optional arguments, for example \includegraphics[width=...]{...}.
        while k < n and text[k] == "[":
            k = _consume_balanced(text, k, "[", "]")
            while k < n and text[k].isspace():
                k += 1

        # Protect required arguments. Most protected commands have one argument;
        # \newcommand and \newtheorem usually need at least two protected groups.
        groups_needed = 2 if command in {"newcommand", "newtheorem"} else 1
        groups = 0

        while k < n and text[k] == "{" and groups < groups_needed:
            k = _consume_balanced(text, k, "{", "}")
            groups += 1

            while k < n and text[k].isspace():
                k += 1

        out.append(text[last:i])
        token = f"@@LATEX_PROTECTED_{len(masks)}@@"
        masks.append(text[i:k])
        out.append(token)
        last = k
        i = k

    out.append(text[last:])
    return "".join(out), masks


def restore_masks(text: str, masks):
    for idx, original in enumerate(masks):
        text = text.replace(f"@@LATEX_PROTECTED_{idx}@@", original)
    return text


def convert_natural_text(text: str) -> str:
    masked, masks = mask_protected_regions(text)

    for source, target in REPLACEMENTS:
        masked = re.sub(rf"\b{re.escape(source)}\b", target, masked)

    return restore_masks(masked, masks)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python latex_british_english_safe.py input.tex output.tex", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    text = src.read_text(encoding="utf-8")
    converted = convert_natural_text(text)
    dst.write_text(converted, encoding="utf-8")

    print(f"Wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
