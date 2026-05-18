#!/usr/bin/env python3
"""Analyze C/C++ include dependencies under src/cannon_knn/_clib.

Parses local `#include "..."` directives, builds a dependency graph, finds
files that depend (directly or transitively) on clusterer-related headers,
flags redundant local includes, detects cycles, and writes a Graphviz DOT
diagram plus JSON and text reports for use from Make.

Run:
    python3 scripts/clib_deps.py [--root DIR] [--out DIR]

The script is intentionally a lightweight static analyzer: it does NOT run a
real C preprocessor, so conditional compilation (#if/#ifdef), macro-based
include paths, and platform-specific guards are NOT evaluated. Findings are
reported on a best-effort basis; see the "Limitations" section in the
generated text report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".c++"}
HEADER_EXTS = {".h", ".hh", ".hpp", ".hxx", ".h++", ".inl", ".tcc"}
CODE_EXTS = SOURCE_EXTS | HEADER_EXTS

# Match `#include "..."` allowing leading whitespace. We deliberately ignore
# `#include <...>` (system / external headers) since the task scope is local
# dependencies inside _clib.
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"')

# Heuristic: a file is "clusterer-related" if its base name contains any of
# these tokens or its content defines a symbol whose name contains them.
CLUSTERER_TOKENS = ("cluster", "clusterer", "clustering")
CLUSTERER_SYMBOL_RE = re.compile(
    r"\b(?:typedef|struct|class|enum|union)\b[^;{]*\b\w*cluster\w*\b",
    re.IGNORECASE,
)


@dataclass
class FileNode:
    path: Path  # absolute
    rel: str  # relative to root, posix-style
    direct: list[str] = field(default_factory=list)  # rel paths, in source order
    raw_includes: list[str] = field(default_factory=list)  # original strings
    unresolved: list[str] = field(default_factory=list)
    is_clusterer: bool = False
    clusterer_reason: str = ""


def strip_comments(text: str) -> str:
    """Remove /* ... */ and // ... comments. Good enough for #include scanning."""
    # Block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r"//[^\n]*", "", text)
    return text


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in CODE_EXTS:
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def detect_clusterer(path: Path, text_no_comments: str) -> tuple[bool, str]:
    name = path.name.lower()
    for tok in CLUSTERER_TOKENS:
        if tok in name:
            return True, f"filename contains '{tok}'"
    m = CLUSTERER_SYMBOL_RE.search(text_no_comments)
    if m:
        return True, f"defines symbol matching /{CLUSTERER_SYMBOL_RE.pattern}/"
    return False, ""


def resolve_include(spec: str, including_file: Path, root: Path) -> Path | None:
    """Resolve an `#include "spec"` against the including file's directory,
    falling back to the project root. Returns an absolute, resolved Path that
    exists, or None.
    """
    candidates = [
        including_file.parent / spec,
        root / spec,
    ]
    for c in candidates:
        try:
            rc = c.resolve()
        except OSError:
            continue
        if rc.is_file():
            # Keep within root to avoid pulling in arbitrary system files that
            # happen to share names (defensive — shouldn't normally trigger).
            try:
                rc.relative_to(root.resolve())
            except ValueError:
                continue
            return rc
    return None


def parse_file(path: Path, root: Path) -> FileNode:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"warning: could not read {path}: {e}", file=sys.stderr)
        text = ""
    clean = strip_comments(text)
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    node = FileNode(path=path.resolve(), rel=rel)
    seen_rel: set[str] = set()
    for line in clean.splitlines():
        m = INCLUDE_RE.match(line)
        if not m:
            continue
        spec = m.group(1).strip()
        node.raw_includes.append(spec)
        resolved = resolve_include(spec, path, root)
        if resolved is None:
            node.unresolved.append(spec)
            continue
        rrel = resolved.relative_to(root.resolve()).as_posix()
        if rrel in seen_rel:
            # Duplicate after normalization is recorded by leaving it out of
            # `direct` while still present in `raw_includes`; the redundancy
            # detector flags it as a duplicate.
            continue
        seen_rel.add(rrel)
        node.direct.append(rrel)
    is_c, reason = detect_clusterer(path, clean)
    node.is_clusterer = is_c
    node.clusterer_reason = reason
    return node


def build_graph(nodes: dict[str, FileNode]) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {rel: [] for rel in nodes}
    for rel, node in nodes.items():
        for dep in node.direct:
            if dep in nodes:  # only intra-_clib edges
                g[rel].append(dep)
    return g


def transitive_closure(g: dict[str, list[str]]) -> dict[str, set[str]]:
    """For each node, the set of nodes reachable from it (excluding itself
    unless a cycle reaches back)."""
    closure: dict[str, set[str]] = {}
    for start in g:
        seen: set[str] = set()
        stack = list(g[start])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(g.get(n, ()))
        closure[start] = seen
    return closure


def reachable_excluding_edge(
    g: dict[str, list[str]], src: str, skip_target: str
) -> set[str]:
    """Reachable set from src if we remove the single direct edge src->skip_target.

    Other edges (including parallel paths through different direct includes)
    are preserved. Used to test whether a redundant include is *truly*
    reachable through another route.
    """
    seen: set[str] = set()
    # Seed with direct neighbors except the skipped edge.
    queue: deque[str] = deque(n for n in g.get(src, ()) if n != skip_target)
    while queue:
        n = queue.popleft()
        if n in seen:
            continue
        seen.add(n)
        queue.extend(g.get(n, ()))
    return seen


def find_redundant_includes(
    nodes: dict[str, FileNode], g: dict[str, list[str]]
) -> list[dict]:
    """A direct include A->B is redundant if, after removing that single edge,
    B is still reachable from A through some other direct include.

    Also reports textual duplicate includes (same spec appearing >1 times) and
    duplicates that normalize to the same file.
    """
    findings: list[dict] = []
    for rel, node in nodes.items():
        # 1. Transitive redundancy
        for dep in node.direct:
            other_reach = reachable_excluding_edge(g, rel, dep)
            if dep in other_reach:
                # Find a witnessing path through one of the other direct deps.
                witness = None
                for od in node.direct:
                    if od == dep:
                        continue
                    seen: set[str] = set()
                    stack = [od]
                    while stack:
                        x = stack.pop()
                        if x == dep:
                            witness = od
                            break
                        if x in seen:
                            continue
                        seen.add(x)
                        stack.extend(g.get(x, ()))
                    if witness:
                        break
                findings.append(
                    {
                        "file": rel,
                        "kind": "transitive",
                        "redundant_include": dep,
                        "reachable_via": witness,
                        "explanation": (
                            f"'{rel}' includes '{dep}' directly, but already "
                            f"reaches it transitively via '{witness}'."
                        ),
                    }
                )

        # 2. Textual / normalized duplicates
        raw_counts: dict[str, int] = defaultdict(int)
        for spec in node.raw_includes:
            raw_counts[spec] += 1
        for spec, count in raw_counts.items():
            if count > 1:
                findings.append(
                    {
                        "file": rel,
                        "kind": "duplicate-spec",
                        "redundant_include": spec,
                        "count": count,
                        "explanation": (
                            f"'{rel}' contains the literal include "
                            f"\"{spec}\" {count} times."
                        ),
                    }
                )

        # Different spec strings that normalize to the same target.
        normalized: dict[str, list[str]] = defaultdict(list)
        for spec in node.raw_includes:
            r = resolve_include(spec, node.path, node.path.parents[0])
            # Re-resolve relative to graph root via stored direct entry when possible.
            # The above call uses the file's own dir as root; that's fine for
            # detecting equivalence — same spec resolving to same path is what
            # we care about.
            key = str(r) if r else spec
            normalized[key].append(spec)
        for key, specs in normalized.items():
            unique_specs = sorted(set(specs))
            if len(unique_specs) > 1:
                findings.append(
                    {
                        "file": rel,
                        "kind": "duplicate-normalized",
                        "redundant_include": unique_specs,
                        "explanation": (
                            f"'{rel}' has multiple include forms that resolve "
                            f"to the same file: {unique_specs}."
                        ),
                    }
                )
    return findings


def find_cycles(g: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's SCC; report any SCC of size >1 or self-loops."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in g.get(v, ()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    sys.setrecursionlimit(max(1000, len(g) * 4 + 100))
    for v in list(g.keys()):
        if v not in indices:
            strongconnect(v)

    cycles: list[list[str]] = []
    for comp in sccs:
        if len(comp) > 1:
            cycles.append(sorted(comp))
        elif len(comp) == 1:
            v = comp[0]
            if v in g.get(v, ()):
                cycles.append([v])
    return cycles


def find_clusterer_dependents(
    nodes: dict[str, FileNode], closure: dict[str, set[str]]
) -> dict[str, dict]:
    clusterer_files = {rel for rel, n in nodes.items() if n.is_clusterer}
    out: dict[str, dict] = {}
    for rel, node in nodes.items():
        if rel in clusterer_files:
            continue
        direct_hits = sorted(set(node.direct) & clusterer_files)
        transitive_hits = sorted((closure.get(rel, set()) & clusterer_files) - set(direct_hits))
        if direct_hits or transitive_hits:
            out[rel] = {
                "direct": direct_hits,
                "transitive": transitive_hits,
            }
    return out


def write_dot(
    nodes: dict[str, FileNode],
    g: dict[str, list[str]],
    cycles: list[list[str]],
    clusterer_dependents: dict[str, dict],
    out_path: Path,
) -> None:
    cycle_nodes: set[str] = set()
    for c in cycles:
        cycle_nodes.update(c)
    clusterer_set = {rel for rel, n in nodes.items() if n.is_clusterer}
    dependent_set = set(clusterer_dependents.keys())

    lines: list[str] = []
    lines.append("digraph clib_deps {")
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=box, fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')
    for rel in sorted(nodes):
        attrs = []
        if rel in clusterer_set:
            attrs.append('fillcolor="#ffd6d6"')
            attrs.append('style="filled,bold"')
        elif rel in dependent_set:
            attrs.append('fillcolor="#fff5cc"')
            attrs.append('style=filled')
        if rel in cycle_nodes:
            attrs.append('color="#cc0000"')
            attrs.append('penwidth=2')
        attrs.append(f'label="{rel}"')
        lines.append(f'  "{rel}" [{", ".join(attrs)}];')
    cycle_edges: set[tuple[str, str]] = set()
    for comp in cycles:
        comp_set = set(comp)
        for u in comp:
            for v in g.get(u, ()):
                if v in comp_set:
                    cycle_edges.add((u, v))
    for u in sorted(g):
        for v in g[u]:
            if (u, v) in cycle_edges:
                lines.append(f'  "{u}" -> "{v}" [color="#cc0000", penwidth=2];')
            else:
                lines.append(f'  "{u}" -> "{v}";')
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_png(dot_path: Path, png_path: Path) -> tuple[bool, str]:
    if shutil.which("dot") is None:
        return False, "graphviz 'dot' executable not found on PATH"
    try:
        proc = subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"dot failed to execute: {e}"
    if proc.returncode != 0:
        return False, f"dot exited {proc.returncode}: {proc.stderr.strip()}"
    return True, str(png_path)


def write_text_report(
    root: Path,
    nodes: dict[str, FileNode],
    g: dict[str, list[str]],
    redundant: list[dict],
    cycles: list[list[str]],
    clusterer_dependents: dict[str, dict],
    dot_path: Path,
    png_status: tuple[bool, str],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("CANNON KNN — _clib include-dependency report")
    lines.append("=" * 60)
    lines.append(f"Scanned root : {root}")
    lines.append(f"Files scanned: {len(nodes)}")
    lines.append(f"Edges        : {sum(len(v) for v in g.values())}")
    lines.append("")
    if not nodes:
        lines.append("No C/C++ files found under the scan root. The _clib")
        lines.append("directory is either missing or empty in this branch.")
        lines.append("Reports below are therefore empty by construction.")
        lines.append("")

    clusterer_files = sorted(rel for rel, n in nodes.items() if n.is_clusterer)
    lines.append("Clusterer-defining files")
    lines.append("-" * 60)
    if clusterer_files:
        for rel in clusterer_files:
            lines.append(f"  - {rel}  ({nodes[rel].clusterer_reason})")
    else:
        lines.append("  (none detected)")
    lines.append("")

    lines.append("Files depending on clusterer definitions")
    lines.append("-" * 60)
    if clusterer_dependents:
        for rel in sorted(clusterer_dependents):
            info = clusterer_dependents[rel]
            lines.append(f"  - {rel}")
            lines.append(f"      direct    : {info['direct'] or '(none)'}")
            lines.append(f"      transitive: {info['transitive'] or '(none)'}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Redundant local includes")
    lines.append("-" * 60)
    if redundant:
        for f in redundant:
            lines.append(f"  - [{f['kind']}] {f['file']}: {f['explanation']}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Circular include dependencies")
    lines.append("-" * 60)
    if cycles:
        for c in cycles:
            lines.append(f"  - cycle: {' -> '.join(c + [c[0]])}")
    else:
        lines.append("  (none)")
    lines.append("")

    unresolved: list[tuple[str, str]] = []
    for rel, node in nodes.items():
        for u in node.unresolved:
            unresolved.append((rel, u))
    lines.append("Unresolved local includes (informational)")
    lines.append("-" * 60)
    if unresolved:
        for f, u in unresolved:
            lines.append(f"  - {f} : \"{u}\"")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Artifacts")
    lines.append("-" * 60)
    lines.append(f"  DOT diagram : {dot_path}")
    if png_status[0]:
        lines.append(f"  PNG render  : {png_status[1]}")
    else:
        lines.append(f"  PNG render  : not produced ({png_status[1]})")
    lines.append("")

    lines.append("Limitations")
    lines.append("-" * 60)
    lines.append("  * The parser does not run a real C preprocessor; it ignores")
    lines.append("    #if/#ifdef/#else conditionals and macro-built include paths.")
    lines.append("  * Only #include \"...\" (local) directives are tracked; system")
    lines.append("    headers (#include <...>) are ignored by design.")
    lines.append("  * Redundancy detection assumes textual inclusion semantics.")
    lines.append("    A header guarded behind a feature flag may be falsely")
    lines.append("    flagged or missed.")
    lines.append("  * Clusterer detection is heuristic (filename token or")
    lines.append("    typedef/struct/class/enum/union with a 'cluster' identifier).")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_report(
    root: Path,
    nodes: dict[str, FileNode],
    g: dict[str, list[str]],
    redundant: list[dict],
    cycles: list[list[str]],
    clusterer_dependents: dict[str, dict],
    dot_path: Path,
    png_status: tuple[bool, str],
    out_path: Path,
) -> None:
    payload = {
        "root": str(root),
        "files": sorted(nodes.keys()),
        "edges": {rel: sorted(g[rel]) for rel in sorted(g)},
        "clusterer_files": [
            {"path": rel, "reason": nodes[rel].clusterer_reason}
            for rel in sorted(nodes)
            if nodes[rel].is_clusterer
        ],
        "clusterer_dependents": clusterer_dependents,
        "redundant_includes": redundant,
        "cycles": cycles,
        "unresolved": {
            rel: list(nodes[rel].unresolved)
            for rel in sorted(nodes)
            if nodes[rel].unresolved
        },
        "artifacts": {
            "dot": str(dot_path),
            "png": png_status[1] if png_status[0] else None,
            "png_error": None if png_status[0] else png_status[1],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default="src/cannon_knn/_clib",
        help="Directory to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="build/clib-deps",
        help="Output directory for reports (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-cycles",
        action="store_true",
        help="Exit non-zero if any include cycle is found.",
    )
    parser.add_argument(
        "--fail-on-redundant",
        action="store_true",
        help="Exit non-zero if any redundant include is found.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(
            f"note: scan root '{root}' does not exist yet; emitting empty reports.",
            file=sys.stderr,
        )
        files: list[Path] = []
    else:
        files = discover_files(root)

    # Use a stable scan-root for relative paths even if root is absent.
    scan_root = root if root.exists() else Path.cwd()

    nodes: dict[str, FileNode] = {}
    for p in files:
        node = parse_file(p, scan_root)
        nodes[node.rel] = node

    g = build_graph(nodes)
    closure = transitive_closure(g)
    cycles = find_cycles(g)
    redundant = find_redundant_includes(nodes, g)
    dependents = find_clusterer_dependents(nodes, closure)

    dot_path = out_dir / "clib_deps.dot"
    png_path = out_dir / "clib_deps.png"
    text_path = out_dir / "clib_deps.txt"
    json_path = out_dir / "clib_deps.json"

    write_dot(nodes, g, cycles, dependents, dot_path)
    png_status = render_png(dot_path, png_path)
    write_text_report(
        scan_root, nodes, g, redundant, cycles, dependents,
        dot_path, png_status, text_path,
    )
    write_json_report(
        scan_root, nodes, g, redundant, cycles, dependents,
        dot_path, png_status, json_path,
    )

    print(text_path.read_text(encoding="utf-8"))

    rc = 0
    if args.fail_on_cycles and cycles:
        rc = 2
    if args.fail_on_redundant and redundant:
        rc = max(rc, 3)
    return rc


if __name__ == "__main__":
    sys.exit(main())
