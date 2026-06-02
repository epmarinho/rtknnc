Draft bundle for Springer Nature / Neural Computing and Applications style

Main file:
  round_trip_knn_nca_sn_draft.tex

Bibliography:
  round_trip_knn_refs.bib

Figures:
  figure1_roundtrip_reverse_neighbor.pdf
  fig_2018_knn_graph.pdf
  fig_2018_leadership_graph.pdf
  fig_2018_cluster_tree.pdf
  fig_k_layer_cluster_count.pdf

Important:
  This draft uses \documentclass[pdflatex,sn-mathphys-num]{sn-jnl}.
  It requires the official Springer Nature LaTeX template files, especially:
    sn-jnl.cls
    sn-mathphys-num.bst
  The easiest path is to upload this bundle to Overleaf inside a project created
  from the official Springer Nature template, or copy the official template files
  into the same directory locally.

Suggested compile sequence locally after adding SN template files:
  pdflatex round_trip_knn_nca_sn_draft.tex
  bibtex round_trip_knn_nca_sn_draft
  pdflatex round_trip_knn_nca_sn_draft.tex
  pdflatex round_trip_knn_nca_sn_draft.tex

Notes:
  - Author declarations are placeholders and must be completed.
  - The bibliography is a working draft and must be sanity-checked.
  - The 2013 material is intentionally reduced to relational motivation.
  - The 2018 mechanism is the algorithmic core: kNN forward step + reverse-neighbor discovery + popularity leadership.
