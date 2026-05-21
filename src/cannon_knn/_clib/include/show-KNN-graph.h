/**@<show-KNN-graph.h>::**/

#pragma once

#ifdef __GRAPHVIZ_OUTPUT_SYNTAX__

#    ifndef MAPSCALE
#        define MAPSCALE 256
#    endif
#    ifndef NBSCALE
#        define NBSCALE .03125*.03125
#    endif

/** Hereafter are performed the graphs facilities
    whose output are fetched to comply the GraphViz
    syntax **/

extern double  *received;

#endif                          //__GRAPHVIZ_OUTPUT_SYNTAX__

void            show_KNN_graph(int n);
