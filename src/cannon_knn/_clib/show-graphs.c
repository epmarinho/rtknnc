/**@<show-graphs.c>::**/
#include "main.h"

#ifdef __GRAPHVIZ_OUTPUT_SYNTAX__

#    include <stdio.h>
#    include <constants.h>
#    include <k-nn.h>
#    include <show-KNN-graph.h>

char           *outputformatgv = "%s_K%zu.gv";
char           *outputerrmsggv = "%s: cannot open output file %s... exiting\n";

double         *received;

// #define MAPSCALE 32*sqrt((double)n)
// #define NBSCALE .1

void show_KNN_graph(int n)
{
    int             fnamelen = snprintf(NULL, 0, outputformatgv, outputprefix, K);
    if (fnamelen < 0) {
        fprintf(stderr, "%s: cannot format output file name... exiting\n", cmdname);
        exit(-666);
    }

    char           *outputfname = calloc((size_t)fnamelen + 1, 1);
    if (outputfname == NULL) {
        fprintf(stderr, "%s: cannot allocate output file name... exiting\n", cmdname);
        exit(-666);
    }

    snprintf(outputfname, (size_t)fnamelen + 1, outputformatgv, outputprefix, K);

    output = fopen(outputfname, "w");

    if (output == NULL) {
        fprintf(stderr, outputerrmsggv, cmdname, outputfname);
        exit(-666);
    }

    received = calloc(n, sizeof(info_t));

    /** Setup graphviz header to plot a k-NN graph showing a
        geographical map highlighting clusters. **/
    fprintf(output, "digraph k_NN {\n");
    fprintf(output, "\tgraph [charset=latin1, overlap_scaling=3, pack=90, ");
    fprintf(output, "label=\"kNN connection map for n=%d and k=%zu\"];\n", n, K);
    fprintf(output, "\tnode [fontsize=\"7\", shape=circle, label=\"\\N\", ");
    fprintf(output, "width=\"0.001\", height=\"0.001\", margin=\"0.001\"];\n");
    fprintf(output, "\tedge [arrowsize=\".5\"];\n");
    fprintf(output, "\tgraph[bb=\"0,0,1024,1024\"];\n");

    index_t         i;

    for (i = 0; i < n; i++) {

        ID_t            query_ID = i + 1;

        int             j;

        for (j = 0; j < K; j++) {

            ID_t            jID = KNN[i].list[j];
            index_t         l = jID - 1;

            fprintf(output,
                    "\t%i [width=\"%g\", height=\"%g\", pos=\"%g,%g\"];\n",
                    query_ID, NBSCALE * KNN[i].radius, NBSCALE * KNN[i].radius, MAPSCALE * (2 + x[i][0]), MAPSCALE * (2 + x[i][1]));
            fprintf(output,
                    "\t%i [width=\"%g\", height=\"%g\", pos=\"%g,%g\"];\n",
                    jID, NBSCALE * KNN[l].radius, NBSCALE * KNN[l].radius, MAPSCALE * (2 + x[l][0]), MAPSCALE * (2 + x[l][1]));

#    ifdef __SHOW_EDGE_LABELS__
            fprintf(output, "\tedge [label=\"%2.2g\"];\n", KNN[i].d[j]);
#    endif

            fprintf(output, "\t%i -> %i;\n", query_ID, jID);
        }
    }
    fprintf(output, "}\n");
    free(received);
    received = NULL;
    free(outputfname);
    fclose(output);
}
#endif
