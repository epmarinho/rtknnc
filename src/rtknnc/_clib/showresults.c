/**@<showresults.c>::**/
#include "main.h"
#include "k-nn-topology.h"

// char           *outputformat = "%s_K%zuCs%zu.layer";
char           *outputformat = "%s_K%zuCs%zu_layer.csv";
char           *outputerrmsg = "%s: cannot open output file %s... exiting\n";

static void write_layer_header(void)
{
    fprintf(output, "# format: cannon-knn layer\n");
    fprintf(output, "# columns: cluster_id: point_id");
    for (dim_t j = 0; j < dimension; j++) {
        fprintf(output, " x%d", j);
    }
    fprintf(output, "\n");
}

void showresults(void)
{
    int             fnamelen = snprintf(NULL, 0, outputformat, outputprefix, K, clcount);
    if (fnamelen < 0) {
        fprintf(stderr, "%s: cannot format output file name... exiting\n", cmdname);
        exit(-666);
    }

    char           *outputfname = calloc((size_t)fnamelen + 1, 1);
    if (outputfname == NULL) {
        fprintf(stderr, "%s: cannot allocate output file name... exiting\n", cmdname);
        exit(-666);
    }

    snprintf(outputfname, (size_t)fnamelen + 1, outputformat, outputprefix, K, clcount);

    size_t          cluster_number = clcount;

    output = fopen(outputfname, "w");

    if (output == NULL) {
        fprintf(stderr, outputerrmsg, cmdname, outputfname);
        exit(-666);
    }
    write_layer_header();
    //fprintf(output, "K = %d\n", K);
    for (clcount = 0; clcount < cluster_number; clcount++) {
        if (cluster[clcount]->n > 0) {
            //fprintf(output, "cluster %i with %d data instances ", clcount + 1, cluster[clcount]->n);
            //fprintf(output, "with expectation vector\n");
            //for (int j = 0; j < dimension; j++) {
            //    fprintf(output,"\t%lg", cluster[clcount]->center_of_mass[j]);
            //}
            //fprintf(output, "\n");
            set__list(cluster[clcount]);
            fprintf(output, "\n");
        }
    }

    free(outputfname);
    fclose(output);
}
