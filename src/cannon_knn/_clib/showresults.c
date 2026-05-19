/**@<showresults.c>::**/
#include "main.h"
#include "k-nn-topology.h"

char           *outputformat = "%s_K%dCs%d.layer";
char           *outputerrmsg = "%s: cannot open output file %s... exiting\n";

void showresults(void)
{
    char           *outputfname = calloc(strlen(outputprefix) + strlen(outputformat) + 10, 1);
    int             cluster_number = clcount;

    sprintf(outputfname, outputformat, outputprefix, K, clcount);

    output = fopen(outputfname, "w");

    if (output == NULL) {
        fprintf(stderr, outputerrmsg, cmdname, outputfname);
        exit(-666);
    }
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
