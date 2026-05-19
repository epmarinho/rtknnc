/**@<main.c>::*/

/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 **********************************/

#include "main.h"

int main(int argc, char const *argv[])
{

    input = cmdline_parser(argc, argv);
    if (input == NULL) {
        usage(stderr, argv[0]);
        fprintf(stderr, "%s: cannot open input file or missing required arguments. Exiting\n", argv[0]);
        exit(-667);
    }

    if (K < K0) {
        fprintf(stdout, "%s: K0=%zu cannot be greater than K=%zu... exiting\n", argv[0], K0, K);
        exit(-666);
    }

    n = getdata(input);

    grade = 1 << dimension;

    if (K > n - 1) {
        fprintf(stderr, "%s: maximum number of neighbors ", basename((char *) argv[0]));
        fprintf(stderr, "K=%zu cannot be greater than or equal to n=%zu...", K, n);
        fprintf(stderr, " exiting\n");
        exit(-666);
    }

    fprintf(stdout, "main: %zd points in a %d-dimension vector space\n", n, dimension);

    fprintf(stdout, "main: %zu through %zu nearest neighbors required by user to perform the cluster analysis\n\n", K0, K);

    fprintf(stdout, "main: calling %zd-branch hyperquadtree_build(%zd)\n", grade, n);
    hyperquadtree_build(n);

    fprintf(stdout, "main: calling knn_search(%zd) to find the %zu nearest neighbors\n", n, K);
    knn_search(n);

    fprintf(stdout, "main: calling hyperquadtree_destroy() to release the hyperquadtree memory space\n");
    hyperquadtree_destroy();

    fprintf(stdout, "main: calling roundtrip_KNN_cluster_partition(%zd) to perform the hierarquical cluster analysis\n", n);
    roundtrip_KNN_cluster_partition(n);

    exit(0);
}
