/**@<z-norm.c>::**/

#include <stddef.h>
#include <io-stuff.h>
#include <libgen.h>

/** prototypes: **/
//extern int getdata(FILE * input);

extern size_t   n;
double         *xmean;
double         *sigma;

//dim_t    dimension;
void znorm(int n)
{
    xmean = calloc(dimension, sizeof(*xmean));
    sigma = calloc(dimension, sizeof(*sigma));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dimension; j++) {
            xmean[j] += x[i][j];
            sigma[j] += x[i][j] * x[i][j];
        }
    }
    for (int j = 0; j < dimension; j++) {
        xmean[j] /= n;
        sigma[j] /= n;
        sigma[j] -= xmean[j] * xmean[j];
        sigma[j] = sqrt(sigma[j]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dimension; j++) {
            x[i][j] = (x[i][j] - xmean[j]) / sigma[j];
        }
    }
}

int main(int argc, const char *argv[])
{
    n = getdata(cmdline_parser(argc, argv));

    znorm(n);

    for (int i = 0; i < n; i++) {
        fprintf(stdout, "%20.15le", x[i][0]);
        for (int j = 1; j < dimension; j++) {
            fprintf(stdout, "\t%20.15le", x[i][j]);
        }
        fprintf(stdout, "\n");
    }

    return 0;
}
