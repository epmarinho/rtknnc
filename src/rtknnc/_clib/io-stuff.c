/***@<io-stuff.c>::**/
#include "main.h"

/***********************************
 * AUTHOR: Eraldo Pereira Marinho *
 **********************************/
#include <stdio.h>
#include <io-stuff.h>
#include <stdlib.h>

/*** input and output file-handler pointers ***/
FILE           *input;
FILE           *output;

enum {
    DATA_PARSEERR = -9999
};

/*** the maximum number of nearest neighbors ***/
size_t          K = 1;

/*** the lowest number of nearest neighbor connections to start the roundtripKNNfind() ***/
size_t          K0 = 1;

/*** char-string pointers to file name composition ***/
char           *cmdname;
char           *inputfname;
char           *inputprefix;
char           *outputprefix;

/*** the data-set properly said: ***/
info_t        **x;

/*** dimension comes out from the comand line or is changed by guessdataboundaries ***/
dim_t           dimension;

/*** grade = 1 << dimension; it is the number of branches from an internal hyperquatree node ***/
size_t          grade;

/**** N is a guessdataboundaries()'s public attribute to store the number of intergal lines read: ***/
size_t          N = 0;

/*** n is the minimum of either used defined or N ***/
size_t          n;

char           *wrongK = "%s:\tthe number of nearest neighbors must be positive integer\n\tbut you've passed --K %d\n";
char           *inputerr = "%s: cannot open %s or file does not exist... exiting\n";

void            usage(FILE *output, const char *cmdname)
{
    /*
     * Expected input is a numerical table separated by whitespace, commas,
     * semicolons, or colons. The parser needs at least the input path and
     * the data dimensionality to interpret the file correctly.
     */
    fprintf(output, "usage: %s --input <datafile> --dim <dimension> [--n <points>] [--K <neighbors>] [--K0 <neighbors>] [--vote-kernel uniform|inv_r|inv_r2|sigmoid] [--sigmoid-alpha <value>] [--structural-pruning]\n", cmdname);
    fprintf(output, "\n");
    fprintf(output, "  --input  input data file\n");
    fprintf(output, "  --dim    data-space dimension\n");
    fprintf(output, "  --n      number of points to read; defaults to the full dataset\n");
    fprintf(output, "  --K      maximum number of nearest neighbors; default is 1\n");
    fprintf(output, "  --K0     minimum neighborhood level for the clustering roundtrip; default is 1\n");
    fprintf(output, "  --vote-kernel  source-budget-matched shape: uniform, inv_r, inv_r2 (default), or sigmoid\n");
    fprintf(output, "                 every source redistributes B_i=sum_j 1/r_ij^2; inv_r2 remains historical exactly\n");
    fprintf(output, "  --sigmoid-alpha  positive sigmoid slope; default is 8\n");
    fprintf(output, "  --structural-pruning  split strongly multimodal closures and prune their crossing visible edges\n");
}

FILE           *cmdline_parser(int argc, const char *argv[])
{
    FILE           *input = stdin;
    int             arg_index;

    structural_pruning = 0;
    vote_kernel = VOTE_KERNEL_INV_R2;
    vote_sigmoid_alpha = 8.;

    if (argc < 2) {
        usage(stderr, argv[0]);
        return NULL;
    }

    cmdname = basename((char *) argv[0]);
    int             just_entered_flag = arg_index = 1;

    do {
        just_entered_flag = 0;

        if (strcmp(argv[arg_index], "--input") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            if ((input = fopen(argv[arg_index], "r")) == NULL) {
                fprintf(stderr, inputerr, cmdname, argv[arg_index]);
                exit(INPUTERR);
            }
            inputfname = strdup(argv[arg_index]);
            inputprefix = strrchr(inputfname, '.');
            size_t          outputprefixlen = inputprefix ? (size_t) (inputprefix - inputfname) : strlen(inputfname);

            //outputprefix = strncat(calloc(outputprefixlen, 1), inputfname, outputprefixlen);
            //outputprefix = strcat(calloc(outputprefixlen + 10, 1), inputprefix);
            outputprefix = calloc(outputprefixlen + 1, 1);
            strncpy(outputprefix, inputfname, outputprefixlen);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--n") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            n = atoi(argv[arg_index]);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--dim") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            dimension = atoi(argv[arg_index]);
//             fprintf(stdout, "io-stuff.c: dimension = %d\n", dimension);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--structural-pruning") == 0) {

            structural_pruning = 1;
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--vote-kernel") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            if (strcmp(argv[arg_index], "uniform") == 0)
                vote_kernel = VOTE_KERNEL_UNIFORM;
            else if (strcmp(argv[arg_index], "inv_r") == 0)
                vote_kernel = VOTE_KERNEL_INV_R;
            else if (strcmp(argv[arg_index], "inv_r2") == 0)
                vote_kernel = VOTE_KERNEL_INV_R2;
            else if (strcmp(argv[arg_index], "sigmoid") == 0)
                vote_kernel = VOTE_KERNEL_SIGMOID;
            else {
                fprintf(stderr,
                        "%s: unknown vote kernel '%s'; expected uniform, inv_r, inv_r2, or sigmoid\n",
                        cmdname, argv[arg_index]);
                return NULL;
            }
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--sigmoid-alpha") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            vote_sigmoid_alpha = strtod(argv[arg_index], NULL);
            if (!isfinite(vote_sigmoid_alpha) || vote_sigmoid_alpha <= 0.) {
                fprintf(stderr,
                        "%s: --sigmoid-alpha must be a positive finite number\n",
                        cmdname);
                return NULL;
            }
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--help") == 0 || strcmp(argv[arg_index], "-h") == 0) {

            usage(stderr, argv[0]);
            exit(0);

        } else if (strcmp(argv[arg_index], "--K") == 0) {

            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            K = atoi(argv[arg_index]);
            if (K < 1) {
                fprintf(stderr, wrongK, cmdname, K);
                exit(WRONGKERR);
            }
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--K0") == 0) {
            ++arg_index;
            if (arg_index >= argc) {
                usage(stderr, cmdname);
                return NULL;
            }
            K0 = atoi(argv[arg_index]);
            if (K0 < 1) {
                fprintf(stderr, wrongK, cmdname, K0);
                exit(WRONGKERR);
            }
            ++arg_index;
            just_entered_flag++;
        }

    } while (just_entered_flag && arg_index < argc);

    return input;
}

int seekeol(FILE * input)
{
    int             c;

    while ((c = getc(input)) != '\n' && c != EOF);
    return c;
}

static int isdataseparator(int c)
{
    return isspace(c) || c == ',' || c == ';' || c == ':';
}

int skipseparators(FILE * input)
{
    int             c;

    while (isdataseparator(c = getc(input))) {
        if (c == '\n') {

            /*** \n has the semantic of line counting ***/
            return c;
        }
    }

    /*** check eof ***/
    if (c == EOF) {
        return c;
    }

    ungetc(c, input);
    return c;
}

static index_t count_numeric_columns(const char *line)
{
    char           *endptr;
    const char     *p = line;
    index_t         dim = 0;

    while (*p) {
        while (isdataseparator((unsigned char) *p))
            ++p;

        if (*p == '\0' || *p == '#')
            break;

        strtod(p, &endptr);
        if (endptr == p)
            return dim == 0 ? 0 : DATA_PARSEERR;

        ++dim;
        p = endptr;

        if (*p && !isdataseparator((unsigned char) *p) && *p != '#')
            return DATA_PARSEERR;
    }

    return dim;
}

static int read_numeric_columns(const char *line, info_t * point, index_t dim)
{
    char           *endptr;
    const char     *p = line;
    index_t         j = 0;

    while (*p && j < dim) {
        while (isdataseparator((unsigned char) *p))
            ++p;

        if (*p == '\0' || *p == '#')
            break;

        point[j] = strtod(p, &endptr);
        if (endptr == p)
            return 0;

        ++j;
        p = endptr;

        if (*p && !isdataseparator((unsigned char) *p) && *p != '#')
            return 0;
    }

    return j == dim;
}

index_t guessdataboundaries(FILE * input)
{
    char            line[65536];
    index_t         dim = 0;

    N = 0;

    while (fgets(line, sizeof(line), input)) {
        index_t         line_dim = count_numeric_columns(line);

        if (line_dim == DATA_PARSEERR) {
            rewind(input);
            return SNGPTERR;
        }
        if (line_dim == 0)
            continue;

        dim = dim > line_dim ? dim : line_dim;
        ++N;
    }

    /*** rewind the file offset once known the data-set dimensions ***/
    rewind(input);

    /*** return the inferred data dimensionality ***/
    return dim;
}

size_t getdata(FILE * input)
{
    char            line[65536];
    index_t         i = 0;

    /*** guessdataboundaries return the number of dimensions and stores in a global variable, size_t N, the
         number of lines read from the data-set ***/
    index_t         ncol = guessdataboundaries(input);

    /*** ncol represents the maximum data dimensionality ***/
    if (ncol == SNGPTERR) {
        fprintf(stderr, "@getdata: data-set must have more than one point... exiting with error status\n");
        exit(ncol);
    }

    /*** if the user has not sent the command line message --dim <dimension>, or if it was intentionally passed as a
         non-positive value, then it must be redimensioned ***/
    dimension = nzmin(dimension, ncol);
    grade = 1 << dimension;  // ← aqui, uma vez só, com o valor final de dimension  

    /*** similarly to the expected number of data points, maybe entered by --n <n> ***/
    n = nzmin(n, N);

    /*** allocate enough room space to get the required/available number of data points... ***/
    x = calloc(n, sizeof(*x));

    /*** ... regarding the required/available dimensionality ***/
    while (i < n && fgets(line, sizeof(line), input)) {
        index_t         line_dim = count_numeric_columns(line);

        if (line_dim == 0)
            continue;
        if (line_dim == DATA_PARSEERR || line_dim < dimension) {
            fprintf(stderr, "@getdata: invalid data row or fewer columns than requested dimension=%d\n", dimension);
            exit(SNGPTERR);
        }

        x[i] = calloc(dimension, sizeof(double));

        if (!read_numeric_columns(line, x[i], dimension)) {
            fprintf(stderr, "@getdata: could not parse input row %d\n", i);
            exit(SNGPTERR);
        }
        ++i;
    }
    return n;
}
