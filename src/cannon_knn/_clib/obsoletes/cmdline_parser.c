/**@<cmdline_parser.c>::**/

char           *cmdname;
char           *inputfname;
char           *outputprefix;

FILE           *cmdline_parser(int argc, const char *argv[])
{

    FILE           *input = stdin;

    int             n;

    int             arg_index;

    cmdname = basename((char *) argv[0]);

    int             just_entered_flag = arg_index = 1;

    do {
        just_entered_flag = 0;

        if (strcmp(argv[arg_index], "--input") == 0) {

            ++arg_index;
            if ((input = fopen(argv[arg_index], "r")) == NULL) {
                fprintf(stderr, inputerr, cmdname, argv[arg_index]);
                exit(INPUTERR);
            }
            inputfname = strdup(argv[arg_index]);
            size_t          outputprefixlen = strlen(inputfname) - strlen(strrchr(inputfname, '.'));

            outputprefix = strncat(calloc(outputprefixlen, 1), inputfname, outputprefixlen);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--n") == 0) {

            ++arg_index;
            n = atoi(argv[arg_index]);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--dim") == 0) {

            ++arg_index;
            dimension = atoi(argv[arg_index]);
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--help") == 0) {

            usage(stderr, argv[0]);
            exit(0);

        } else if (strcmp(argv[arg_index], "--K") == 0) {

            ++arg_index;
            K = atoi(argv[arg_index]);
            if (K < 1) {
                fprintf(stderr, wrongK, cmdname, K);
                exit(WRONGK);
            }
            ++arg_index;
            just_entered_flag++;

        } else if (strcmp(argv[arg_index], "--K0") == 0) {
            ++arg_index;
            K0 = atoi(argv[arg_index]);
            if (K0 < 1) {
                fprintf(stderr, wrongK, cmdname, K0);
                exit(WRONGK);
            }
            ++arg_index;
            just_entered_flag++;
        }

    } while (just_entered_flag && arg_index < argc);

    return input;
}
