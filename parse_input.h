/* This file was created, because I only wanted to use the
   input file parsing and nothing else. */

#ifndef PARSE_PAK_H
#define PARSE_PAK_H

#ifdef __cplusplus
extern "C" {
#endif
  
#include <stdio.h>
#include <stdlib.h>

/* parameters */
#define ALWAYS 0      /* required */
#define OPTION 1      /* optional */
#define OPTION2 2     /* optional, doesn't require an argument */

#define OERROR -1
#define DISK 0
#define PIPE 1
#define STDIO 2

struct fdesc {
  FILE *p;   /* Actual pointer to file */
  signed char type; /* Type, tells us should we use fclose or pclose */
};
 
long pi_oatoi(const char *str, long def);
float pi_oatof(const char *str, float def);
char *pi_extract_parameter(int argc, char **argv, char *param, int when);
int pi_parameters_left(void);
struct fdesc pi_open_file(int argc,char **argv,char *argkey, int when,char *mode,
		   int verbose);
struct fdesc pi_fopen(const char *,const char *);
void pi_fclose(struct fdesc);

#ifdef __cplusplus
}
#endif
#endif /* PARSE_PAK_H */
