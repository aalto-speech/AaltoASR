/* These functions are ripped from lvq_pak, the original copyright
   from the lvq_pak.c follows:*/


/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  lvq_pak.c                                                           *
 *   -very general routines needed in many places                       *
 *                                                                      *
 *  Version 3.0                                                         *
 *  Date: 1 Mar 1995                                                    *
 *                                                                      *
 *  NOTE: This program package is copyrighted in the sense that it      *
 *  may be used for scientific purposes. The package as a whole, or     *
 *  parts thereof, cannot be included or used in any commercial         *
 *  application without written permission granted by its producents.   *
 *  No programs contained in this package may be copied for commercial  *
 *  distribution.                                                       *
 *                                                                      *
 *  All comments  concerning this program package may be sent to the    *
 *  e-mail address 'lvq@cochlea.hut.fi'.                                *
 *                                                                      *
 ************************************************************************/

/*#include <stdio.h>*/
#include "string.h"
#include <float.h>
#include "parse_input.h"

/******************************************************************* 
 * Routines to get the parameter values                            * 
 *******************************************************************/

static int no_parameters = -1;

int pi_parameters_left(void)
{
  return(no_parameters);
}

long pi_oatoi(const char *str, long def)
{
  if (str == (char *) NULL)
    return(def);
  else
    return(atoi(str));
}

float pi_oatof(const char *str, float def)
{
  if (str == (char *) NULL)
    return(def);
  else
    return((float) atof(str));
}

char *pi_extract_parameter(int argc, char **argv, char *param, int when)
{
  int i = 0;

  if (no_parameters == -1)
    no_parameters = argc - 1;

  while ((i < argc) && (strcmp(param, argv[i]))) {
    i++;
  }

  if ((i <= argc - 1) && (when == OPTION2))
    {
      no_parameters -= 1;
      return "";
    }

  if (i < argc-1) {
    no_parameters -= 2;
    return(argv[i+1]);
  }
  else {
    if (when == ALWAYS) {
      fprintf(stderr, "Can't find asked option %s\nIf you are lucky, -h or -help could work.\n", param);
      exit(-1);
    }
  }

  return((char *) NULL);
}

/* End of lvq_pak rip*/

/* Here I have added some of my own functions */

#define UNCOMPRESS "gzcat"
#define COMPRESS "gzip >"

/* use pi_close(FILE) to close files opend by this function */ 
struct fdesc pi_fopen(const char *cp, const char *mode) {
  char *buf;

  /* Open only one of these! */
  static int stdin_used=0;
  static int stdout_used=0;
  
  struct fdesc fd;
  
  /* stdin or stdout ? */
  if (!strcmp(cp,"-")) {
    if (!strcmp(mode,"r")) {
      if (stdin_used++) {
	fprintf(stderr,"Could not open stdin (%dth)\n",stdin_used);
	exit(-1);
      }
      fd.p=stdin;
      fd.type=STDIO;
      return(fd);
    }  
    if (!strcmp(mode,"w")) {
      if (stdout_used++) {
	fprintf(stderr,"Will not open stdout (%dth)\n",stdout_used);
	exit(-1);
      }
      fd.p=stdout;
      fd.type=STDIO;
      return(fd);
    }
  }
  
  
  /* A pipe ? */
  if (cp[0]=='|') {
    if (!(fd.p=popen(&cp[1],mode))) {
      fprintf(stderr,"Could not open pipe %s\n",cp);
      exit(-1);
    }
    fd.type=PIPE;
    return (fd);
  }

  /* A gzipped file ? */
  if (!strcmp(&cp[strlen(cp)-3],".gz")) {
    buf=(char *) malloc(sizeof(char) * (strlen(UNCOMPRESS)+strlen(cp)+2));
    if (!strcmp(mode,"w")) sprintf(buf,"%s %s",COMPRESS,cp);
    else sprintf(buf,"%s %s",UNCOMPRESS,cp);
    fd.p=popen(buf,mode);
    free(buf);
    if(!fd.p) {
      fprintf(stderr,"Could not (un)compress %s\n",cp);
      exit(-1);
    }
    fd.type=PIPE;
    return(fd);
  }
  
  /* Just an ordinary file ? */
  if (!(fd.p=fopen(cp,mode))) {
    fprintf(stderr,"Could not open %s\n",cp);
    exit(-1);
  }
  fd.type=DISK;
  return(fd);
}  

/* This one opens a file, a gzipped file or a pipe */
/* use close(FILE) to close files opend by this function */ 

struct fdesc pi_open_file(int argc,char **argv,char *argkey, int when,
			    char *mode, int verbose) {
  char *cp;
  if (!(cp = pi_extract_parameter(argc,argv,argkey,when))) {
    struct fdesc fd;
    fd.p=NULL;
    fd.type=OERROR;
    return (fd);
  }

  if (verbose) printf(" Opening %s: %s\n",argkey+1,cp);

  return (pi_fopen(cp,mode));
}

void pi_fclose(struct fdesc fd) {
  if (fd.p)
    switch (fd.type) {
    case PIPE: 
      pclose(fd.p);
    break;
    case STDIO:
    case DISK:
      fclose(fd.p);
      break;
    case OERROR:
      break;
    }
}
