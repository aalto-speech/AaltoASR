#include "Ngram.hh"
#include "ArpaNgramReader.hh"
#include "BinNgramReader.hh"

int main(int argc, char *argv[]) {
  try {
    ArpaNgramReader reader;
    BinNgramReader writer;
    FILE *out;

    if ((out=fopen(argv[2],"w"))==NULL) {
      fprintf(stderr,"Can't write bin %s\n",argv[2]);
      exit(-1);
    }

    reader.read(argv[1]);
    writer.write(out,&reader.ngram(),false);
    fclose(out);
  }
  catch (std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
  }
}
