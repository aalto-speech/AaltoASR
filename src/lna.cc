#include <iomanip>
#include <vector>
#include <map>

#include <cmath>

#include "NowayHmmReader.hh"
#include "LnaReaderCircular.hh"

struct Label {
  Label() { }
  Label(const char *name, int end) : name(name), end(end) { }
  const char *name;
  int end;
};

struct State {
  bool first;
  char label;
  char state;
  float log_prob;
};

struct StateCompare {
  bool operator()(State *a, State *b) {
    return a->log_prob > b->log_prob;
  }
};

typedef std::map<int,Label> TransMap;
TransMap trans;
TransMap::iterator last = trans.end();

std::string symbols(" ·.oX@");
std::vector<State> states;
std::vector<State*> sorted_states;
LnaReaderCircular lna_reader;
int start;
int end;
int models;
float log_prob_offset = 6;

void
load_trans(const char *filename)
{
  std::ifstream in(filename);
  if (!in) {
    std::cerr << "could not open " << filename << std::endl;
    exit(1);
  }

  int start;
  int end;
  std::string label;
  label.reserve(128);
  while (in >> start >> end >> label) {
    trans[start] = Label(strdup(label.c_str()), end);
  }
}

void
load_hmms(const char *filename)
{
  std::ifstream in(filename);
  if (!in) {
    std::cerr << "could not open " << filename << std::endl;
    exit(1);
  }
  NowayHmmReader r;
  r.read(in);

  const std::vector<Hmm> &hmms = r.hmms();
  for (int h = 0; h < hmms.size(); h++) {
    const Hmm &hmm = hmms[h];

    for (int s = 2; s < hmm.states.size(); s++) {
      State state;
      state.first = (s == 2);
      state.label = hmm.label[0];
      state.state = s - 1;
      states.push_back(state);
      sorted_states.push_back(&states.back());
    }
  }
}

void
display_labels()
{
  std::cout << std::endl;
  for (int i = 0; i < models; i++) {
    if (states[i].first)
      std::cout << " ";
    std::cout << states[i].label;
  }
  std::cout << std::endl << std::endl;
}

void
display()
{
  lna_reader.seek(start);

  for (int frame = start; frame < end; frame++) {
    lna_reader.go_to(frame);
    for (int i = 0; i < models; i++)
      states[i].log_prob = lna_reader.log_prob(i);
    std::sort(sorted_states.begin(), sorted_states.end(), StateCompare());

    for (int i = 0; i < models; i++) {
      
    }


    std::cout << " " << frame;

    TransMap::iterator it = trans.find(frame);
    if (it != trans.end())
      last = it;

    if (last != trans.end()) {
      std::cout << " " << (*last).second.name;
      if ((*last).second.end == frame)
	last = trans.end();
    }
    std::cout << std::endl;
  }
}

void
display2()
{
  lna_reader.seek(start);

  int count = 0;

  for (int frame = start; frame < end; frame++) {
    if (count % 50 == 0)
      display_labels();
    count++;

    lna_reader.go_to(frame);

    for (int i = 0; i < models; i++) {
      float tmp = (lna_reader.log_prob(i) + log_prob_offset);
      tmp = tmp * symbols.length() / log_prob_offset;
      int index = (int)floor(tmp);
      if (index < 0)
	index = 0;

      if (states[i].first)
	std::cout << " ";
      std::cout << symbols[index];
    }

    std::cout << " " << frame;

    TransMap::iterator it = trans.find(frame);
    if (it != trans.end())
      last = it;

    if (last != trans.end()) {
      std::cout << " " << (*last).second.name;
      if ((*last).second.end == frame)
	last = trans.end();
    }
    std::cout << std::endl;
  }
}

int
main(int argc, char *argv[])
{
  char *hmmfile = argv[1];
  char *lnafile = argv[2];
  char *transfile = argv[3];
  start = atoi(argv[4]);
  end = atoi(argv[5]);
  models = 76;

  try {
    load_hmms(hmmfile);
    lna_reader.open(lnafile, 1024);
    load_trans(transfile);
    display2();
  }
  catch (std::exception &e) {
    std::cerr << std::endl << e.what() << std::endl;
    exit(1);
  }

  exit(0);
}
