#include <string>
#include <numeric>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "Recipe.hh"

conf::Config config;
Recipe recipe;
HmmSet model;
int info;
int max_dur;

std::vector< std::vector<int> >  dur_table;

void
add_duration(std::string &label, int state, int num_frames, int skip_states)
{
  Hmm &hmm = model.hmm(label);
  int state_index = hmm.state(state);
  if (state_index >= skip_states)
  {
    if (num_frames > max_dur)
      num_frames = max_dur;
    assert( num_frames > 0 );
    dur_table[state_index][num_frames-1]++;
  }
}

void
collect_dur_stats(PhnReader *phn_reader, int skip_states)
{
  PhnReader::Phn phn;

  while (phn_reader->next_phn_line(phn))
  {
    if (phn.state == -1)
      throw std::string("Collecting duration statistics requires phn files with state numbers!");

    add_duration(phn.label[0], phn.state, phn.end - phn.start, skip_states);
  }
}

void
write_dur_histograms(const std::string &filename, int skip_states)
{
  io::Stream out_file(filename, "w");
  for (int i = skip_states; i < (int)dur_table.size(); i++)
  {
    fprintf(out_file, "%d ", i);
    for (int j = 0; j < (int)dur_table[i].size(); j++)
      fprintf(out_file, "%d ", dur_table[i][j]);
    fprintf(out_file, "\n");
  }
}


double
negative_gamma_loglikelihood(double a, double mean_log, double log_mean)
{
   return a*(1+log_mean-log(a))+lgamma(a)+(1-a)*mean_log;
}


bool
estimate_gamma_models(std::vector<int> &hist, double &a_out, double &b_out)
{
  std::vector<double> sequence;
  double count, mean, var;
  double mean_log, log_mean;
  count = std::accumulate(hist.begin(), hist.end(), 0);
  if (count < 2)
  {
    if (info > 0)
      fprintf(stderr, "Warning: Needs at least 2 points for estimating gamma models\n");
    return false;
  }
  mean = 0;
  for (int i = 0; i < (int)hist.size(); i++)
    mean += (i+1)*hist[i];
  mean /= (double)count;
  var = 0;
  for (int i = 0; i < (int)hist.size(); i++)
    var += (i+1-mean)*(i+1-mean)*hist[i];
  var = std::max(var/((double)count-1), 0.25);

  log_mean = log(mean);
  mean_log = 0;
  for (int i = 0; i < (int)hist.size(); i++)
    mean_log += log(i+1)*hist[i];
  mean_log /= (double)count;

  // Find the maximum likelihood solution using the golden ratio method
  double x1, x2, x1v, x2v;
  double a, b, r;
  r = (sqrt(5)-1)/2;
  a = 1; // Lower limit for parameter A
  b = 2*std::max(mean*mean/var, 1.5) - 1;
  x1 = a+(1-r)*(b-a);
  x2 = a+r*(b-1);
  x1v = negative_gamma_loglikelihood(x1, mean_log, log_mean);
  x2v = negative_gamma_loglikelihood(x2, mean_log, log_mean);

  while (b-a > 0.01)
  {
    if (x2v > x1v)
    {
      b = x2;
      x2 = x1;
      x2v = x1v;
      x1 = a+(1-r)*(b-a);
      x1v = negative_gamma_loglikelihood(x1, mean_log, log_mean);
    }
    else
    {
      a = x1;
      x1 = x2;
      x1v = x2v;
      x2 = b-(1-r)*(b-a);
      x2v = negative_gamma_loglikelihood(x2, mean_log, log_mean);
    }
  }
  a_out = (a + b)/2;
  b_out = mean / a_out;
  
  return true;
}

void
write_gamma_models(const std::string &filename, int skip_states, int min_count)
{
  io::Stream out_file(filename, "w");
  fprintf(out_file, "4\n%d\n", model.num_states());
  for (int i = 0; i < (int)dur_table.size(); i++)
  {
    double a, b;
    if (i < skip_states ||
        std::accumulate(dur_table[i].begin(),dur_table[i].end(),0) < min_count
        || !estimate_gamma_models(dur_table[i], a, b))
    {
      if (info > 0)
        fprintf(stderr, "Warning: No duration model for state %i\n", i);
      a = b = 0;
    }
    fprintf(out_file, "%d %.4f %.4f\n", i, a, b);
  }
}


int
main(int argc, char *argv[])
{
  PhnReader phn_reader(NULL);
  int skip_states;
  try {
    config("usage: dur_est [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('p', "ph=FILE", "arg must", "", "HMM definitions")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for VTLN")
      ('M', "maxdur=INT", "arg", "100", "maximum duration noted")
      ('\0', "hist=FILE", "arg", "", "write duration histograms to file")
      ('\0', "gamma=FILE", "arg", "", "write gamma models for durations to file")
      ('\0', "skip=INT", "arg", "0", "skip first INT states")
      ('\0', "mincount=INT", "arg", "10", "minimum occurance count to estimate gamma models")
      ('i', "info=INT", "arg", "0", "info level")
      ;

    config.default_parse(argc, argv);
    
    info = config["info"].get_int();

    // Load the model
    model.read_ph(config["ph"].get_str());

    skip_states = config["skip"].get_int();

    // Initialize the duration table
    max_dur = config["maxdur"].get_int();
    dur_table.resize(model.num_states());
    for (int i = 0; i < (int)dur_table.size(); i++)
      dur_table[i].resize(max_dur);

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()), 0, 0, false);
        
    for (int f = 0; f < (int)recipe.infos.size(); f++)
    {
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s\n", 
                recipe.infos[f].audio_path.c_str());
      }
       
      recipe.infos[f].init_phn_files(NULL, false, false,
                                     config["ophn"].specified,
                                     NULL, false,
                                     &phn_reader);
      phn_reader.set_collect_transition_probs(false);
      if (!phn_reader.init_utterance_segmentation())
      {
        fprintf(stderr, "Could not initialize the utterance for PhnReader.");
        fprintf(stderr,"Current file was: %s\n",
                recipe.infos[f].transcript_path.c_str());
      }
      else
      {
        collect_dur_stats(&phn_reader, skip_states);
      }

      phn_reader.close();
    }

    if (config["hist"].specified)
      write_dur_histograms(config["hist"].get_str(), skip_states);
    if (config["gamma"].specified)
      write_gamma_models(config["gamma"].get_str(), skip_states,
                         config["mincount"].get_int());
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
}
