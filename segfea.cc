#include <iostream>
#include <map>
#include <string>
#include <list>
#include <math.h>

#include "str.hh"
#include "io.hh"
#include "conf.hh"
#include "Recipe.hh"
#include "FeatureGenerator.hh"
#include "SpeakerConfig.hh"

#define MAXLINE 4000

typedef struct {
  char *label;
  int num_states;
  std::vector<int> state_index;
} PhonemeInfo;

typedef struct {
  int num_features;
  float *features;
} FeatureBlock;

typedef struct {
  std::vector<std::list<FeatureBlock> > features;
  std::vector<int> counts;
  int max_buffer;
  int max_count;
  int num_states;
  int dim;
  int lines_in_buffer;
  int buffer_size;
  std::string outbase;
  bool zip;
} PrintBuffers;

conf::Config config;
PrintBuffers buffers;
Recipe recipe;
bool binary;

char *get_seg_info(char *line, int *beg, int *end)
{
  char *token;

  token = strtok(line," \t\n");
  if (token == NULL) {
    throw std::string("ERROR: cannot read beginning sample number from ") + line;
  }
  *beg = atoi(token);
  token = strtok(NULL," \t\n");
  if (token == NULL) {
    throw std::string("ERROR: cannot read end sample number from ") + line;
  }
  *end = atoi(token);
  token = strtok(NULL," \t\n");
  if (token == NULL) {
    throw std::string("ERROR: cannot read phoneme label from ") + line;
  }

  return (token);
}

void flush_buffer(int buffer_number)
{
  std::string outfile;

  if (buffers.features[buffer_number].empty())
    return;
  outfile = buffers.outbase + str::fmt(64, "_%d", buffer_number); 
  io::Stream out;

  if (buffers.zip)
    out.open(std::string("|gzip >> ") + outfile + std::string(".gz"), "w");
  else
    out.open(outfile, "a");
  
  while (!buffers.features[buffer_number].empty())
  {
    float *buf = buffers.features[buffer_number].front().features;
    int count = buffers.features[buffer_number].front().num_features;
    if (binary)
      fwrite(buf, sizeof(float), count, out);
    else
    {
      for (int i = 0; i < count; i++)
      {
        for (int d = 0; d < buffers.dim; d++)
          fprintf(out,"%f ",buf[i*buffers.dim + d]);
        fprintf(out,"\n");
      }
    }
    delete [] (buffers.features[buffer_number].front()).features;
    buffers.features[buffer_number].pop_front();
  }
  if (ferror(out))
  {
    out.close();
    throw std::string("Error when writing to file ") + std::string(outfile);
  }
  out.close();
}


void flush_max()
{
  flush_buffer(buffers.max_buffer);
  buffers.lines_in_buffer -= buffers.max_count;
  buffers.counts[buffers.max_buffer] =0 ;
  buffers.max_count = 0;
  // Refresh max_count and max_buffer
  for (int i = 0; i < buffers.num_states; i++)
  {
    if (buffers.counts[i] > buffers.max_count)
    {
      buffers.max_count = buffers.counts[i];
      buffers.max_buffer = i;
    }
  }
}

void add_features_to_buffer(FeatureBlock block, int buffer_number) 
{
  if (buffers.lines_in_buffer + block.num_features > buffers.buffer_size)
    flush_max();
  
  buffers.features[buffer_number].push_back(block);
  buffers.counts[buffer_number] += block.num_features;
  if (buffers.counts[buffer_number] > buffers.max_count)
  {
    buffers.max_count = buffers.counts[buffer_number];
    buffers.max_buffer = buffer_number;
  }
  buffers.lines_in_buffer += block.num_features;
}


void init_print_buffers(std::map<std::string,PhonemeInfo> &pho_info,
                        const std::string &outbase, bool zip,
                        int buffer_size, int num_states,
                        FeatureGenerator &gen) 
{
  buffers.dim = gen.dim();
  buffers.buffer_size = buffer_size;
  buffers.outbase = outbase;
  buffers.num_states = num_states;
  buffers.features.resize(buffers.num_states);
  buffers.counts.resize(buffers.num_states);
  for (int i = 0; i < buffers.num_states; i++)
    buffers.counts[i] = 0;
  buffers.lines_in_buffer = 0;
  buffers.max_count = 0;
  buffers.zip = zip;
}


void load_phoneme_state_info(const std::string &filename,
                             std::map<std::string,PhonemeInfo> &pho_info)
{
  FILE *fp;
  char line[MAXLINE], *token;
  PhonemeInfo item;
  bool error_flag = false;

  if ((fp = fopen(filename.c_str(),"r")) == NULL)
  {
    throw std::string("Can not open file ") + filename +
      std::string(" for reading.");
  }

  while (fgets(line,MAXLINE,fp)) {
    if ((token = strtok(line," \t\n")) != NULL) {
      std::string tmp = token;
      item.label = strdup(token);
      if ((token = strtok(NULL," \t\n")) != NULL) {
        item.num_states = atoi(token);
        if (item.num_states > 0)
        {
          item.state_index.clear();
          for (int i = 0; i < item.num_states; i++) {
            if ((token = strtok(NULL," \t\n")) == NULL) {
              error_flag = true;
              break;
            }
            item.state_index.push_back(atoi(token));
          }
          if (error_flag)
            break;
        }
        else
        {
          error_flag = true; break;
        }
      }
      else
      {
        error_flag = true; break;
      }
      pho_info[tmp] = item;
    }
  }
  fclose(fp);
  if (error_flag)
  {
    throw std::string("Invalid file format in state binding file ") + filename;
  }
}



int 
compute_features_segfea(const std::string &in_fname,
                        const std::string &phn_fname, 
		        int start_frame, int end_frame,
			std::map<std::string, PhonemeInfo> &pho_info,
			bool state_segmentation, bool raw,
                        std::vector<long int> &occurrences,
                        int info, FeatureGenerator &gen)
{
  FILE *fp;
  char line[MAXLINE], *token;
  int s_beg, s_end, beg, end, dur, linecount;
  int p,pnum;
  int state = 0; // Suppress warning
  char *label_token;
  char *state_token;
  int state_index;
  int offset;

  gen.open(in_fname, raw);
  
  if ((fp = fopen(phn_fname.c_str(),"r")) == NULL)
  {
    throw std::string("Can not open file ") + phn_fname +
      std::string(" for reading.");
  }

  linecount = 1;
  while (fgets(line,MAXLINE,fp))
  {
    token = get_seg_info(line, &s_beg, &s_end);
    s_beg = (int)((double)s_beg/gen.sample_rate()*gen.frame_rate());
    s_end = (int)((double)s_end/gen.sample_rate()*gen.frame_rate());

    if (s_end < start_frame)
      continue;
    else if (s_beg < start_frame)
      s_beg = start_frame;

    if (end_frame > 0 && s_end > end_frame)
      s_end = end_frame;

    if (s_beg >= s_end) // Empty
      continue;

    // PHN labels may include several models, iterate them through
    label_token = strtok(token,",");
    while (label_token != NULL)
    {
      if (state_segmentation) {
        // The state index comes after a dot
        label_token = strtok(label_token,".");
        // This might cause problems if using state segmentation and
        // there are several models per PHN label?
        state_token = strtok(NULL,"");
        state = atoi(state_token);
      }
	
      std::map<std::string,PhonemeInfo>::const_iterator it =
        pho_info.find(label_token);
      if (it == pho_info.end())
      {
        fclose(fp);
        throw str::fmt(1024,"ERROR: Unrecognized phoneme '%s' in token '%s' in file at line %i",
                       label_token, token, phn_fname.c_str(), linecount);
      }
      
      // A little trick so the same loop works also with state segmentation
      if (state_segmentation)
        pnum = 1;
      else
        pnum=(*it).second.num_states;
	
      dur = (s_end-s_beg);
      if (info > 1)
        printf("%s, %d frames, (%d-%d)\n",label_token,dur,s_beg,s_end);
	
      for (p=0; p<pnum; p++)
      {
        if (state_segmentation)
        {
          // In this case we got the state info already from the .phn
          state_index = (*it).second.state_index[state];
        }
        else
        {
          state_index = (*it).second.state_index[p];
        }
	  
        // If no accurate state segmentation available, 
        // divide the phoneme segment evenly into states
        if (state_segmentation) {
          beg = s_beg;
          end = s_end;
        }
        else {
          beg = (s_beg + p*dur/pnum);
          end = s_beg + ((p+1)*dur)/pnum;
        }   
        
        // ocmpute features in this data segment beg...end
        if (info > 2)
          printf("  part %d/%d: frames %d-%d\n",p+1,pnum,beg,end);

        if (beg < end)
        {
          FeatureBlock feablock;
          feablock.num_features = end - beg;
          // NOTE: *features will be deleted when flushing the buffer
          feablock.features = new float[buffers.dim*feablock.num_features];

          occurrences[state_index]++;
        
          for (int f = beg; f < end; f++)
          {
            const FeatureVec feavec = gen.generate(f);
            if (gen.eof())
            {
              if (end_frame > 0)
              {
                fprintf(stderr,"Going past eof in file %s at frame %d. Skipping to next file.\n",
                        in_fname.c_str(), f);
              }
              feablock.num_features = f - beg;
              add_features_to_buffer(feablock, state_index);
              goto CLOSE_FILES_AND_EXIT;
            }
            offset = (f-beg)*buffers.dim;
            for (int d = 0; d < buffers.dim; d++)
              feablock.features[offset+d] = feavec[d];
          }
          add_features_to_buffer(feablock, state_index);
        }
      }
      label_token = strtok(NULL,",");
    }
    linecount++;
  }
 CLOSE_FILES_AND_EXIT:    
  fclose(fp);
  gen.close();
  return(0);
}

int
main(int argc, char *argv[])
{
  FeatureGenerator gen;
  SpeakerConfig speaker_config(gen); // Speaker configuration handler
  int info;
  int num_states;
  std::vector <long int> occurrences;
  std::map<std::string, PhonemeInfo> pho_info;
  int start_frame, end_frame;
  
  try {
    config("usage: segfea [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "bind=FILE", "arg must", "", "model state configuration and bindings")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('o', "out=FILE", "arg must", "", "base filename for features")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "occ=FILE", "arg", "", "save state occurrence information to file")
      ('z', "zip", "", "", "zip the feature files")
      ('s', "stateseg", "", "", "the segmentation is based on states")
      ('\0', "binary", "", "", "write feature files as binary floats")
      ('\0', "bufsize=INT", "arg", "2000000", "buffer size, default 2000000")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    binary = config["binary"].specified;
    gen.load_configuration(io::Stream(config["config"].get_str()));

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()));

    // Load state configuration
    load_phoneme_state_info(config["bind"].get_str(), pho_info);

    // Speaker configuration
    if (config["speakers"].specified)
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));

    // Count the number of states for allocating the buffers
    num_states = 0;
    for (std::map<std::string,PhonemeInfo>::iterator it=pho_info.begin();
         it!=pho_info.end(); it++)
    {
      num_states += (*it).second.num_states;
    }
    occurrences.resize(num_states);
    init_print_buffers(pho_info, config["out"].get_str().c_str(),
                       config["zip"].specified, config["bufsize"].get_int(),
                       num_states, gen);

    for (int fi = 0; fi < (int)recipe.infos.size(); fi++)
    {
      if (info > 0)
        printf("file %d/%d '%s' '%s'\n", fi+1, (int)recipe.infos.size(),
               recipe.infos[fi].audio_path.c_str(), 
               recipe.infos[fi].phn_path.c_str());

      if (config["speakers"].specified)
        speaker_config.set_speaker(recipe.infos[fi].speaker_id);

      start_frame = (int)(recipe.infos[fi].start_time * gen.frame_rate());
      end_frame = (int)(recipe.infos[fi].end_time * gen.frame_rate());
      compute_features_segfea(recipe.infos[fi].audio_path,
                              recipe.infos[fi].phn_path,
                              start_frame, end_frame,
                              pho_info, config["stateseg"].specified,
                              config["raw-input"].specified,
                              occurrences, info, gen);
    }

    for (int i=0; i < buffers.num_states; i++)
      flush_buffer(i);

    // Print also the state occurrence counts into file (outfile.occ)
    if (config["occ"].specified)
    {
      FILE *occfile;
      
      if ((occfile = fopen(config["occ"].get_str().c_str(),"w")) == NULL)
      {
        throw std::string("error opening occurrence file ") +
          config["occ"].get_str();
      }

      for (int i = 0; i < (int)occurrences.size(); i++)
        fprintf(occfile,"%i %i\n", i, occurrences[i]);
      
      fclose(occfile);
    }
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }

  return (0);
}
