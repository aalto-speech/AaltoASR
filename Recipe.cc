#include <errno.h>

#include "Recipe.hh"
#include "str.hh"

Recipe::Info::Info()
  : start_time(0), end_time(0), start_line(0), end_line(0),
    speaker_id(""), utterance_id("")
{
}

void
Recipe::clear()
{
  infos.clear();
}

void
Recipe::read(FILE *f)
{
  std::string line;
  std::vector<std::string> field;

  while (1) {
    bool ok = str::read_line(&line, f);

    // Do we have an error or eof?
    if (!ok) {
      if (ferror(f)) {
	fprintf(stderr, "Recipe: str::read_file(): read error: %s\n",
		strerror(errno));
	exit(1);
      }

      if (feof(f))
	return; // RETURN
    }

    // Parse string in fields and skip empty or commented lines
    str::clean(&line, "\n\t ");
    if (line.length() == 0 || line[0] == '#')
      continue;
    str::split(&line, " \t", true, &field);
    
    // Add info in the vector
    infos.push_back(Info());
    Info &info = infos.back();
    if (field.size() > 0)
      info.audio_path = field[0];
    if (field.size() > 1)
      info.phn_path = field[1];
    if (field.size() > 2)
      info.phn_out_path = field[2];
    if (field.size() > 3)
      info.start_time = atof(field[3].c_str());
    if (field.size() > 4)
      info.end_time = atof(field[4].c_str());
    if (field.size() > 5)
      info.start_line = atoi(field[5].c_str());
    if (field.size() > 6)
      info.end_line = atoi(field[6].c_str());
    if (field.size() > 7)
      info.speaker_id = field[7];
    if (field.size() > 8)
      info.speaker_id = field[8];
  }
}
