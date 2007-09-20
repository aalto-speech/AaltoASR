#!/usr/bin/perl
# Usage: copy_recipe_files.pl RECIPE source-key target-key

use locale;
use strict;

my $recipe_file = shift;
my $source_key = shift;
my $target_key = shift;

die "Usage: copy_recipe_files.pl RECIPE source-key target-key" if (!defined($target_key));

open REC, "< $recipe_file" || die "Could not open $recipe_file\n";

my $cur_line = 1;
my $source_file;
my $target_file;

while (<REC>) {
  chomp;
  if (length > 0) {
    if (!/$source_key=(\S*)/) {
      die "recipe key ".$source_key." not found at line $cur_line\n";
    }
    $source_file = $1;
    if (!/$target_key=(\S*)/) {
      die "recipe key ".$target_key." not found at line $cur_line\n";
    }
    $target_file = $1;
    system("cp ".$source_file." ".$target_file);
  }
  $cur_line++;
}

close REC;
