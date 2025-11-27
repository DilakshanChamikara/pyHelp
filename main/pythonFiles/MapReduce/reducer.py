#!/usr/bin/env python

import sys

# initialize variables
current_word = None
current_count = 0
word = None

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove trailing spaces at the end of each line
    line = line.strip()

    # parse the input we got from mapper.py and store as word and count variables
    # [FIX ME!]
    word, count = line.split("\t")

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        continue

    # this IF-switch only works because Hadoop sorts map output
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT: key <tab> value
            # [FIX ME!]
            print(f"{current_word}\t{current_count}")

        current_count = count
        current_word = word

# do not forget to output the last word if needed!
if current_word == word:
    # [FIX ME!]
    print(f"{current_word}\t{current_count}")
