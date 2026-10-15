#!/bin/sh
# convert.sh <clips dir> <out dir>: one NN.f32 per clip (mono f32le, 44.1 kHz).
# Clip files are named "NN - anything.ext"; NN is the id used in the reference.
set -eu
in="$1"; out="$2"; mkdir -p "$out"
for f in "$in"/*; do
  case "$f" in *.mp3|*.m4a|*.wav|*.aiff|*.flac) ;; *) continue ;; esac
  b=$(basename "$f"); n=$(printf '%s' "$b" | cut -c1-2)
  ffmpeg -nostdin -loglevel error -y -i "$f" -ac 1 -ar 44100 -f f32le "$out/$n.f32"
done
ls "$out" | wc -l
