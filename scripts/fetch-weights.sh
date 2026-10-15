#!/bin/sh
# Fetches the official Beat This! `final0` checkpoint (CPJKU, JKU Linz; MIT)
# from the URL its own repository uses (beat_this/inference.py at b95c8ab0)
# and checks it against the published checksum. The weights are never part
# of this repository. Usage: scripts/fetch-weights.sh [destination]
set -eu
dest="${1:-final0.ckpt}"
url="https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
want="8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
curl -sSfL -o "$dest.part" "$url"
if command -v sha256sum >/dev/null 2>&1; then
  got=$(sha256sum "$dest.part" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$dest.part" | cut -d' ' -f1)
fi
if [ "$got" != "$want" ]; then
  rm -f "$dest.part"
  echo "checksum mismatch: got $got, want $want" >&2
  exit 1
fi
mv "$dest.part" "$dest"
echo "$dest: official final0 checkpoint, sha256 $want"
