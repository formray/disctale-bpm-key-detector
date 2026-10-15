#!/usr/bin/env python3
"""Scores disctale-bpm-key output against a reference.

usage: score.py reference.tsv results.jsonl [-v]

reference.tsv: id, bpm, key, genre (tab-separated, '#' comments).
results.jsonl: the tool's output; a file's id is its name before the dot.

BPM: right within ±1; double or half within 2 %; other ratios (3:2, 2:3,
4:3, 3:4) and anything else are wrong. Key: exact, relative, fifth,
parallel or wrong. Rates are over the values given, abstentions apart.
The floor: BPM ≥ 94 % right, double or half ≤ 3.5 %, abstentions ≤ 25 %;
key ≥ 55 % exact.
"""
import collections
import json
import math
import re
import sys

NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def parse_key(s):
    s = s.strip().replace('♯', '#').replace('♭', 'b')
    m = re.match(r'^([A-G])([#b]?)(m?)$', s)
    if not m:
        raise ValueError(s)
    tonic = (NAMES[m[1]] + {'#': 1, 'b': -1, '': 0}[m[2]]) % 12
    return tonic, 'minor' if m[3] else 'major'


def fifths(key):
    tonic, mode = key
    major = (tonic + 3) % 12 if mode == 'minor' else tonic
    return major * 7 % 12


def classify_key(ref, got):
    if got is None:
        return 'abstained'
    r, g = parse_key(ref), parse_key(got)
    if r == g:
        return 'exact'
    if r[1] != g[1] and fifths(r) == fifths(g):
        return 'relative'
    if r[1] == g[1] and (fifths(r) - fifths(g)) % 12 in (1, 11):
        return 'fifth'
    if r[0] == g[0]:
        return 'parallel'
    return 'wrong'


def classify_bpm(ref, got):
    if got is None:
        return 'abstained'
    if abs(got - ref) <= 1:
        return 'right'
    q = got / ref
    if abs(q / 2 - 1) <= 0.02:
        return 'double'
    if abs(q * 2 - 1) <= 0.02:
        return 'half'
    if any(abs(q / x - 1) <= 0.02 for x in (1.5, 2 / 3, 4 / 3, 0.75)):
        return 'other_ratio'
    return 'wrong'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)


def main():
    ref = {}
    for line in open(sys.argv[1], encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            cols = line.rstrip('\n').split('\t')
            ref[cols[0]] = cols
    results = {}
    for line in open(sys.argv[2], encoding='utf-8'):
        j = json.loads(line)
        results[j['file'].split('.')[0]] = j
    verbose = '-v' in sys.argv

    bpm = collections.Counter()
    key = collections.Counter()
    rows = []
    for clip, (_, rbpm, rkey, *_) in sorted(ref.items()):
        j = results[clip]
        b = classify_bpm(float(rbpm), j['bpm'])
        k = classify_key(rkey, j['key'])
        bpm[b] += 1
        key[k] += 1
        rows.append((clip, rbpm, j['bpm'], b, rkey, j['key'], k))

    n = len(rows)
    given = n - bpm['abstained']
    kgiven = n - key['abstained']
    right = bpm['right'] / given * 100 if given else 0
    dh = (bpm['double'] + bpm['half']) / given * 100 if given else 0
    abst = bpm['abstained'] / n * 100
    exact = key['exact'] / kgiven * 100 if kgiven else 0
    lo, hi = wilson(bpm['right'], given)
    print(f"clips {n}")
    print(f"BPM  given {given}, right {bpm['right']} ({right:.1f} %, 95 % CI {lo:.1f}-{hi:.1f}), "
          f"double {bpm['double']}, half {bpm['half']}, other ratio {bpm['other_ratio']}, "
          f"wrong {bpm['wrong']}, abstained {bpm['abstained']} ({abst:.1f} %)")
    lo, hi = wilson(key['exact'], kgiven)
    print(f"KEY  given {kgiven}, exact {key['exact']} ({exact:.1f} %, 95 % CI {lo:.1f}-{hi:.1f}), "
          f"relative {key['relative']}, fifth {key['fifth']}, parallel {key['parallel']}, "
          f"wrong {key['wrong']}, abstained {key['abstained']}")
    bpm_ok = given and right >= 94 and dh <= 3.5 and abst <= 25
    key_ok = kgiven and exact >= 55
    print(f"floor: BPM {'met' if bpm_ok else 'NOT met'}, key {'met' if key_ok else 'NOT met'}")
    if verbose:
        for r in rows:
            if r[3] != 'right' or r[6] != 'exact':
                print('  ', *r)


if __name__ == '__main__':
    main()
