#!/usr/bin/env python3

# Deepnote -- Music embedding generator
# Copyright 2018 Ruud van Asseldonk

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3. A copy
# of the License is available in the root of the repository.

"""
Deepnote -- Music embedding generator

OVERVIEW

Take a Listenbrainz [1] listening history, and generate an embedding vector for
every track listened often enough, based on the tracks listened before and
after. The idea is twofold:

 * On the long term, there are popularity changes and changes in taste, so
   listens of similar music, or at least music that goes well together, tends to
   be clustered over time (on a time scale of weeks to months).

 * On the short term, it is uncommon to listen to very different genres in one
   session. Sometimes I am in the mood for rock music, sometimes for electronic.
   But I rarely put both rock and electronic in the same play queue. On the
   other hand, I do often listen to entire albums, or a few albums that go well
   together. So if two tracks are played in a short time window of one another,
   they are probably similar.

SEE ALSO

[1]: https://listenbrainz.org

USAGE

    ./deepnote.py listenbrainz.json

"""

import collections
import json
import sys
import typing

if len(sys.argv) != 2:
    print(__doc__.strip())
    sys.exit(1)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    print('Loading listens ...', end='', flush=True)
    raw_listens = json.load(f)
    print(f' done, loaded {len(raw_listens)} listens.')


class Listen(typing.NamedTuple):
    track_name: str
    artist_name: str
    timestamp: str
    recording_msid: str
    release_msid: str
    artist_msid: str
    additional_info: dict


class Track(typing.NamedTuple):
    title: str
    artist: str


listens = []
listencounts = collections.defaultdict(lambda: 0)
tracks = {}

print('Parsing listens ...', end='', flush=True)
for raw_listen in raw_listens:
    listen = Listen(**raw_listen)
    listencounts[listen.recording_msid] += 1
    tracks[listen.recording_msid] = Track(listen.track_name, listen.artist_name)
    listens.append(listen)

raw_listens = None
print(f' done.')

for msid, lcount in sorted(listencounts.items(), key=lambda x: -x[1]):
    track = tracks[msid]
    print(f'{lcount:3} {track.artist} - {track.title}')
