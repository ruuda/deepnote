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
import tensorflow as tf
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
    timestamp: int
    recording_msid: str
    release_msid: str
    artist_msid: str
    additional_info: dict


class Track(typing.NamedTuple):
    title: str
    artist: str


listens = []
listencounts = collections.defaultdict(lambda: 0)

print('Parsing listens ...', end='', flush=True)
for raw_listen in raw_listens:
    listen = Listen(**raw_listen)
    listencounts[listen.recording_msid] += 1
    listens.append(listen)

raw_listens = None
print(f' done.')
print(f'Listened to {len(listencounts)} distinct tracks.')

# Remove tracks with few listens. We need at least *some* data
# to get a decent embedding.
min_listens = 5
listens = [
    listen
    for listen in listens
    if listencounts[listen.recording_msid] >= min_listens
]

tracks = {
    listen.recording_msid: Track(listen.track_name, listen.artist_name)
    for listen in listens
}

artists = {
    listen.artist_msid: listen.artist_name
    for listen in listens
}

print(f'Listened to {len(tracks)} distinct tracks at least {min_listens} times.')
print(f'{len(artists)} distinct artists in remaining listens.')

# Data from Listenbrainz appears to be ordered by timestamp already, but it
# might be submission order too -- I should verify. In the mean time, sort by
# timestamp to be safe.
listens.sort(key=lambda listen: listen.timestamp)

# Assign every track an index in the range 0..len(tracks) for one-hot encoding.
track_to_id = {}
id_to_track = []

# Assign ids by reverse frequency, so the more frequently listened tracks get
# lower ids. This is required for tf.nn.nce_loss.
for msid in sorted(tracks.keys(), key=lambda x: -listencounts[x]):
    track_to_id[msid] = len(id_to_track)
    id_to_track.append(msid)

# Set up Tensorflow variables for the embedding. For the embedding dimension, in
# my case I have roughly 7k distinct tracks and about 900 artists that occur in
# those, so if we used dimension 7k, every track would get its own embedding,
# and there would be nothing to learn. On the other hand, with log2(7000)
# dimensions (about 12.7) we would need a very dense packing of information to
# be able to identify every track uniquely. Perhaps their geometric mean, 265,
# is a good place to start? It is a bit less than the number of artists, so we
# do not get a dimension per artist either -- the model will need to eliminte
# redundancies to fit. Let's try 265.
dim_embedding = 265

batch_size = 64
num_tracks = len(id_to_track)

# Number of negative samples per training batch. Should be at most the number of
# tracks.
num_sampled = 250

embeddings = tf.Variable(tf.truncated_normal((num_tracks, dim_embedding)))
nce_weights = tf.Variable(tf.truncated_normal((num_tracks, dim_embedding)))
nce_biases = tf.Variable(tf.zeros((num_tracks,)))

# From the track, we are going to predict 6 context tracks: 3 before,
# and 3 after.
num_context = 6

# Define input placeholders.
train_inputs = tf.placeholder(tf.int32, shape=(batch_size,))
train_labels = tf.placeholder(tf.int32, shape=(batch_size, num_context))

embed = tf.nn.embedding_lookup(embeddings, train_inputs)
loss = tf.reduce_mean(
    tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=train_labels,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=num_tracks,
        num_true=6,
    )
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
