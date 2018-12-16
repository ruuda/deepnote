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
import os
import random
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

# For a given listen, we are going to predict 6 context tracks: 3 before,
# and 3 after. We want these to be part of the "same listening session", close
# together in time. I am going to define this as "listened within 30 minutes of
# the target track". Compute how many windows that leaves us, and store the
# start indices of the windows so we can use them for training batches.
num_context = 6
windows = []
for i in range(0, len(listens) - num_context - 1):
    start = listens[i]
    center = listens[i + num_context // 2]
    end = listens[i + 1 + num_context]
    diff_start = center.timestamp - start.timestamp
    diff_end = end.timestamp - center.timestamp
    # Allow listens in a window of 30 minutes (1800 seconds).
    if diff_start < 1800 and diff_end < 1800:
        windows.append(i)

print(f'Found {len(windows)} suitable training samples of {num_context + 1} listens.')

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
num_sampled = 500

embeddings = tf.Variable(tf.truncated_normal((num_tracks, dim_embedding)))
nce_weights = tf.Variable(tf.truncated_normal((num_tracks, dim_embedding)))
nce_biases = tf.Variable(tf.zeros((num_tracks,)))

# Define input placeholders.
train_inputs = tf.placeholder(tf.int32, shape=(batch_size,))
train_labels = tf.placeholder(tf.int32, shape=(batch_size, num_context))
learning_rate = tf.placeholder(tf.float32, shape=())

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

predict_loss = tf.reduce_mean(
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

# Add a bit of L2 regularization.
regularize_loss = 0.1 * tf.reduce_mean(tf.square(embeddings))
loss = predict_loss + regularize_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Add a saver to save the embedding, in order to visualize it in Tensorboard.
saver = tf.train.Saver([embeddings])

# Fix the random seed for reproducible results. TODO: ALso fix Tensorflow's
# initializer.
random.seed(42)

num_batches = len(windows) // batch_size

def iterate_windows_batch():
    """Return (center_msids, contexts_msids) of the batch size."""
    random.shuffle(windows)
    for i in range(0, num_batches * batch_size, batch_size):
        if i + batch_size >= len(windows):
            break

        centers = []
        contexts = []

        for w in windows[i:i + batch_size]:
            context = []
            for k in range(0, num_context + 1):
                msid = listens[w + k].recording_msid
                if k == num_context // 2:
                    centers.append(msid)
                else:
                    context.append(msid)

            contexts.append(context)

        yield (centers, contexts)


# Write metadata file so Tensorboard can show track titles with the embeddings.
os.makedirs('model', exist_ok=True)
with open('model/metadata.tsv', 'w', encoding='utf-8') as f:
    for msid in id_to_track:
        track = tracks[msid]
        print(f'{track.artist} - {track.title}', file=f)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(0, 500):
        total_loss = 0.0
        rate = 0.03

        for b, (batch_centers, batch_contexts) in enumerate(iterate_windows_batch()):
            inputs = [track_to_id[msid] for msid in batch_centers]
            labels = [[track_to_id[msid] for msid in context] for context in batch_contexts]

            feed_dict = {
                train_inputs: inputs,
                train_labels: labels,
                learning_rate: rate,
            }

            _, current_loss = session.run((optimizer, loss), feed_dict=feed_dict)

            total_loss += current_loss

            if b % 100 == 99:
                mean_loss = total_loss / 100.0
                total_loss = 0.0

                # Decay the learning rate during training,
                # mostly for a faster start.
                if mean_loss < 8.0:
                    rate = min(rate, 0.001)
                if mean_loss < 10.0:
                    rate = min(rate, 0.005)
                elif mean_loss < 20.0:
                    rate = min(rate, 0.01)

                print(f'Epoch {epoch} batch {b:4}: loss = {mean_loss:0.5f}')

                save_path = saver.save(
                    session,
                    save_path='model/model.ckpt',
                    global_step=epoch * num_batches + b
                )
