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

import matplotlib
matplotlib.use('gtk3cairo')

import collections
import json
import matplotlib.pyplot as plt
import numpy as np
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
min_listens = 3
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
for msid in sorted(tracks.keys(), key=lambda x: -min(listencounts[x], 20)):
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
    end = listens[i + num_context]
    diff_start = center.timestamp - start.timestamp
    diff_end = end.timestamp - center.timestamp
    # Allow listens in a window of 30 minutes (1800 seconds).
    if diff_start < 1800 and diff_end < 1800:
        assert end.timestamp - start.timestamp < 3600
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
#dim_embedding = 265
dim_embedding = 16

batch_size = 75
num_tracks = len(id_to_track)

embeddings = tf.Variable(0.1 * tf.truncated_normal((num_tracks, dim_embedding)))
weights = tf.Variable(0.1 * tf.truncated_normal((num_context, num_tracks, dim_embedding)))

# Pre-set biases to track frequencies to get a warmer start.
frequencies = np.array([listencounts[i] for i in id_to_track], dtype=np.float32)
initial_biases = np.log(frequencies / np.sum(frequencies))
biases = tf.Variable(tf.stack([initial_biases] * num_context), dtype=tf.float32)

# Define input placeholders.
train_inputs = tf.placeholder(tf.int32, shape=(batch_size,))
train_labels = tf.placeholder(tf.int32, shape=(batch_size, num_context))
learning_rate = tf.placeholder(tf.float32, shape=())

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# The loss is a weighted loss over all context tracks, where tracks closer to
# the focus track weigh more.
context_weight = np.array([0.15, 0.15, 0.2, 0.2, 0.15, 0.15])
assert context_weight.shape == (num_context,)
assert np.sum(context_weight) == 1.0

predict_loss = 0.0
for i in range(0, num_context):
    bias = biases[i]
    logits = tf.matmul(embed, tf.transpose(weights[i])) + bias
    assert logits.shape == (batch_size, num_tracks)
    predict_loss = predict_loss + tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels[:, i],
        logits=logits,
    ) * context_weight[i]

# Add a bit of L2 regularization.
regularize_loss = (
    0.01 * tf.reduce_mean(tf.square(embeddings)) +
    0.01 * tf.reduce_mean(tf.square(biases)) +
    0.01 * tf.reduce_mean(tf.square(weights))
)
loss = tf.reduce_mean(predict_loss) + regularize_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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
            times = []
            context = []
            for k in range(0, num_context + 1):
                msid = listens[w + k].recording_msid
                if k == num_context // 2:
                    centers.append(msid)
                else:
                    context.append(msid)
                times.append(listens[w + k].timestamp)

            assert len(times) == num_context + 1
            assert max(times) - min(times) < 3600, f'Invalid window {times}'

            contexts.append(context)

        assert len(centers) == batch_size
        assert len(contexts) == batch_size
        assert all(len(ctx) == num_context for ctx in contexts)

        yield (centers, contexts)


# Write metadata file so Tensorboard can show track titles with the embeddings.
os.makedirs('model', exist_ok=True)
with open('model/metadata.tsv', 'w', encoding='utf-8') as f:
    for msid in id_to_track:
        track = tracks[msid]
        print(f'{track.artist} - {track.title}', file=f)


plt.ion()
fig = plt.figure()
num_plot = 45
ax = fig.add_subplot(111)
tracknames = [tracks[msid].artist for msid in id_to_track[:num_plot]]
scatter, = ax.plot(
    np.zeros(num_plot),
    np.zeros(num_plot),
    marker='o',
    markersize=3,
    linestyle='',
)

annotations = [
    plt.annotate(
        f'{tracks[msid].artist}\n{tracks[msid].title}',
        (0.0, 0.0),
        fontsize=10.0,
    )
    for msid in id_to_track[:num_plot]
]

plt.show()

projection = np.zeros((dim_embedding, 2))
projection[0, 0] = 1.0
projection[1, 1] = 1.0

def update_projection(emb: np.array):
    global projection

    # Extract the first two principal components of the embeddings of the first
    # num_plot tracks.
    u, magnitudes, _v = np.linalg.svd(emb.T)
    assert u.shape == (dim_embedding, dim_embedding)
    assert magnitudes.shape == (dim_embedding,)

    # Project embeddings on the space spanned by the first two principal
    # components.
    projection = u[:2].T
    print(
        'Projection quality:',
        magnitudes[:3] / np.sum(magnitudes),
        magnitudes[-3:] / np.sum(magnitudes)
    )


def plot_embedding(emb: np.array):
    points = np.matmul(emb[:num_plot], projection)

    # Scatter plot them.
    scatter.set_xdata(points[:, 0])
    scatter.set_ydata(points[:, 1])
    # TODO: Check that the projection makes sense and use it.
    # For now, just taking the first two coordinates is more stable.
    #scatter.set_xdata(emb[:num_plot, 0])
    #scatter.set_ydata(emb[:num_plot, 1])

    ax.set_xlim((np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1))
    ax.set_ylim((np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1))

    for i, ann in enumerate(annotations):
        ann.set_x(points[i, 0])
        ann.set_y(points[i, 1])

    plt.pause(0.0001)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    rate = 0.01
    total_loss = 0.0
    n_loss = 0

    for epoch in range(0, 500):

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
            n_loss += 1

            # Keep the window responsive.
            plt.pause(0.0001)

            if n_loss == 200:
                mean_loss = total_loss / n_loss
                total_loss = 0.0
                n_loss = 0

                # Decay the learning rate during training,
                # mostly for a faster start.
                if mean_loss < 0.5:
                    rate = min(rate, 0.0005)
                if mean_loss < 3.0:
                    rate = min(rate, 0.001)
                if mean_loss < 6.0:
                    rate = min(rate, 0.005)
                elif mean_loss < 10.0:
                    rate = min(rate, 0.009)

                print(f'Epoch {epoch} batch {b:4}: loss = {mean_loss:0.5f}')

                emb, = session.run((embeddings,))
                plot_embedding(emb)

        # Update the projection after every epoch. It is expensive, so don't do
        # it every time.
        update_projection(emb)
