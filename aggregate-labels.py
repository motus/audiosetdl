#!/usr/bin/env python3

import pandas
import ontology

ont = ontology.Ontology("ontology.json")

labels_all = frozenset(ont.all_children(ont.top()).keys())
labels_music = frozenset(ont.children('/m/04rlf').keys())
labels_speech = frozenset(ont.children('/m/09l8g').keys())

# Make sure music and speech sets are disjoint.
# Make the ambiguous labels music only: ['Mantra', 'Choir', 'Chant']
labels_speech -= labels_music

labels_noise = labels_all - (labels_music | labels_speech)

data = pandas.read_csv("mobileNet_trainingdata_V1_labels_Wavegram_Logmel_Cnn14.csv.bz2")

index_music = []
index_speech = []
index_noise = []
for (i, label) in enumerate(data.columns):
    if label in labels_music:
        index_music.append(label)
    elif label in labels_speech:
        index_speech.append(label)
    elif label in labels_noise:
        index_noise.append(label)

res = pandas.DataFrame(columns=['filename', 'music', 'speech', 'noise'])
for i in range(len(data)):
    row = data.iloc[i]
    res.loc[i] = [
        row[0],
        row[index_music].sum(),
        row[index_speech].sum(),
        row[index_noise].sum()
    ]

res.to_csv("mobileNet_trainingdata_V1_labels_3way.csv.bz2",
           index=False, header=True, line_terminator='\n', float_format="%.5f")
