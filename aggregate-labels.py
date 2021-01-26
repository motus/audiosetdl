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

# make sure we use only the available labels
labels_available = frozenset(data.columns)
labels_music &= labels_available
labels_speech &= labels_available
labels_noise &= labels_available

res = pandas.DataFrame({
    "filename": data.filename,
    "music": data[labels_music].sum(axis=1),
    "speech": data[labels_speech].sum(axis=1),
    "noise": data[labels_noise].sum(axis=1)
})

res.to_csv("mobileNet_trainingdata_V1_labels_3way.csv.bz2",
           index=False, header=True, line_terminator='\n', float_format="%.5f")
