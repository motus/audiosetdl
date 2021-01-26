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

data = pandas.read_csv(
    "mobileNet_trainingdata_V1_labels_Wavegram_Logmel_Cnn14.csv.bz2")

# Use only the labels that exist in the data
labels_available = frozenset(data.columns)
labels_music &= labels_available
labels_speech &= labels_available
labels_noise &= labels_available

res = pandas.DataFrame({
    "filename": data.filename,
    "music": data[labels_music].max(axis=1),
    "speech": data[labels_speech].max(axis=1),
    "noise": data[labels_noise].max(axis=1)
})

res.to_csv("mobileNet_trainingdata_V1_labels_3way.csv.bz2",
           index=False, header=True, line_terminator='\n',
           float_format="%.5f")

# Produce a CSV file in audiosetdl format
res['filename'] = res.filename.apply(lambda s: s[:-4])  # Remove .wav
res['labels'] = ""
res['start'] = 0.0
res['end'] = 10.0

_LABEL_THRESHOLD = 0.6

# Top-level music category
res.loc[res.music >= _LABEL_THRESHOLD, 'labels'] = \
    res.loc[res.music >= _LABEL_THRESHOLD, 'labels'].apply('/m/04rlf,'.__add__)

# Top-level speech category
res.loc[res.speech >= _LABEL_THRESHOLD, 'labels'] = \
    res.loc[res.speech >= _LABEL_THRESHOLD, 'labels'].apply('/m/09l8g,'.__add__)

# Top-level category: Sounds of things
res.loc[res.noise >= _LABEL_THRESHOLD, 'labels'] = \
    res.loc[res.noise >= _LABEL_THRESHOLD, 'labels'].apply('/t/dd00041,'.__add__)

res['labels'] = res.labels.apply(lambda s: s.strip(','))

res[['filename', 'start', 'end', 'labels']].to_csv(
    "mobileNet_trainingdata_V1_labels_audiosetdl.csv.bz2",
    index=False, header=False, line_terminator='\n',
    float_format="%.1f")
