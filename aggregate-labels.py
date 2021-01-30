#!/usr/bin/env python3

import argparse

import pandas
import ontology


def aggregate_labels(fname_input, fname_3way, fname_audioset,
                     threshold=0.6):
    """
    Read a CSV file produced by PANN inference and aggregate
    the data within speech, music, and noise categories.
    Save the results in two CSV files: one with max probabilities
    in each of three categories, and another one with three tags
    in audiosetdl format.
    """

    ont = ontology.Ontology("ontology.json")

    labels_all = frozenset(ont.all_children(ont.top()).keys())
    labels_music = frozenset(ont.children('/m/04rlf').keys())
    labels_speech = frozenset(ont.children('/m/09l8g').keys())

    # Make sure music and speech sets are disjoint.
    # Make the ambiguous labels music only: ['Mantra', 'Choir', 'Chant']
    labels_speech -= labels_music

    labels_noise = labels_all - (labels_music | labels_speech)

    data = pandas.read_csv(fname_input)

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

    res.to_csv(fname_3way, index=False, header=True,
               line_terminator='\n', float_format="%.5f")

    # Produce a CSV file in audiosetdl format
    res['filename'] = res.filename.apply(
        lambda s: s[:-4] if s.endswith(".wav") else s)  # Remove .wav
    res['labels'] = ""
    res['start'] = 0.0
    res['end'] = 10.0

    # Top-level music category
    res.loc[res.music >= threshold, 'labels'] = \
        res.loc[res.music >= threshold, 'labels'].apply('/m/04rlf,'.__add__)

    # Top-level speech category
    res.loc[res.speech >= threshold, 'labels'] = \
        res.loc[res.speech >= threshold, 'labels'].apply('/m/09l8g,'.__add__)

    # Top-level category: Sounds of things
    res.loc[res.noise >= threshold, 'labels'] = \
        res.loc[res.noise >= threshold, 'labels'].apply('/t/dd00041,'.__add__)

    res['labels'] = res.labels.apply(lambda s: s.strip(','))

    res[['filename', 'start', 'end', 'labels']].to_csv(
        fname_audioset, index=False, header=False,
        line_terminator='\n', float_format="%.1f")


def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="Input CSV file as produced by PANN inference.py."
             " Can be .csv.bz2")
    parser.add_argument(
        "three_way",
        help="Output CSV file for the aggregated data. Can be .bz2")
    parser.add_argument(
        "audioset",
        help="Output CSV file for AudioSet-compatible data."
             " Can be .bz2")
    parser.add_argument(
        "--threshold", type=int, default=0.6,
        help="Inclusion threshold for the AudioSet-compatible labels."
             " Default = 0.6")

    args = parser.parse_args()

    aggregate_labels(args.input, args.three_way, args.audioset,
                     args.threshold)


if __name__ == "__main__":
    _main()
