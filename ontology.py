#!/usr/bin/env python3

import json
import pandas


class Ontology:

    def __init__(self, fname):
        self._nodes = {node['id']: node for node in json.load(open(fname))}

    def get(self, node_id, default=None):
        return self._nodes.get(node_id, default)

    def names(self, ids):
        return [self._nodes.get(n, {}).get('name', n) for n in ids]

    def any(self, ids):
        return self._nodes

    def children(self, node_id, children_dict=None):
        if children_dict is None:
            children_dict = {}
        node = self._nodes.get(node_id)
        if node:
            children_dict[node_id] = node.get('name')
            for n in node.get('child_ids', []):
                self.children(n, children_dict)
        return children_dict


def main(fname, skip_cat='/m/0dgw9r'):
    ontology = Ontology("ontology.json")
    skipset = frozenset(ontology.children(skip_cat).keys())

    data = pandas.read_csv(
        fname, skiprows=3, skipinitialspace=True, header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"])

    data = data[data['positive_labels'].apply(lambda s: skipset.isdisjoint(s.split(',')))]
    return data


if __name__ == "__main__":
    data = main("unbalanced_train_segments.csv", skip_cat='/m/0dgw9r')  # all human sounds
    data['YTID'].to_csv(
        "unbalanced_ytids_no_human.txt",
        index=False, header=False, line_terminator='\n')
