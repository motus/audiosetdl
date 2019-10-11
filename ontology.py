#!/usr/bin/env python3

import json
import pandas


class Ontology:

    def __init__(self, fname):
        self._nodes = {node['id']: node for node in json.load(open(fname))}
        self._nodes_by_name = {node['name']: node for node in self._nodes.values()}

    def get(self, node_id, default=None):
        return self._nodes.get(node_id, default)

    def get_by_name(self, name, default=None):
        return self._nodes_by_name.get(name, default)

    def names(self, ids):
        return [self._nodes.get(n, {}).get('name', n) for n in ids]

    def graph(self, fname, subgraph=None):
        with open(fname, "wt") as f:
            f.write("digraph G {\n")
            all_nodes = self._nodes.values()
            if subgraph:
                all_nodes = [node for node in self._nodes.values() if node['id'] in subgraph]
            for node in all_nodes:
                f.write('  "%s" [label="%s"];\n' % (node['id'], node['name']))
                children = node.get("child_ids")
                if children:
                    f.write('  "%s" -> { "%s" };\n' % (node['id'], '","'.join(children)))
            f.write("}\n")

    def children(self, node_id, children_dict=None):
        if children_dict is None:
            children_dict = {}
        node = self._nodes.get(node_id)
        if node:
            children_dict[node_id] = node.get('name')
            for n in node.get('child_ids', []):
                self.children(n, children_dict)
        return children_dict


def read_categories(fname, ontology):
    data = pandas.read_csv(fname, sep="\t", header=None, names=["name", "quality", "num"])
    data.quality = data.quality.apply(lambda s: float('0' + s[:-1]) / 100.)
    data.num = data.num.apply(lambda s: int(s.replace(',', '')))
    data['label'] = data.name.apply(lambda n: ontology.get_by_name(n, {}).get('id'))
    return data


def main(fname, skip_cat=None):

    ontology = Ontology("ontology.json")
    skipset = frozenset()
    if skip_cat is not None:
        skipset = frozenset(ontology.children(skip_cat).keys())

    data = pandas.read_csv(
        fname, skiprows=3, skipinitialspace=True, header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"])

    accuracy = read_categories("accuracy.tsv", ontology)
    accuracy = accuracy[accuracy.quality >= 0.9]

    include = frozenset(accuracy.label)
    exclude = frozenset(read_categories("exclude.tsv", ontology).label)

    select = include.difference(skipset.union(exclude))

    data.positive_labels = data.positive_labels.apply(lambda s: s.split(','))
    data = data[data.positive_labels.apply(select.issuperset)]

    return data


if __name__ == "__main__":

    # ontology = Ontology("ontology.json")
    # ontology.graph("ontology-human.dot", subgraph=ontology.children('/m/0dgw9r'))

    data = main("unbalanced_train_segments.csv")  # skip_cat='/m/0dgw9r')  # all human sounds
    data.positive_labels = data.positive_labels.apply(",".join)
    data.to_csv("unbalanced_train_segments_no_human.csv", index=False, header=False,
                line_terminator='\n', float_format="%.3f")
