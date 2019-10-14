#!/usr/bin/env python3

import json
import itertools

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

    def graph(self, fname, subgraph=None, highlight=frozenset()):

        all_nodes = self._nodes.values()
        if subgraph:
            all_nodes = [node for node in self._nodes.values() if node['id'] in subgraph]

        with open(fname, "wt") as f:
            f.write("digraph G {\n")
            for node in all_nodes:
                node_id = node['id']
                f.write('  "%s" [label="%s"%s];\n' % (
                        node_id, node['name'], ", style=filled" if node_id in highlight else ""))
                children = node.get("child_ids")
                if children:
                    f.write('  "%s" -> { "%s" };\n' % (node_id, '","'.join(children)))
            f.write("}\n")

    def all_children(self, node_ids, children_dict=None):
        if children_dict is None:
            children_dict = {}
        for node_id in node_ids:
            self.children(node_id, children_dict)
        return children_dict

    def children(self, node_id, children_dict=None):

        if children_dict is None:
            children_dict = {}

        node = self._nodes.get(node_id)
        if node:
            children_dict[node_id] = node.get('name')
            for n in node.get('child_ids', []):
                self.children(n, children_dict)

        return children_dict

    def top(self, level=0):
        nodes = set(self._nodes.keys())
        for node in self._nodes.values():
            nodes.difference_update(node.get('child_ids', {}))
        frontier = nodes
        for _i in range(level):
            frontier = frozenset(itertools.chain.from_iterable(
                self._nodes[node_id].get('child_ids', {}) for node_id in frontier))
            nodes.update(frontier)
        return nodes


def read_categories(fname, ontology):
    data = pandas.read_csv(fname, sep="\t", header=None, names=["name", "quality", "num"])
    data.quality = data.quality.apply(lambda s: float('0' + s[:-1]) / 100.)
    data.num = data.num.apply(lambda s: int(s.replace(',', '')))
    data['label'] = data.name.apply(lambda n: ontology.get_by_name(n, {}).get('id'))
    return data


def main(fname, ontology, skip_cat=None):

    skipset = frozenset()
    if skip_cat is not None:
        # skip_cat = '/m/0dgw9r'  # all human sounds
        # skip_cat = '/m/09l8g'   # human voice subcategory
        skipset = frozenset(ontology.children(skip_cat).keys())

    data = pandas.read_csv(
        fname, skiprows=3, skipinitialspace=True, header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"])

    accuracy = read_categories("accuracy.tsv", ontology)
    accuracy = accuracy[accuracy.quality >= 0.9]

    include = frozenset(accuracy.label)
    exclude = frozenset(ontology.all_children(read_categories("exclude.tsv", ontology).label))

    select = include.difference(skipset.union(exclude))

    data.positive_labels = data.positive_labels.apply(lambda s: s.split(','))
    data = data[data.positive_labels.apply(select.issuperset)]

    return data


if __name__ == "__main__":

    ontology = Ontology("ontology.json")
    ontology.graph(
        "ontology-human.dot",
        subgraph=ontology.children('/m/0dgw9r'),
        highlight=frozenset(ontology.all_children(read_categories("exclude.tsv", ontology).label)))

    # data = main("unbalanced_train_segments.csv", ontology)
    # data.positive_labels = data.positive_labels.apply(",".join)
    # data.to_csv("unbalanced_train_segments_no_human.csv", index=False, header=False,
    #             line_terminator='\n', float_format="%.3f")
