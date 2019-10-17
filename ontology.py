#!/usr/bin/env python3

import os
import json
import random
import itertools

import pandas


class Ontology:

    def __init__(self, fname):
        self._nodes = {}
        for node in json.load(open(fname)):
            node['parents'] = []
            self._nodes[node['id']] = node
        self._nodes_by_name = {}
        for node in list(self._nodes.values()):
            for node_id in node.get('child_ids', []):
                self._nodes[node_id]['parents'].append(node['id'])
            self._nodes_by_name[node['name']] = node

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

    def children(self, node_id, collect_dict=None):
        return self.paths('child_ids', node_id, collect_dict)

    def parents(self, node_id, collect_dict=None):
        return self.paths('parents', node_id, collect_dict)

    def paths(self, attr_name, node_id, collect_dict=None):
        if collect_dict is None:
            collect_dict = {}
        node = self._nodes.get(node_id)
        if node:
            collect_dict[node_id] = node.get('name')
            for n in node.get(attr_name, []):
                self.paths(attr_name, n, collect_dict)
        return collect_dict

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


def stratified_sample(data, ontology, categories, threshold):
    idx = set()
    for cat in categories:
        size = threshold - data.loc[idx].positive_labels.apply(lambda s: cat in s).sum()
        print(f"*** select {size} for {cat} {ontology.names([cat])[0]}")
        if size > 0:
            sample = data[data.positive_labels.apply(lambda s: cat in s)].index.to_list()
            if size < len(sample):
                sample = random.sample(sample, size)
            idx.update(sample)
    return data.loc[idx]


def load_selection_set(ontology, skipset=frozenset(), threshold=0.9):
    accuracy = read_categories("accuracy.tsv", ontology)
    accuracy = accuracy[accuracy.quality >= threshold]
    include = frozenset(accuracy.label)
    exclude = frozenset(ontology.all_children(read_categories("exclude.tsv", ontology).label))
    select = include.difference(skipset.union(exclude))
    accuracy = accuracy[accuracy.label.apply(select.__contains__)]
    return accuracy.label[accuracy.num.sort_values().index].to_list()


def load_data(fname, skip=0):
    data = pandas.read_csv(
        fname, skiprows=skip, skipinitialspace=True, header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"])
    data.positive_labels = data.positive_labels.apply(lambda s: frozenset(s.split(',')))
    return data


def filter_data(data, ontology, sample_size, skip_cat=None):

    skipset = frozenset()
    if skip_cat is not None:
        # skip_cat = '/m/0dgw9r'  # all human sounds
        # skip_cat = '/m/09l8g'   # human voice subcategory
        skipset = frozenset(ontology.children(skip_cat).keys())

    select = load_selection_set(ontology, skipset)
    data = data[data.positive_labels.apply(frozenset(select).issuperset)]

    data = stratified_sample(data, ontology, select, sample_size)

    return data


def parse_fname(path):
    fname = os.path.basename(path)[:-4]
    split = fname.split("_")
    ts_start, ts_end = [float(n) / 1000 for n in split[-2:]]
    ytid = "_".join(split[:-2])
    return (ytid, ts_start, ts_end)


def lookup_file(data, path):
    (ytid, ts_start, ts_end) = parse_fname(path)
    return data[(data.YTID == ytid) &
                (data.start_seconds == ts_start) &
                (data.end_seconds == ts_end)]


def rename_mturk(data, ontology, path):
    fname = os.path.basename(path)
    df = lookup_file(data, fname)
    if len(df) != 1:
        return fname  # None?
    return "%s_%s%s" % (
        fname[:-4], "_".join(ontology.names(df.positive_labels.iloc[0])), fname[-4:])


_CAT_SAMPLE_SIZE = 1000

if __name__ == "__main__":

    ontology = Ontology("ontology.json")
    # ontology.graph(
    #     "ontology-human.dot",
    #     subgraph=ontology.children('/m/0dgw9r'),
    #     highlight=frozenset(ontology.all_children(
    #         read_categories("exclude.tsv", ontology).label)))

    data = load_data("unbalanced_train_segments.csv", skip=3)
    data = filter_data(data, ontology, _CAT_SAMPLE_SIZE)
    data.positive_labels = data.positive_labels.apply(",".join)
    data.to_csv("unbalanced_train_segments_no_human_stratified_%d.csv" % _CAT_SAMPLE_SIZE,
                index=False, header=False, line_terminator='\n', float_format="%.3f")
