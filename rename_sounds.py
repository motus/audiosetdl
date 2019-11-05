#!/usr/bin/env python3

import os
import sys
import glob

import ontology as ont


class OntologyRenamer:

    def __init__(self, data, ontology):
        self.data = data
        self.ontology = ontology

    def parse_fname(self, path):
        fname = os.path.basename(path)[:-4]
        split = fname.split("_")
        ts_start, ts_end = [float(n) / 1000 for n in split[-2:]]
        ytid = "_".join(split[:-2])
        return (ytid, ts_start, ts_end)

    def lookup_file(self, path):
        (ytid, ts_start, ts_end) = self.parse_fname(path)
        return self.data[(self.data.YTID == ytid) &
                         (self.data.start_seconds == ts_start) &
                         (self.data.end_seconds == ts_end)]

    def mturk_name(self, path):
        fname = os.path.basename(path)
        df = self.lookup_file(fname)
        if len(df) != 1:
            return None
        return "%s_%s%s" % (fname[:-4], "_".join(
            self.ontology.names(df.positive_labels.iloc[0])), fname[-4:])

    def rename_mturk_all(self, source, dest, dry_run=False):
        if not os.path.exists(dest):
            os.mkdir(dest)
        for path in glob.glob(source):
            fname = self.mturk_name(path)
            if fname:
                fname_dst = os.path.join(dest, fname)
                print("%s --> %s" % (path, fname_dst))
                if not dry_run:
                    os.rename(path, fname_dst)


def main(source, dest):
    ontology = ont.Ontology("ontology.json")
    data = ont.load_data("unbalanced_train_segments.csv", skip=3)
    renamer = OntologyRenamer(data, ontology)
    renamer.rename_mturk_all(source, dest, dry_run=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 %s SOURCE DESTINATION\n"
              "  e.g. python3 %s ./data-in/*.wav ./data-out/" % (sys.argv[0], sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
