import toolz


def write_dzn(data, fname, threshold=500):
    "Produce input data for MiniZinc solver"
    labels_idx = {label: i for (i, label) in enumerate(
        sorted(set(toolz.concat(data.positive_labels))))}
    with open(fname, "w") as f:
        print("THRESHOLD = %d;" % threshold, file=f)
        print("NFILES = 1..%d;" % len(data), file=f)
        print("NLABELS = 1..%d;" % len(labels_idx), file=f)
        print("data = [", end="", file=f)
        for s in data.positive_labels:
            row = ["0"] * len(labels_idx)
            for label in s:
                row[labels_idx[label]] = "1"
            print("| %s" % ", ".join(row), file=f)
        print("|];", file=f)
