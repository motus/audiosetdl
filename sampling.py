import toolz


def write_dzn(data, fname, threshold=500):
    "Produce input data for MiniZinc solver"
    labels_idx = {label: i for (i, label) in enumerate(
        sorted(set(toolz.concat(data.positive_labels))))}
    labels_count = [0] * len(labels_idx)
    with open(fname, "w") as f:
        print("THRESHOLD = %d;" % threshold, file=f)
        print("NFILES = 1..%d;" % len(data), file=f)
        print("NLABELS = 1..%d;" % len(labels_idx), file=f)
        print("DATA = [", end="", file=f)
        for s in data.positive_labels:
            row = ["0"] * len(labels_idx)
            for label in s:
                i = labels_idx[label]
                row[i] = "1"
                if labels_count[i] < threshold:
                    labels_count[i] += 1
            print("| %s" % ", ".join(row), file=f)
        print("|];", file=f)
        print("LIMITS = %s;" % labels_count, file=f)
