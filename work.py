from scipy.cluster.hierarchy import dendrogram
from backend import *
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

splits, mapping = parse_file("data.txt")
linkage_matrix: Matrix = construct_linkage_matrix(splits, mapping)

# ----------------------------------------------------------------------

indices = pd.DataFrame([[clade, index] for clade, index in mapping.items()])
indices.columns = ["Clade/Species", "Index"]
indices = indices.set_index("Index").T

# ----------------------------------------------------------------------

def process(row: Vector) -> list[int | float]:
    data: list[int | float] = []
    for index, entry in enumerate(row):
        data.append(float(entry) if index == 2 else int(entry))
    return data

assert len(splits) == len(linkage_matrix)

for i in range(len(splits)):
    print(f"{splits[i]}\n{process(linkage_matrix[i])}\n")

indices

# ----------------------------------------------------------------------

ax = plt.gca()
dendrogram(
    linkage_matrix,
    ax = ax,
    leaf_label_func = lambda index: "\n".join({index: clade for clade, index in mapping.items()}[index])
)
plt.xticks(fontsize = 7)
plt.xlabel("Species")
plt.ylabel("Divergence Time (MYA)")
plt.title("Time-Calibrated Phylogenetic Tree of Ursidae")
plt.show()