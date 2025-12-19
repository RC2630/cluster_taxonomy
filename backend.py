from scipy.cluster.hierarchy import fcluster, dendrogram
import numpy as np
from numpy import ndarray as Matrix, ndarray as Vector
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
import re

# -------------------------------------------------------------

class Split:

    parent_clade: str      # the name of the parent clade
    left_child_clade: str  # the name of the 1st (left) child clade/species
    right_child_clade: str # the name of the 2nd (right) child clade/species
    split_time: float      # the time this split occurred, in MYA counting from the present
    size: int              # the number of species of the parent clade
    orig_string: str       # the original string that was parsed to get this Split

    def __init__(self, line: str) -> None:
        '''
        parses 1 line from input file to a Split
        '''
        match: re.Match[str] | None = re.match(r"(.+) -> (.+), (.+) \((.+)\)", line)
        if match is None:
            raise ValueError("line is not in the correct format")
        self.parent_clade = match.groups()[0]
        self.left_child_clade = match.groups()[1]
        self.right_child_clade = match.groups()[2]
        self.split_time = float(match.groups()[3])
        self.size = -1 # we will work on the sizes later
        self.orig_string = line.strip()

    def __repr__(self) -> str:
        return self.orig_string

# -------------------------------------------------------------

def parse_file(file: str) -> tuple[list[Split], dict[str, int]]:
    '''
    parses the input file, and returns (splits, mapping from clade/species name to index).
    index is for use in the linkage matrix later
    '''

    with open(file, "r") as f:
        lines: list[str] = f.readlines()
    splits: list[Split] = [Split(line) for line in lines]
    splits.sort(key = lambda split: split.split_time)

    clade_mapping: dict[str, int] = {}
    counter: int = 0
    for split in splits:
        clade_mapping[split.parent_clade] = counter
        counter += 1
    
    full_mapping: dict[str, int] = {}
    num_species: int = 0
    for split in splits:
        l: str = split.left_child_clade
        r: str = split.right_child_clade
        if l not in clade_mapping and l not in full_mapping:
            full_mapping[l] = num_species
            num_species += 1
        if r not in clade_mapping and r not in full_mapping:
            full_mapping[r] = num_species
            num_species += 1

    for clade, clade_index in clade_mapping.items():
        full_mapping[clade] = num_species + clade_index

    sizes: dict[str, int] = {}
    def get_and_set_size(clade_or_species: str | Split) -> int:
        name: str = clade_or_species if isinstance(clade_or_species, str) \
                    else clade_or_species.parent_clade
        if name in sizes:
            return sizes[name]
        if full_mapping[name] < num_species:
            sizes[name] = 1 # not a clade, but a terminal species
        else:
            split: Split = next(split for split in splits if split.parent_clade == name) \
                           if isinstance(clade_or_species, str) else clade_or_species
            if split.size == -1:
                left_size: int = get_and_set_size(split.left_child_clade)
                right_size: int = get_and_set_size(split.right_child_clade)
                split.size = left_size + right_size
            sizes[name] = split.size
        return sizes[name]

    for split in splits:
        if split.size == -1:
            get_and_set_size(split)

    return (splits, full_mapping)

# -------------------------------------------------------------

def construct_linkage_matrix(splits: list[Split], mapping: dict[str, int]) -> Matrix:
    '''
    creates a valid scipy linkage matrix from the splits data
    '''
    data: list[list[float]] = []
    for split in splits:
        data.append([
            mapping[split.left_child_clade],
            mapping[split.right_child_clade],
            split.split_time,
            split.size
        ])
    return np.array(data)

# -------------------------------------------------------------

def find_optimal_clusters(
    linkage_matrix: Matrix, intensity: float = 1.5, curve: float = 1.1,
    correction_strength: float = 0.5, print_scaled_diffs: bool = False
) -> tuple[Vector, float, int]:
    '''
    finds the optimal height at which to cut the dendrogram, then
    returns (labels, height, num_clusters),
    where `height` is the optimal height, `num_clusters` is the number of clusters,
    and `labels` is the clustering produced by fcluster() with that height
    '''

    def build_scaling_weights(n: int, intensity: float, curve: float) -> Vector:
        '''
        encourages not too many and not too few clusters by explicitly favouring
        cluster counts around the square root of the number of tips
        '''
        if n == 0: return np.array([])
        if n == 1: return np.array([1])
        if n == 2: return np.array([1, 1])
        if n == 3: return np.array([1, ((intensity + 1) / 2) ** curve, 1])
        denom: int = int(np.round(np.sqrt(n - 1))) - 1
        left: Vector = np.arange(1, intensity, (intensity - 1) / denom)
        right: Vector = np.arange(intensity, 1, -(intensity - 1) / (n - 1 - denom))
        return np.concatenate([left, right, np.array([1])]) ** curve

    heights: Vector = linkage_matrix[:, 2]
    diffs: Vector = heights[1:] - heights[:-1]
    # the line below gently nudges the model towards sqrt(n) clusters (parameters are tunable)
    diffs *= build_scaling_weights(len(diffs), intensity, curve)[::-1]
    # the line below corrects for reduced waiting time for a split when lots of lineages exist
    diffs *= np.arange(len(diffs) + 1, 1, -1) ** correction_strength

    if print_scaled_diffs:
        cluster_counts: list[int] = list(range(2, len(diffs) + 2))[::-1]
        info: dict[str, float] = \
            {f"{count} clusters": float(diff) for count, diff in zip(cluster_counts, diffs)}
        for count, diff in info.items():
            print(f"{count}: {diff}")

    largest_diff_index: int = int(np.argmax(diffs))
    optimal_height: float = (heights[largest_diff_index] + heights[largest_diff_index + 1]) / 2
    labels: Vector = fcluster(linkage_matrix, optimal_height, "distance")
    num_clusters: int = np.max(labels)
    return (labels, float(optimal_height), int(num_clusters))

# -------------------------------------------------------------

def plot_dendrogram(
    linkage_matrix: Matrix, mapping: dict[str, int],
    tip_type: str = "species", num_clusters: int | None = None
) -> None:
    '''
    plots the linkage matrix as a dendrogram (time-calibrated phylogenetic tree),
    labelling the indices as their species (or other tip type) names
    '''
    height_threshold: float | None = None
    if num_clusters is not None:
        height_threshold = linkage_matrix[-num_clusters + 1, 2]
    ax = plt.gca()
    dendrogram(
        linkage_matrix,
        ax = ax,
        orientation = "left",
        leaf_label_func = lambda index: {index: clade for clade, index in mapping.items()}[index],
        color_threshold = height_threshold
    )
    plt.xlabel("Divergence Time (MYA)")
    plt.ylabel(tip_type.title())
    ax.yaxis.set_label_position("right")
    root_name: str = max(mapping, key = lambda clade: mapping[clade])
    plt.title(f"Time-Calibrated Phylogenetic Tree of {root_name} ({num_clusters} clusters)")
    plt.show()

# -------------------------------------------------------------

def get_clusters_rich_info(
    clusters: Vector, mapping: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    given the raw output fcluster(), provides 2 dataframes
    (1 for the species, and 1 for the clusters)
    containing various info about them that may be useful
    '''
    
    num_clusters: int = np.max(clusters)
    species_data: list[dict[str, Any]] = []
    reverse_mapping: dict[int, str] = {index: clade for clade, index in mapping.items()}

    for index, entry in enumerate(clusters):
        species_data.append({
            "tip": reverse_mapping[index],
            "index": index,
            "cluster": int(entry)
        })

    cluster_data: list[dict[str, Any]] = []
    for cluster in range(1, num_clusters + 1):
        data: dict[str, Any] = {}
        data["cluster"] = cluster
        tips: list[str] = [
            str(species["index"]) for species in species_data if species["cluster"] == cluster
        ]
        data["tips"] = ",".join(tips)
        data["num_tips"] = len(tips)
        cluster_data.append(data)

    return (
        pd.DataFrame.from_records(species_data).set_index("tip"),
        pd.DataFrame.from_records(cluster_data).set_index("cluster")
    )