from scipy.cluster.hierarchy import fcluster, dendrogram
import numpy as np
from numpy import ndarray as Matrix, ndarray as Vector
import matplotlib.pyplot as plt
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

def find_optimal_clusters(linkage_matrix: Matrix) -> Vector:
    '''
    loops through each k, evaluates the clustering produced by that k, and picks the optimal one.
    returns the clustering produced by fcluster() on that optimal k
    '''
    raise

# -------------------------------------------------------------

def plot_dendrogram(
    linkage_matrix: Matrix, mapping: dict[str, int], tip_type: str = "species"
) -> None:
    ax = plt.gca()
    dendrogram(
        linkage_matrix,
        ax = ax,
        orientation = "left",
        leaf_label_func = lambda index: {index: clade for clade, index in mapping.items()}[index]
    )
    plt.xlabel("Divergence Time (MYA)")
    plt.ylabel(tip_type.title())
    ax.yaxis.set_label_position("right")
    root_name: str = max(mapping, key = lambda clade: mapping[clade])
    plt.title(f"Time-Calibrated Phylogenetic Tree of {root_name}")
    plt.show()