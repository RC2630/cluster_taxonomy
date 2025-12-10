# Cluster Taxonomy

This is a fairly rudimentary stab at computational biology by using hierarchical clustering techniques on a time-calibrated phylogenetic tree.

More specifically, I skip the linkage step and construct the dendrogram directly from the data, since the phylogenetic tree is already a well-formed dendrogram and can be converted to a `scipy`-accepted format using some processing.

For example, if the phylogenetic tree in the data is rooted at a family, and has tips for each species, then this clustering procedure can be used to find and delimit plausible genera for that family.