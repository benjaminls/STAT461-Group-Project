"""
Functions for clustering detector hits using DBSCAN in eta-phi space.

This module provides utilities to cluster particle detector hits based on their
pseudorapidity (eta) and azimuthal angle (phi) coordinates using DBSCAN.
The clustering helps identify groups of hits likely originating from the same particle.

Key functionality:
- DBSCAN clustering in (eta,phi) space
- Configurable clustering parameters (epsilon, min_samples)
- Noise point identification
- Cluster label assignment to hits
"""
