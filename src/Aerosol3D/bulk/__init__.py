"""Bulk aerosol optics aggregation."""

from Aerosol3D.bulk.builder import BulkOpticsBuilder
from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData, SizeDistribution
from Aerosol3D.bulk.density import compute_effective_density
from Aerosol3D.bulk.merge import compute_bin_weights, merge_method1, merge_method2

__all__ = [
    "BulkAerosolOpticsData",
    "BulkOpticsBuilder",
    "SizeDistribution",
    "compute_bin_weights",
    "compute_effective_density",
    "merge_method1",
    "merge_method2",
]
