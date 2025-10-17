"""
Source adapters for loading data from various sources.
"""

from .base import SourceAdapter
from .generic_api import GenericAPISource
from .opendata_csv import OpenDataCSVSource
from .trends_csv import TrendsCSVSource

__all__ = ["SourceAdapter", "OpenDataCSVSource", "TrendsCSVSource", "GenericAPISource"]
