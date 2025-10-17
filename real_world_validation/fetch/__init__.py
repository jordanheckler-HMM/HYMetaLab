"""
Data fetchers for real-world validation system.
"""

from .census_acs import CensusACSClient
from .fema_disasters import FEMAClient
from .socrata_client import SocrataClient
from .trends_pytrends import TrendsClient
from .worldbank_gini import WorldBankClient

__all__ = [
    "SocrataClient",
    "CensusACSClient",
    "TrendsClient",
    "FEMAClient",
    "WorldBankClient",
]
