"""
Guardian v4.0 - Active Ethics Co-Pilot
HYMetaLab Ethical Alignment System
"""

__version__ = "4.0-alpha"
__author__ = "HYMetaLab Lab Tech Team"

from .dashboard_patch_v4 import GuardianDashboardPatch
from .guardian_v4 import GuardianV4
from .integration.hooks import GuardianHooks
from .metrics.risk_assessor import RiskAssessor, TransparencyIndexV2
from .nlp.objectivity_model import ObjectivityModel
from .nlp.sentiment_analyzer import SentimentAnalyzer

__all__ = [
    "GuardianV4",
    "ObjectivityModel",
    "SentimentAnalyzer",
    "RiskAssessor",
    "TransparencyIndexV2",
    "GuardianHooks",
    "GuardianDashboardPatch",
]
