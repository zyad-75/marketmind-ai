from enum import Enum
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# ==========================================
# Intent Types
# ==========================================

class IntentType(str, Enum):
    FACT_LOOKUP = "FACT_LOOKUP"
    CAMPAIGN_GENERATION = "CAMPAIGN_GENERATION"
    SEGMENT_ANALYSIS = "SEGMENT_ANALYSIS"


# ==========================================
# Campaign Idea
# ==========================================

class CampaignIdea(BaseModel):

    idea: str = Field(
        description="The campaign idea"
    )

    classification: Literal[
        "knowledge_based",
        "reasonable_inference",
        "general_recommendation"
    ] = Field(
        description="How strongly the idea is grounded in the knowledge base"
    )

    evidence: Optional[str] = Field(
        default=None,
        description=(
            "Evidence from the retrieved knowledge base. "
            "Must be provided for knowledge_based ideas."
        )
    )


# ==========================================
# Campaign Response
# ==========================================

class CampaignResponse(BaseModel):

    campaign_name: str

    objective: str

    target_audience: str

    key_message: str

    recommended_channels: List[str]

    campaign_ideas: List[CampaignIdea]

    reasoning: str

    general_recommendations: List[str]

    confidence: Literal[
        "High",
        "Medium",
        "Low"
    ]

    sources: List[str]


# ==========================================
# Fact Response
# ==========================================

class FactResponse(BaseModel):

    answer: str

    confidence: Literal[
        "High",
        "Medium",
        "Low"
    ]

    sources: List[str]


# ==========================================
# Segment Response
# ==========================================

class SegmentResponse(BaseModel):

    segment_name: str

    characteristics: List[str]

    pain_points: List[str]

    digital_behavior: List[str]

    buying_motivation: List[str]

    marketing_recommendations: List[str]

    confidence: Literal[
        "High",
        "Medium",
        "Low"
    ]

    sources: List[str]