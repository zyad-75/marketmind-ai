import re
from functools import lru_cache
from typing import Type

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.rag import retrieve_context
from src.schemas import (
    IntentType,
    FactResponse,
    CampaignResponse,
    SegmentResponse,
)


# ============================================================
# Model Configuration
# ============================================================

OLLAMA_MODEL = "llama3.2:1b"


# ============================================================
# Local LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm():
    """
    Lightweight local LLM for the MarketMind demo.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_gpu=0,
        num_ctx=4096,
        num_predict=768,
    )


# ============================================================
# Query Normalization
# ============================================================

def normalize_query(
    user_query: str
) -> str:

    query = (
        user_query
        .lower()
        .strip()
    )

    query = query.replace(
        "-",
        " "
    )

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query


# ============================================================
# Intent Router
# ============================================================

def detect_intent(
    user_query: str
) -> IntentType:

    query = normalize_query(
        user_query
    )

    query_words = set(
        re.findall(
            r"\b\w+\b",
            query,
        )
    )

    # ========================================================
    # Campaign Generation
    # ========================================================

    campaign_phrases = [
        "create a campaign",
        "generate a campaign",
        "design a campaign",
        "build a campaign",
        "plan a campaign",
        "marketing campaign",
        "launch campaign",
        "campaign strategy",
        "marketing strategy",
        "campaign idea",
        "campaign ideas",
        "give me campaign ideas",
        "suggest campaign ideas",
        "create content",
        "generate content",
        "suggest content",
        "marketing content",
        "create video",
        "create videos",
        "generate video",
        "generate videos",
        "video series",
        "create post",
        "create posts",
        "generate post",
        "generate posts",
        "write post",
        "write posts",
        "create ad",
        "create ads",
        "generate ad",
        "generate ads",
        "social media campaign",
        "instagram campaign",
        "tiktok campaign",
        "youtube campaign",
    ]

    if any(
        phrase in query
        for phrase in campaign_phrases
    ):
        return IntentType.CAMPAIGN_GENERATION

    campaign_actions = {
        "create",
        "generate",
        "design",
        "develop",
        "suggest",
        "write",
        "plan",
        "build",
        "prepare",
        "make",
    }

    marketing_objects = {
        "campaign",
        "campaigns",
        "video",
        "videos",
        "content",
        "post",
        "posts",
        "reel",
        "reels",
        "ad",
        "ads",
        "advertisement",
        "advertisements",
        "promotion",
        "promotions",
        "strategy",
        "creative",
        "creatives",
    }

    has_action = bool(
        campaign_actions
        & query_words
    )

    has_marketing_object = bool(
        marketing_objects
        & query_words
    )

    if (
        has_action
        and has_marketing_object
    ):
        return IntentType.CAMPAIGN_GENERATION

    # ========================================================
    # Segment Analysis
    # ========================================================

    segment_phrases = [
        "customer segment",
        "customer segments",
        "analyze segment",
        "analyse segment",
        "analyze customer segment",
        "analyse customer segment",
        "segment analysis",
        "customer persona",
        "customer personas",
        "analyze persona",
        "analyse persona",
        "target audience",
        "audience analysis",
        "analyze audience",
        "analyse audience",
        "customer behavior",
        "customer behaviour",
        "analyze customer",
        "analyse customer",
    ]

    if any(
        phrase in query
        for phrase in segment_phrases
    ):
        return IntentType.SEGMENT_ANALYSIS

    if "persona" in query_words:

        analysis_words = {
            "analyze",
            "analyse",
            "describe",
            "explain",
            "understand",
        }

        if (
            analysis_words
            & query_words
        ):
            return IntentType.SEGMENT_ANALYSIS

    # ========================================================
    # Default
    # ========================================================

    return IntentType.FACT_LOOKUP


# ============================================================
# Deterministic Fact Extraction
# ============================================================

def deterministic_fact_lookup(
    user_query: str,
    documents,
):
    """
    Deterministically extracts explicit numerical facts.

    Supported:
    - CTR
    - impressions
    - clicks
    - conversions
    - persona age ranges

    Entity-aware matching prevents a value belonging to one
    campaign/channel/persona from being returned for another.
    """

    query = normalize_query(
        user_query
    )

    metric = None
    pattern = None

    # ========================================================
    # Metric Detection
    # ========================================================

    if "ctr" in query:

        metric = "ctr"

        pattern = (
            r"\bCTR\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?%)"
        )

    elif (
        "impression" in query
        or "impressions" in query
    ):

        metric = "impressions"

        pattern = (
            r"\bImpressions\s*:\s*"
            r"([\d,]+)"
        )

    elif (
        "click" in query
        or "clicks" in query
    ):

        metric = "clicks"

        pattern = (
            r"\bClicks\s*:\s*"
            r"([\d,]+)"
        )

    elif (
        "conversion" in query
        or "conversions" in query
    ):

        metric = "conversions"

        pattern = (
            r"\bConversions\s*:\s*"
            r"([\d,]+)"
        )

    elif (
        "age range" in query
        or "what age" in query
        or "age of" in query
    ):

        metric = "age"

        pattern = (
            r"\bAge\s*:\s*"
            r"([0-9]{1,2}"
            r"\s*[–-]\s*"
            r"[0-9]{1,2})"
        )

    else:

        return None

    # ========================================================
    # Query Entity Identification
    # ========================================================

    stop_words = {
        "what",
        "was",
        "were",
        "the",
        "of",
        "did",
        "does",
        "how",
        "many",
        "much",
        "is",
        "are",
        "a",
        "an",
        "for",
        "to",
        "in",
        "on",
        "from",
        "novaglow",
        "campaign",
        "campaigns",
        "rate",
        "metric",
        "result",
        "results",
        "range",
        "age",
        "ctr",
        "impressions",
        "impression",
        "clicks",
        "click",
        "conversions",
        "conversion",
        "receive",
        "received",
    }

    raw_query_words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query,
    )

    identifier_words = [
        word
        for word in raw_query_words
        if (
            len(word) > 2
            and word not in stop_words
        )
    ]

    # ========================================================
    # Search Retrieved Documents
    # ========================================================

    candidates = []

    for document in documents:

        text = document.page_content

        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            start = max(
                0,
                match.start() - 500,
            )

            end = min(
                len(text),
                match.end() + 250,
            )

            local_context = (
                text[start:end]
                .lower()
            )

            matched_identifiers = [
                word
                for word in identifier_words
                if word in local_context
            ]

            source = (
                document
                .metadata
                .get(
                    "source",
                    "Unknown",
                )
            )

            page = (
                document
                .metadata
                .get(
                    "page",
                    0,
                )
            )

            candidates.append(
                {
                    "value": match.group(1),
                    "score": len(
                        matched_identifiers
                    ),
                    "source": source,
                    "page": page,
                    "metric": metric,
                    "context": local_context,
                }
            )

    # ========================================================
    # No Value Found
    # ========================================================

    if not candidates:

        return {
            "status": "unsupported",
            "metric": metric,
        }

    # ========================================================
    # Entity-Aware Validation
    # ========================================================

    if identifier_words:

        valid_candidates = []

        for candidate in candidates:

            all_identifiers_present = all(
                word in candidate["context"]
                for word in identifier_words
            )

            if all_identifiers_present:

                valid_candidates.append(
                    candidate
                )

        if not valid_candidates:

            return {
                "status": "unsupported",
                "metric": metric,
            }

        candidates = valid_candidates

    else:

        if len(candidates) > 1:

            return {
                "status": "ambiguous",
                "metric": metric,
            }

    # ========================================================
    # Best Candidate
    # ========================================================

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    source_label = (
        f"{best['source']} - "
        f"page {best['page'] + 1}"
    )

    return {
        "status": "found",
        "answer": best["value"],
        "confidence": "High",
        "sources": [
            source_label
        ],
        "metric": best["metric"],
    }


# ============================================================
# Shared Guardrails
# ============================================================

GUARDRAILS = """
STRICT RULES:

1. Retrieved NovaGlow knowledge is the primary source of truth.

2. Never invent company-specific information.

3. Never invent:
   - CTR values
   - impressions
   - clicks
   - conversions
   - campaign performance
   - customer statistics
   - product features
   - prices
   - discounts
   - historical results
   - medical claims
   - health claims
   - clinical claims
   - efficacy claims
   - before-and-after outcomes

4. If information is unavailable, say:

"The available knowledge base does not contain this information."

5. Never transfer a fact from one product, campaign,
   channel, or persona to another.

6. Creative recommendations are not company facts.

7. Do not add product benefits that are not explicitly
   supported by retrieved knowledge.

8. When persona-specific information exists, prefer it
   over broader brand information.

9. When product-specific information exists, never replace it
   with information belonging to another product.

10. A science-driven brand does NOT automatically mean:
    - scientifically backed
    - clinically proven
    - doctor approved
    - dermatologist approved
    - expert endorsed

11. Confidence:

High:
Direct evidence strongly supports the response.

Medium:
The response combines evidence with reasonable inference.

Low:
Important information is unavailable or weakly supported.

12. Clearly distinguish:
    - grounded knowledge
    - reasonable inference
    - general recommendations
"""


# ============================================================
# Fact Prompt
# ============================================================

def build_fact_prompt(
    user_query: str,
    context: str,
    sources: list[str],
):

    return f"""
You are MarketMind AI.

Answer a factual question about NovaGlow.

{GUARDRAILS}

============================================================
RETRIEVED KNOWLEDGE
============================================================

{context}

============================================================
TRUSTED SOURCES
============================================================

{sources}

============================================================
USER QUESTION
============================================================

{user_query}

============================================================
FACT RULES
============================================================

- Answer only the requested fact.
- Do not generate campaigns.
- Do not give marketing recommendations.
- Do not estimate missing values.
- Do not transfer metrics between campaigns,
  products, personas, or channels.

If the exact requested information is unsupported,
answer exactly:

"The available knowledge base does not contain this information."

Use High confidence only when directly supported.

Use Low confidence when the requested fact is unavailable.
"""


# ============================================================
# Campaign Prompt
# ============================================================

def build_campaign_prompt(
    user_query: str,
    context: str,
    sources: list[str],
):

    return f"""
You are MarketMind AI, a marketing strategist for NovaGlow.

{GUARDRAILS}

============================================================
RETRIEVED KNOWLEDGE
============================================================

{context}

============================================================
TRUSTED SOURCES
============================================================

{sources}

============================================================
USER REQUEST
============================================================

{user_query}

============================================================
TASK
============================================================

Create a structured marketing campaign.

Every campaign idea MUST be classified as exactly one of:

knowledge_based
reasonable_inference
general_recommendation


KNOWLEDGE_BASED:

Use ONLY when the actual campaign execution is directly
supported by retrieved company information.

Evidence is required.


REASONABLE_INFERENCE:

Use when retrieved company information supports the logic,
but the exact execution is not explicitly stated.

Evidence must refer to real retrieved company information.


GENERAL_RECOMMENDATION:

Use for a new creative marketing tactic that is not directly
supported by retrieved company information.

Evidence MUST be null.


============================================================
FEW-SHOT EXAMPLE
============================================================

Retrieved knowledge:

"Short educational videos performed better than static images."

"Simple morning-routine content generated higher engagement."

Idea:

"Create short educational morning-routine videos."

Classification:

knowledge_based

Evidence:

"Short educational videos performed better than static images
and simple morning-routine content generated higher engagement."


============================================================
CLASSIFICATION RULES
============================================================

Unless explicitly supported by retrieved company knowledge,
these MUST be general_recommendation:

- influencer partnerships
- influencers
- contests
- challenges
- giveaways
- paid advertising
- branded hashtags
- partnerships
- expert campaigns
- doctors
- dermatologists


============================================================
CAMPAIGN RULES
============================================================

- Do not invent expected CTR.
- Do not invent expected conversions.
- Do not invent sales growth.
- Do not invent campaign performance.
- Do not invent product benefits.
- Do not claim that a product is scientifically backed.
- Do not claim that a product is clinically proven.
- Do not claim dermatologist approval.
- Do not claim doctor approval.
- Do not invent prices or discounts.
- Do not mix product-specific information.

- Prefer channels supported by persona behavior
  or previous campaign history.

- If the request concerns HydraBoost Serum,
  do not use facts belonging to another product.

- If the request concerns Young Professional,
  use its specific persona information.

- general_recommendations must not duplicate campaign_ideas.

- If campaign_ideas already cover the useful recommendations,
  return an empty general_recommendations list.

- Maximum 3 general recommendations.

- Reasoning must explain what is grounded,
  what is inference,
  and what is only a general recommendation.

Confidence MUST be one of:

High
Medium
Low
"""


# ============================================================
# Segment Prompt
# ============================================================

def build_segment_prompt(
    user_query: str,
    context: str,
    sources: list[str],
):

    return f"""
You are MarketMind AI.

Analyze a NovaGlow customer segment.

{GUARDRAILS}

============================================================
RETRIEVED KNOWLEDGE
============================================================

{context}

============================================================
TRUSTED SOURCES
============================================================

{sources}

============================================================
USER REQUEST
============================================================

{user_query}

============================================================
SEGMENT RULES
============================================================

Use only customer information supported by retrieved knowledge.

The characteristics field must contain the actual values.

Correct examples:

"Age: 25-34"

"Occupation: Corporate employees, consultants, engineers
and technology professionals"

Do NOT return only labels such as:

"age"

"occupation"

Pain points belong only in the pain_points field.

Do not place a sentence beginning with "Pain Points:"
inside characteristics.

Do not invent:
- income
- locations not stated in the knowledge base
- demographic statistics
- professions
- interests
- pain points
- digital behavior
- buying motivations

If multiple age ranges exist,
prefer the age range belonging specifically to the requested persona.

Marketing recommendations may use reasonable inference.

Never present marketing recommendations as known customer facts.

Confidence MUST be one of:

High
Medium
Low
"""


# ============================================================
# Memory Error Detection
# ============================================================

def is_memory_error(
    error: Exception
) -> bool:

    error_text = str(
        error
    ).lower()

    memory_signals = [
        "out of memory",
        "failed to allocate",
        "unable to allocate",
        "cpu_repack",
        "cpurepack",
        "cudamalloc",
        "cuda malloc",
    ]

    return any(
        signal in error_text
        for signal in memory_signals
    )


# ============================================================
# Structured Output Invocation
# ============================================================

def invoke_structured(
    schema: Type[BaseModel],
    prompt: str,
):
    """
    Uses Ollama native JSON-schema generation.

    If the first response fails Pydantic validation,
    MarketMind retries exactly once.
    """

    llm = get_llm()

    structured_llm = (
        llm.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        )
    )

    # ========================================================
    # Attempt 1
    # ========================================================

    try:

        result = structured_llm.invoke(
            prompt
        )

    except Exception as error:

        if is_memory_error(
            error
        ):

            raise RuntimeError(
                "The local LLM could not allocate enough memory. "
                "Close memory-heavy applications and retry."
            ) from error

        raise RuntimeError(
            "Structured generation failed: "
            f"{error}"
        ) from error

    parsed = result.get(
        "parsed"
    )

    parsing_error = result.get(
        "parsing_error"
    )

    if parsed is not None:

        if isinstance(
            parsed,
            schema,
        ):
            return parsed

        try:

            return schema.model_validate(
                parsed
            )

        except Exception as error:

            parsing_error = error

    # ========================================================
    # Attempt 2
    # ========================================================

    print(
        "Structured output validation failed. "
        "Retrying once..."
    )

    retry_prompt = f"""
{prompt}

============================================================
STRICT STRUCTURED OUTPUT REQUIREMENTS
============================================================

Your previous response failed schema validation.

Generate the response again.

Rules:

1. Fill EVERY required field.

2. Never omit required fields.

3. Use arrays for list fields.

4. Use strings for string fields.

5. Do not add unknown fields.

6. Do not return markdown.

7. Do not include commentary outside the structured object.

8. Campaign classification MUST be exactly one of:

knowledge_based
reasonable_inference
general_recommendation

9. Confidence MUST be exactly one of:

High
Medium
Low

10. If there are no general recommendations,
return an empty array.

11. Evidence for general_recommendation MUST be null.

12. Return exactly one complete valid structured object.
"""

    try:

        retry_result = (
            structured_llm.invoke(
                retry_prompt
            )
        )

    except Exception as error:

        if is_memory_error(
            error
        ):

            raise RuntimeError(
                "The local LLM could not allocate enough memory. "
                "Close memory-heavy applications and retry."
            ) from error

        raise RuntimeError(
            "Structured generation retry failed: "
            f"{error}"
        ) from error

    retry_parsed = (
        retry_result.get(
            "parsed"
        )
    )

    retry_error = (
        retry_result.get(
            "parsing_error"
        )
    )

    if retry_parsed is not None:

        if isinstance(
            retry_parsed,
            schema,
        ):
            return retry_parsed

        try:

            return schema.model_validate(
                retry_parsed
            )

        except Exception as error:

            retry_error = error

    final_error = (
        retry_error
        or parsing_error
        or "Unknown structured-output validation error."
    )

    raise RuntimeError(
        "The model could not produce a valid structured "
        "response after two attempts. "
        f"Details: {final_error}"
    )


# ============================================================
# Text Helpers
# ============================================================

def get_evidence_words(
    text: str
) -> list[str]:

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    stop_words = {
        "about",
        "above",
        "after",
        "again",
        "also",
        "because",
        "been",
        "being",
        "between",
        "could",
        "does",
        "from",
        "have",
        "into",
        "more",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "using",
        "were",
        "which",
        "with",
        "would",
    }

    return [
        word
        for word in words
        if (
            len(word) > 4
            and word not in stop_words
        )
    ]


def normalize_word_set(
    text: str
) -> set[str]:

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )


def dedupe_text_list(
    items: list[str]
) -> list[str]:

    result = []
    seen = set()

    for item in items:

        if not item:
            continue

        cleaned = (
            item.strip()
        )

        if not cleaned:
            continue

        key = (
            cleaned
            .lower()
            .rstrip(".")
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            cleaned
        )

    return result


# ============================================================
# Campaign Grounding Validation
# ============================================================

def validate_campaign_grounding(
    response: CampaignResponse,
    context: str,
):
    """
    Programmatic guardrail for campaign generation.

    Protects against:
    - unsupported scientific / medical claims
    - generic tactics pretending to be knowledge-based
    - fabricated evidence
    - inconsistent reasoning
    - duplicated recommendations
    """

    context_lower = (
        context.lower()
    )

    # ========================================================
    # Generic Tactics
    # ========================================================

    generic_tactics = [
        "influencer partnership",
        "influencer partnerships",
        "influencer",
        "influencers",
        "contest",
        "contests",
        "challenge",
        "challenges",
        "giveaway",
        "giveaways",
        "paid advertising",
        "paid ads",
        "branded hashtag",
        "branded hashtags",
        "partnership",
        "partnerships",
        "expert campaign",
        "expert campaigns",
        "dermatologist",
        "dermatologists",
        "doctor",
        "doctors",
    ]

    # ========================================================
    # Unsupported Claims
    # ========================================================

    risky_claims = [
        "scientifically-backed",
        "scientifically backed",
        "clinically proven",
        "clinical proof",
        "dermatologist-approved",
        "dermatologist approved",
        "doctor-approved",
        "doctor approved",
        "medically proven",
        "expert-backed",
        "expert backed",
        "guaranteed results",
        "glowing skin",
        "wake up to glowing skin",
    ]

    def contains_unsupported_claim(
        text: str | None
    ) -> bool:

        if not text:
            return False

        text_lower = (
            text.lower()
        )

        return any(
            claim in text_lower
            and claim not in context_lower
            for claim in risky_claims
        )

    # ========================================================
    # Key Message Guardrail
    # ========================================================

    if contains_unsupported_claim(
        response.key_message
    ):

        if (
            "effective skincare that fits into a busy lifestyle"
            in context_lower
        ):

            response.key_message = (
                "Effective skincare that fits into a busy lifestyle."
            )

        elif (
            "morning routine"
            in context_lower
        ):

            response.key_message = (
                "HydraBoost Serum fits into a simple "
                "morning skincare routine."
            )

        else:

            response.key_message = (
                "Simple skincare designed to fit "
                "a busy lifestyle."
            )

    # ========================================================
    # Validate Campaign Ideas
    # ========================================================

    for idea in (
        response.campaign_ideas
    ):

        idea_lower = (
            idea.idea.lower()
        )

        # ----------------------------------------------------
        # Generic tactic validation
        # ----------------------------------------------------

        matched_tactics = [
            tactic
            for tactic in generic_tactics
            if tactic in idea_lower
        ]

        if matched_tactics:

            tactic_supported = any(
                tactic in context_lower
                for tactic in matched_tactics
            )

            if not tactic_supported:

                idea.classification = (
                    "general_recommendation"
                )

                idea.evidence = None

                continue

        # ----------------------------------------------------
        # Risky unsupported idea
        # ----------------------------------------------------

        if contains_unsupported_claim(
            idea.idea
        ):

            idea.classification = (
                "general_recommendation"
            )

            idea.evidence = None

            continue

        # ----------------------------------------------------
        # General recommendation cannot claim evidence
        # ----------------------------------------------------

        if (
            idea.classification
            == "general_recommendation"
        ):

            idea.evidence = None

            continue

        # ----------------------------------------------------
        # Knowledge-based requires evidence
        # ----------------------------------------------------

        if (
            idea.classification
            == "knowledge_based"
            and not idea.evidence
        ):

            idea.classification = (
                "reasonable_inference"
            )

            continue

        # ----------------------------------------------------
        # Evidence grounding
        # ----------------------------------------------------

        if idea.evidence:

            evidence_words = (
                get_evidence_words(
                    idea.evidence
                )
            )

            if evidence_words:

                matching_words = sum(
                    1
                    for word in evidence_words
                    if word in context_lower
                )

                overlap = (
                    matching_words
                    / len(
                        evidence_words
                    )
                )

                if (
                    idea.classification
                    == "knowledge_based"
                    and overlap < 0.40
                ):

                    idea.classification = (
                        "reasonable_inference"
                    )

                elif (
                    idea.classification
                    == "reasonable_inference"
                    and overlap < 0.20
                ):

                    idea.classification = (
                        "general_recommendation"
                    )

                    idea.evidence = None

    # ========================================================
    # Remove Duplicate / Unsafe General Recommendations
    # ========================================================

    cleaned_recommendations = []

    idea_word_sets = [
        normalize_word_set(
            idea.idea
        )
        for idea in (
            response.campaign_ideas
        )
    ]

    for recommendation in (
        response.general_recommendations
    ):

        if not recommendation:
            continue

        recommendation = (
            recommendation.strip()
        )

        if not recommendation:
            continue

        if contains_unsupported_claim(
            recommendation
        ):
            continue

        recommendation_words = (
            normalize_word_set(
                recommendation
            )
        )

        duplicate = False

        for idea_words in (
            idea_word_sets
        ):

            if not recommendation_words:
                continue

            overlap = (
                len(
                    recommendation_words
                    & idea_words
                )
                / len(
                    recommendation_words
                )
            )

            if overlap >= 0.60:

                duplicate = True
                break

        if duplicate:
            continue

        cleaned_recommendations.append(
            recommendation
        )

    # --------------------------------------------------------
    # For this demo, campaign ideas already contain both
    # grounded and general recommendations.
    #
    # Keeping this list empty avoids duplicated content.
    # --------------------------------------------------------

    response.general_recommendations = []

    # ========================================================
    # Rebuild Reasoning AFTER Validation
    # ========================================================

    grounded_ideas = [
        idea.idea
        for idea in response.campaign_ideas
        if (
            idea.classification
            == "knowledge_based"
        )
    ]

    inferred_ideas = [
        idea.idea
        for idea in response.campaign_ideas
        if (
            idea.classification
            == "reasonable_inference"
        )
    ]

    general_ideas = [
        idea.idea
        for idea in response.campaign_ideas
        if (
            idea.classification
            == "general_recommendation"
        )
    ]

    reasoning_parts = []

    if grounded_ideas:

        reasoning_parts.append(
            "The following ideas are directly supported "
            "by retrieved NovaGlow knowledge: "
            + ", ".join(
                grounded_ideas
            )
            + "."
        )

    if inferred_ideas:

        reasoning_parts.append(
            "The following ideas are reasonable inferences "
            "based on retrieved campaign or persona evidence: "
            + ", ".join(
                inferred_ideas
            )
            + "."
        )

    if general_ideas:

        reasoning_parts.append(
            "The following ideas are general marketing "
            "recommendations and are not presented as "
            "NovaGlow historical evidence: "
            + ", ".join(
                general_ideas
            )
            + "."
        )

    if reasoning_parts:

        response.reasoning = (
            " ".join(
                reasoning_parts
            )
        )

    # ========================================================
    # Confidence Consistency
    # ========================================================

    classifications = {
        idea.classification
        for idea in (
            response.campaign_ideas
        )
    }

    if (
        "reasonable_inference"
        in classifications
        or "general_recommendation"
        in classifications
    ):

        response.confidence = "Medium"

    return response


# ============================================================
# Segment Grounding Validation
# ============================================================

def validate_segment_grounding(
    response: SegmentResponse,
    context: str,
):
    """
    Repairs common small-model structured-output issues using
    the retrieved persona context.

    Examples:
    - "age" -> "Age: 25-34"
    - "occupation" -> actual occupation value
    - restores missing LinkedIn
    - restores missing supported pain points
    """

    context_lower = (
        context.lower()
    )

    cleaned_characteristics = []
    recovered_pain_points = []

    # ========================================================
    # Extract Known Context Values
    # ========================================================

    age_match = re.search(
        r"\bAge\s*:\s*"
        r"([0-9]{1,2}\s*[–-]\s*[0-9]{1,2})",
        context,
        flags=re.IGNORECASE,
    )

    occupation_match = re.search(
        r"\bOccupation\s*:\s*([^\n]+)",
        context,
        flags=re.IGNORECASE,
    )

    age_value = None
    occupation_value = None

    if age_match:

        age_value = (
            age_match
            .group(1)
            .strip()
        )

    if occupation_match:

        occupation_value = (
            occupation_match
            .group(1)
            .strip()
        )

    # ========================================================
    # Repair Characteristics
    # ========================================================

    for item in (
        response.characteristics
    ):

        item_clean = (
            item.strip()
        )

        item_lower = (
            item_clean.lower()
        )

        # ----------------------------------------------------
        # Model returned only "age"
        # ----------------------------------------------------

        if item_lower in {
            "age",
            "age:",
        }:

            if age_value:

                cleaned_characteristics.append(
                    f"Age: {age_value}"
                )

            continue

        # ----------------------------------------------------
        # Model returned only "occupation"
        # ----------------------------------------------------

        if item_lower in {
            "occupation",
            "occupation:",
        }:

            if occupation_value:

                cleaned_characteristics.append(
                    f"Occupation: {occupation_value}"
                )

            continue

        # ----------------------------------------------------
        # Pain points accidentally placed in characteristics
        # ----------------------------------------------------

        if item_lower.startswith(
            "pain points:"
        ):

            pain_text = (
                item_clean
                .split(
                    ":",
                    1,
                )[1]
                .strip()
            )

            parts = [
                part.strip()
                for part in (
                    pain_text.split(",")
                )
                if part.strip()
            ]

            recovered_pain_points.extend(
                parts
            )

            continue

        cleaned_characteristics.append(
            item_clean
        )

    # ========================================================
    # Ensure Age Exists
    # ========================================================

    has_age = any(
        item.lower().startswith(
            "age:"
        )
        for item in (
            cleaned_characteristics
        )
    )

    if (
        not has_age
        and age_value
    ):

        cleaned_characteristics.insert(
            0,
            f"Age: {age_value}",
        )

    # ========================================================
    # Ensure Occupation Exists
    # ========================================================

    has_occupation = any(
        item.lower().startswith(
            "occupation:"
        )
        for item in (
            cleaned_characteristics
        )
    )

    if (
        not has_occupation
        and occupation_value
    ):

        cleaned_characteristics.append(
            f"Occupation: {occupation_value}"
        )

    response.characteristics = (
        dedupe_text_list(
            cleaned_characteristics
        )
    )

    # ========================================================
    # Pain Points
    # ========================================================

    pain_points = (
        list(
            response.pain_points
        )
        + recovered_pain_points
    )

    known_pain_points = [
        "Limited time for complicated skincare routines",
        "Stress and long working hours",
        "Wants simple skincare products",
        "Prefers products with clear benefits",
    ]

    for pain_point in (
        known_pain_points
    ):

        if (
            pain_point.lower()
            in context_lower
        ):

            pain_points.append(
                pain_point
            )

    response.pain_points = (
        dedupe_text_list(
            pain_points
        )
    )

    # ========================================================
    # Digital Behavior
    # ========================================================

    digital_behavior = list(
        response.digital_behavior
    )

    known_channels = [
        "Instagram",
        "TikTok",
        "YouTube",
        "LinkedIn",
    ]

    for channel in known_channels:

        if (
            channel.lower()
            in context_lower
        ):

            digital_behavior.append(
                channel
            )

    response.digital_behavior = (
        dedupe_text_list(
            digital_behavior
        )
    )

    # ========================================================
    # Buying Motivation
    # ========================================================

    buying_motivation = list(
        response.buying_motivation
    )

    known_motivation = (
        "Convenience, quality and trustworthy product information"
    )

    if (
        known_motivation.lower()
        in context_lower
    ):

        buying_motivation.append(
            known_motivation
        )

    response.buying_motivation = (
        dedupe_text_list(
            buying_motivation
        )
    )

    # ========================================================
    # Marketing Recommendations
    # ========================================================

    response.marketing_recommendations = (
        dedupe_text_list(
            response.marketing_recommendations
        )
    )

    # Marketing recommendations contain inference,
    # so Medium is safer than High.
    if response.marketing_recommendations:

        response.confidence = "Medium"

    return response


# ============================================================
# Render Fact
# ============================================================

def render_fact(
    response: FactResponse,
):

    return f"""
### Answer

{response.answer}
""".strip()


# ============================================================
# Render Campaign
# ============================================================

def render_campaign(
    response: CampaignResponse,
):

    lines = [
        (
            "### Campaign Name\n"
            f"{response.campaign_name}"
        ),
        (
            "### Objective\n"
            f"{response.objective}"
        ),
        (
            "### Target Audience\n"
            f"{response.target_audience}"
        ),
        (
            "### Key Message\n"
            f"{response.key_message}"
        ),
        "### Recommended Channels",
    ]

    for channel in (
        response.recommended_channels
    ):

        lines.append(
            f"- {channel}"
        )

    lines.append(
        "\n### Campaign Ideas"
    )

    for index, idea in enumerate(
        response.campaign_ideas,
        start=1,
    ):

        classification = (
            idea.classification
            .replace(
                "_",
                " ",
            )
            .title()
        )

        lines.append(
            f"\n**{index}. "
            f"{idea.idea}**"
        )

        lines.append(
            "- Classification: "
            f"`{classification}`"
        )

        if idea.evidence:

            lines.append(
                "- Evidence: "
                f"{idea.evidence}"
            )

    lines.append(
        "\n### Reasoning\n"
        f"{response.reasoning}"
    )

    if (
        response.general_recommendations
    ):

        lines.append(
            "\n### General Recommendations"
        )

        for recommendation in (
            response.general_recommendations
        ):

            lines.append(
                f"- {recommendation}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# Render Segment
# ============================================================

def render_segment(
    response: SegmentResponse,
):

    lines = [
        (
            "### Segment\n"
            f"{response.segment_name}"
        ),
        "\n### Characteristics",
    ]

    for item in (
        response.characteristics
    ):

        lines.append(
            f"- {item}"
        )

    lines.append(
        "\n### Pain Points"
    )

    for item in (
        response.pain_points
    ):

        lines.append(
            f"- {item}"
        )

    lines.append(
        "\n### Digital Behavior"
    )

    for item in (
        response.digital_behavior
    ):

        lines.append(
            f"- {item}"
        )

    lines.append(
        "\n### Buying Motivation"
    )

    for item in (
        response.buying_motivation
    ):

        lines.append(
            f"- {item}"
        )

    lines.append(
        "\n### Marketing Recommendations"
    )

    for item in (
        response.marketing_recommendations
    ):

        lines.append(
            f"- {item}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# Main Agent
# ============================================================

def ask_agent(
    user_query: str,
):

    # ========================================================
    # Intent Routing
    # ========================================================

    intent = detect_intent(
        user_query
    )

    print(
        "\nDetected Intent: "
        f"{intent.value}"
    )

    # ========================================================
    # RAG Retrieval
    # ========================================================

    print(
        "Retrieving relevant knowledge..."
    )

    rag_result = retrieve_context(
        query=user_query,
        k=4,
    )

    context = (
        rag_result[
            "context"
        ]
    )

    sources = (
        rag_result[
            "sources"
        ]
    )

    documents = (
        rag_result[
            "documents"
        ]
    )

    print(
        "Generating structured response..."
    )

    # ========================================================
    # FACT LOOKUP
    # ========================================================

    if (
        intent
        == IntentType.FACT_LOOKUP
    ):

        deterministic_result = (
            deterministic_fact_lookup(
                user_query=user_query,
                documents=documents,
            )
        )

        # ----------------------------------------------------
        # Exact Deterministic Match
        # ----------------------------------------------------

        if (
            deterministic_result
            and deterministic_result.get(
                "status"
            ) == "found"
        ):

            structured_response = (
                FactResponse(
                    answer=(
                        deterministic_result[
                            "answer"
                        ]
                    ),
                    confidence="High",
                    sources=(
                        deterministic_result[
                            "sources"
                        ]
                    ),
                )
            )

            sources = (
                deterministic_result[
                    "sources"
                ]
            )

        # ----------------------------------------------------
        # Unsupported / Ambiguous Numeric Fact
        # ----------------------------------------------------

        elif (
            deterministic_result
            and deterministic_result.get(
                "status"
            )
            in {
                "unsupported",
                "ambiguous",
            }
        ):

            structured_response = (
                FactResponse(
                    answer=(
                        "The available knowledge base "
                        "does not contain this information."
                    ),
                    confidence="Low",
                    sources=sources,
                )
            )

        # ----------------------------------------------------
        # General Fact via LLM
        # ----------------------------------------------------

        else:

            prompt = build_fact_prompt(
                user_query=user_query,
                context=context,
                sources=sources,
            )

            structured_response = (
                invoke_structured(
                    FactResponse,
                    prompt,
                )
            )

            # Only use real retrieval sources.
            structured_response.sources = (
                sources
            )

        answer = render_fact(
            structured_response
        )

    # ========================================================
    # CAMPAIGN GENERATION
    # ========================================================

    elif (
        intent
        == IntentType.CAMPAIGN_GENERATION
    ):

        prompt = build_campaign_prompt(
            user_query=user_query,
            context=context,
            sources=sources,
        )

        structured_response = (
            invoke_structured(
                CampaignResponse,
                prompt,
            )
        )

        # Never trust model-generated source names.
        structured_response.sources = (
            sources
        )

        structured_response = (
            validate_campaign_grounding(
                structured_response,
                context,
            )
        )

        answer = render_campaign(
            structured_response
        )

    # ========================================================
    # SEGMENT ANALYSIS
    # ========================================================

    else:

        prompt = build_segment_prompt(
            user_query=user_query,
            context=context,
            sources=sources,
        )

        structured_response = (
            invoke_structured(
                SegmentResponse,
                prompt,
            )
        )

        structured_response.sources = (
            sources
        )

        structured_response = (
            validate_segment_grounding(
                structured_response,
                context,
            )
        )

        answer = render_segment(
            structured_response
        )

    # ========================================================
    # Final Response
    # ========================================================

    return {
        "intent": intent.value,
        "answer": answer,
        "sources": sources,
        "structured": (
            structured_response
            .model_dump()
        ),
    }


# ============================================================
# CLI Demo
# ============================================================

def main():

    print(
        "\n================================"
    )

    print(
        "        MarketMind AI"
    )

    print(
        "================================"
    )

    print(
        f"\nModel: {OLLAMA_MODEL}"
    )

    print(
        "RAG + Guardrails + Structured Output"
    )

    while True:

        user_query = input(
            "\nAsk MarketMind "
            "(type 'exit' to quit):\n> "
        )

        if (
            user_query
            .lower()
            .strip()
            in {
                "exit",
                "quit",
            }
        ):

            print(
                "\nGoodbye!"
            )

            break

        if not user_query.strip():

            continue

        try:

            result = ask_agent(
                user_query
            )

            print(
                "\n============================"
            )

            print(
                f"Intent: {result['intent']}"
            )

            print(
                "============================\n"
            )

            print(
                result["answer"]
            )

            print(
                "\n============================"
            )

            print(
                "Knowledge Sources"
            )

            print(
                "============================"
            )

            for source in (
                result["sources"]
            ):

                print(
                    f"- {source}"
                )

            print(
                "\n============================"
            )

            print(
                "Structured Object"
            )

            print(
                "============================"
            )

            print(
                result["structured"]
            )

        except Exception as error:

            print(
                "\nError: "
                f"{error}"
            )


# ============================================================
# Run CLI
# ============================================================

if __name__ == "__main__":

    main()