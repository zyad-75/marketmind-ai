# 🧠 MarketMind AI

**MarketMind AI** is a domain-specific AI marketing assistant built for the NationAI technical assessment.

The project demonstrates how a local Large Language Model can be combined with:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Intent routing
- Structured outputs
- Prompt engineering
- Guardrails
- Hallucination control
- Deterministic fact extraction
- Evaluation
- A conversational UI

The current implementation is adapted to the **Marketing** domain using a fictional skincare company called **NovaGlow**.

---

# 🎯 Problem Statement

Marketing teams often need to work with multiple internal information sources such as:

- Brand guidelines
- Customer personas
- Product information
- Previous campaign reports
- Campaign performance metrics

A generic LLM may generate useful ideas, but it can also:

- Invent company-specific facts
- Hallucinate campaign metrics
- Mix information from different products
- Produce recommendations that are not grounded in company knowledge

MarketMind AI addresses this by grounding responses in an internal knowledge base and explicitly separating:

- Knowledge-based facts
- Reasonable inference
- General recommendations

---

# ✨ Main Capabilities

MarketMind supports three main request types.

## 🔎 Fact Lookup

Example:

```text
What was the CTR of the Morning Made Simple campaign?