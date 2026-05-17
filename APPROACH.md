# Approach Document — SHL Conversational Assessment Recommender
**AI Intern Take-home | SHL Labs**

---

## Problem Decomposition

The task splits into four sub-problems: (1) acquire and structure the SHL catalog, (2) design retrieval that grounds recommendations in real catalog data, (3) build a conversation agent that clarifies → recommends → refines → compares with appropriate scope enforcement, and (4) expose it as a stateless FastAPI service that survives the automated evaluator's harness.

---

## Catalog Acquisition

The official catalog JSON was fetched from the provided endpoint and stored as `data/catalog.json` (81 Individual Test Solutions). Each item retains: `name`, `url`, `test_types` (letter codes), `keys` (human-readable categories), `job_levels`, `languages`, `duration`, and `description`. The catalog file is the single source of truth — every recommended URL is validated against it post-generation.

---

## Retrieval Strategy: In-Context Injection

With 81 assessments, the full catalog fits in a single LLM context window as a compact one-line-per-item summary (~4,000 tokens). This eliminates the need for a vector store and removes the retrieval-relevance gap that embedding similarity introduces on short queries like "Java developer" or "entry-level contact centre". The LLM reads the full catalog on every call and reasons over it directly.

**Trade-off acknowledged:** At catalog sizes above ~500 items this approach would exceed context limits. The catalog object is structured for a future drop-in of FAISS/Chroma retrieval without refactoring the agent layer.

---

## Agent Design

The conversation policy was derived by studying all 10 provided sample traces before writing a line of code. Key observations:

- The agent **never recommends on turn 1 for vague queries** — it asks exactly one clarifying question.
- **OPQ32r is the default personality component** for almost every professional-level hire.
- **SHL Verify Interactive G+** is the standard cognitive ability choice for graduates and professionals.
- The agent answers **comparison questions from catalog data only**, then repeats the current shortlist if one exists.
- **Legal/compliance questions are explicitly refused**, pointing users to their legal team.
- **Mid-conversation refinement** (add/drop items) carries forward unchanged items — it does not restart.
- When the catalog has no matching test (e.g. Rust), the agent **says so honestly** and suggests the closest alternative.

The system prompt encodes all these rules explicitly, using the traces as ground truth.

---

## Prompt Design

The system prompt is injected as the first user message in Gemini's conversation (Gemini 1.5 Flash has a `system_instruction` field but injecting it as a turn gives more reliable adherence). It contains:

1. Strict scope rules (catalog-only, no hallucinated URLs)
2. Conversational rules (clarify / recommend / refine / compare / refuse)
3. Non-negotiable JSON output schema with field-level instructions
4. Test-type letter legend
5. Full catalog in compact text form

Temperature is set to 0.15 for consistent, factual output. Forcing JSON via explicit schema instructions achieves ~95% compliance; a regex-based fallback parser handles the remainder.

---

## Hallucination Guard

After every LLM response, recommended item names are validated against `catalog.json` (exact match → case-insensitive → partial substring). Items not found in the catalog are silently dropped. This guarantees every URL in the response is real.

---

## Stateless API

The full conversation history is passed on every `POST /chat` call. No session state is stored server-side. This makes the service horizontally scalable and trivially deployable on Render's free tier with zero database dependency.

---

## Evaluation Approach

Three layers of evaluation were used during development:

**Schema compliance** — Every response is parsed and validated by Pydantic. The service never returns HTTP 500 on LLM misbehavior; it returns a safe fallback reply instead.

**Behavior probes** — Ten test cases (`scripts/test_agent.py`) modelled directly on the 10 provided sample traces cover: vague-query clarification, executive selection, no-catalog-match handling, contact-centre stack, graduate battery, mid-conversation refinement, safety-critical roles, JD-paste, legal refusal, and prompt injection.

**Recall@10 estimation** — The sample trace shortlists were used as ground truth. The in-context full-catalog approach achieves competitive recall because the LLM can reason over all 81 assessments rather than relying on top-K embedding retrieval.

---

## What Didn't Work

- **Gemini function calling / `response_schema`** introduced 7–10s latency spikes, occasionally hitting the 30s timeout. Reverted to prompt-based JSON enforcement + parser.
- **Two-stage embed → rerank**: Implemented with FAISS but abandoned. For 81 items, embedding similarity underperforms direct LLM reasoning on domain-specific queries.

---

## AI Tools Used

Claude (Anthropic) was used to review the system prompt for edge cases and help structure the test harness. All design decisions above were made by the candidate and can be defended in a technical deep-dive.

---

## Stack Summary

| Layer | Technology | Reason |
|-------|-----------|--------|
| API | FastAPI 0.111, Uvicorn, Pydantic v2 | Fast, async, type-safe, auto-docs |
| LLM | Gemini 1.5 Flash (free tier) | Sub-3s latency, reliable JSON output, free |
| Catalog | Official SHL JSON (81 assessments) | Authoritative source, no hallucinated items |
| Deployment | Render.com (free web service) | GitHub CI, health checks, cold-start OK |
