# Financial Advisor Agent

A small agentic chatbot that turns a salary figure into a defensible monthly money plan.
Tell it `"my salary is 1.2 lakhs, I'm 27, no loans"` and it researches today's real
interest rates, computes tax and allocation in plain Python, and explains the result.

The interesting part is not that an LLM gives financial advice. It is the set of
constraints that stop it from making numbers up.

## The design problem

An LLM asked "how should I invest 1.2 lakhs a month?" will confidently produce
a PPF rate, an FD rate and a compounded projection. All three are likely to be
wrong: the rates come from training data that is months or years stale, and the
compounding is mental arithmetic. Financial advice is exactly the domain where a
plausible-sounding wrong number does real damage.

So this project splits the work by what each component is actually good at:

| Job | Who does it | Why |
| --- | --- | --- |
| Fetch search results | Plain Python, in parallel | Deterministic plumbing, no judgement needed |
| Read rates out of those results | CrewAI extractor agent | Needs judgement about what a snippet means |
| Audit the extraction | CrewAI verifier agent | Needs an incentive opposite to the extractor's |
| Enforce the audit | Python validation | Needs to be incorruptible, not persuadable |
| Do the maths | Pure Python | Needs to be exact and reproducible |
| Explain the plan | Gemini via LangChain | Needs fluent natural language |
| Decide the flow | LangGraph | Needs retries and branching, not a straight line |

**The LLM never does arithmetic and never sources a number from memory.**

## Architecture

```
                     ┌──────────────────┐
    user message ───▶│ extract_profile  │  Gemini structured output -> ProfileExtraction
                     └────────┬─────────┘  (reports stated facts only; Python applies defaults)
                              │
               ┌──────────────┴──────────────┐
      new facts│                             │follow-up question about an existing plan
               ▼                             │
        ┌─────────────┐                      │
        │  research   │◀───┐                 │   1. Serper: 6 searches, parallel, in Python
        └──────┬──────┘    │                 │   2. CrewAI extractor  -> structured rates
               │           │                 │   3. CrewAI verifier   -> deletes the unsupported
               ▼           │ retry only      │   4. Python validation -> bounds + citation check
         (validation)──────┘ the failed keys │
               ▼                             │
        ┌─────────────┐                      │
        │   compute   │  pure functions: tax, waterfall, glidepath, SIP maths
        └──────┬──────┘                      │
               ▼                             ▼
        ┌─────────────────────────────────────┐
        │               advise                │  Gemini + vanilla RAG;
        └──────────────────┬──────────────────┘  numbers injected, never generated
                           ▼
                          END
```

The retry edge is not decorative. A real run:

```
. Research pass 1: 4/6 rates verified against a cited source.
. Research pass 2: 6/6 rates verified against a cited source.
```

Two rates failed the citation check on the first pass, were re-researched on the
second, and passed. Had they failed again, they would have become labelled
assumptions rather than silent guesses.

Every rate in `config.RATES` is consumed by the allocation engine, and a test
enforces that. Rates that were merely interesting — EPF, 1-year FD, NPS returns —
were deleted: each cost a web search and prompt space for nothing. EPF is a good
example of why: it is deducted before salary is credited, so it is never allocated
out of take-home pay.

### Five layers that block hallucination

1. **Structured output.** The crew returns Pydantic models, never prose. A rate
   without a `source_url` and an `evidence` field cannot be constructed.
2. **Adversarial verifier agent.** A second CrewAI agent is instructed to *delete*
   findings, and rewarded for removing unsupported ones. Its incentive is
   deliberately opposite to the researcher's.
3. **Citation checking in Python.** `_number_supported_by_evidence` parses every
   number out of the quoted snippet and requires a near-exact match to the claimed
   value. This is the important one: it does not trust the verifier agent's word.
   A plausible but uncited `7.4%` is rejected just as hard as an absurd `45%`.
4. **Plausibility bounds.** Each rate carries a wide band in `config.RATES`.
   Wide enough to allow real market movement, narrow enough to catch nonsense.
5. **Labelled fallbacks.** When a rate fails validation, a documented static
   assumption is used and tagged `[fallback]` with the rejection reason. The user
   is told. The system degrades honestly instead of silently inventing.

Statutory constants live in `data/tax_rules.json` with the financial year and
source URL attached, and every output prints which ruleset produced it. Tax law is
versioned data, not model output.

## What it actually computes

**Tax** — India's new regime: progressive slabs, standard deduction, section 87A
rebate *including marginal relief* (the reason ₹12.1L of taxable income owes
₹10,400 and not ₹63,960), surcharge and cess.

**Allocation** — a priority waterfall rather than a flat percentage split, because
order matters more than ratios:

```
essentials -> EMIs -> insurance -> emergency fund -> growth investing
```

An equity SIP started before an emergency fund exists usually gets redeemed at a
loss in the first crisis, so liquidity is funded first — but capped at half the
surplus, so investing still begins today.

**Asset mix** — a `100 - age` glidepath clamped by risk band, then split across
index / flexi-cap / mid-small equity, PPF and debt funds, and gold.

**Projections** — SIP compounding with start-of-month contributions, shown both
nominal and inflation-adjusted, because a corpus quoted in 2050 rupees is
misleading.

## Setup

Requires **Python 3.10–3.13** (CrewAI does not support 3.14 yet).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # then add your two keys
```

Both keys have free tiers:

- `GEMINI_API_KEY` — [Google AI Studio](https://aistudio.google.com/apikey)
- `SERPER_API_KEY` — [serper.dev](https://serper.dev), 2500 free searches

A note on the Gemini free tier, since it shapes the design: quota is enforced
*per model*, and `gemini-3.6-flash` allows as few as 5 requests per minute. A full
turn therefore costs only two LLM calls (extraction and explanation), plus two
more when rates are actually researched. Rate-limit errors are retried with
backoff, honouring the server's own `retry in Ns` hint, and a flash-lite model can
be set in `.env` for more headroom.

Model names also go stale: this project originally targeted `gemini-2.5-flash` and
got a 404 saying it "is no longer available to new users", so both the model and
the embedding model are configurable.

## Usage

```bash
python main.py                                    # interactive chat
python main.py "my salary is 1.2 lakhs, I'm 27"   # one-shot
```

In-chat commands:

| Command | Effect |
| --- | --- |
| `/sources` | Every rate with its URL and the exact sentence it was taken from |
| `/profile` | The profile currently inferred from the conversation |
| `/reset` | Clear the conversation |
| `/quit` | Exit |

`/sources` is worth running — it is the audit trail that makes the numbers
checkable, and the fastest way to demo the point of the project.

Researched rates are cached in `.cache/` for 24 hours, so follow-up questions are
instant and cost no search quota. Follow-ups that add no new facts skip the
research and compute nodes entirely.

## Testing

```bash
pytest -q     # 66 tests, fully offline, no API key needed
```

`tests/test_finance.py` pins the tax arithmetic against hand-computed values and
asserts allocation invariants (buckets sum to salary, asset weights sum to 100%,
PPF respects its statutory cap, a negative surplus refuses to invest).

`tests/test_guardrails.py` encodes the central claim, including the case that
matters most: a *plausible* rate with mismatched evidence is still rejected.

`tests/test_pipeline.py` stubs the LLM to test the graph itself — routing, the
retry cycle, rate-limit backoff, and the assertion that every number the advisor
is told to quote is actually present in the FACTS block it receives.

## Project structure

```
main.py                 chat loop and terminal rendering
advisor_graph.py        LangGraph state machine, nodes, prompts, retry/backoff
research.py             parallel Serper fetch, CrewAI crew, rate validation
finance.py              deterministic engine: tax, allocation, compounding (no LLM)
knowledge.py            vanilla RAG: chunk, embed, cache, cosine search
schemas.py              Pydantic contracts shared across layers
config.py               env, model names, rate bounds and fallbacks
data/
  knowledge.md          curated personal-finance corpus (the RAG chunks)
  tax_rules.json        versioned statutory constants with source
tests/
  test_finance.py       tax and allocation maths
  test_guardrails.py    the anti-hallucination validation
  test_pipeline.py      graph routing, retries, prompt assembly
```

## Interview notes

**Why both CrewAI and LangGraph?** They solve different problems and the split is
load-bearing. CrewAI models *roles* — a researcher and an adversarial verifier
with opposing incentives, which is a natural multi-agent framing. LangGraph models
*control flow* — the cycle that retries failed rates and the branch that skips
research for follow-up questions. Using CrewAI for the outer flow would mean
hand-rolling retry logic; using LangGraph for the crew would mean rebuilding role
prompting and tool loops.

**Why isn't the web search done by an agent with a tool?** It was at first, and it
was the wrong call. Retrieving a fixed list of known queries is plumbing, not
judgement: letting the agent drive it turned into many sequential LLM round trips,
which was slow, non-reproducible, and died instantly against a 5-request-per-minute
free tier. Moving search into a parallel Python pre-fetch cut research to two LLM
calls while leaving the agents doing the part that genuinely needs a model —
interpreting snippets and auditing each other. The Serper tool is still registered
on the extractor as an escape hatch when the pre-built evidence pack falls short.
The general lesson: give agents the decisions, not the I/O.

**Why is the RAG "vanilla"?** The corpus is a few dozen chunks. An exact numpy
cosine search over that is faster than a vector database and has no index to keep
in sync. Embeddings are cached against a corpus hash, so they are recomputed only
when the corpus actually changes. Reaching for Chroma or FAISS here would be
resume-driven engineering — the honest answer is that it starts to pay off at
roughly 10k+ chunks, where exact search stops fitting comfortably in memory.

**Why two knowledge sources?** Serper supplies live *numbers*; RAG supplies
durable *principles*. Mixing them would let a stale document contradict a fresh
rate. Keeping them separate means the corpus deliberately contains no rates.

**Where would this break?** Four honest limits.

Serper returns snippets, not full pages, so a rate buried below the fold is
invisible; the fix is fetching and parsing the top result.

The citation check proves *attribution*, not *correctness*. A number scraped from
a garbled HTML table can still satisfy it — one real run cited the NPS 5-year
return from the snippet `"... NAV 1 Year 3 Years 5 Years … 14.51 -0.19% 10.0%"`,
where the digits line up but the column mapping is guesswork. Nor does it prove
freshness: a correctly cited stale page passes cleanly. Requiring the verifier to
check publication recency and reject table fragments would tighten this.

`100 - age` is a crude heuristic that ignores goal horizon, which is what should
really drive allocation.

Finally, the advisor is instructed not to compute, but nothing structurally
prevents it from emitting a wrong number in prose. Post-validating the generated
text — extracting every figure and checking it appears in the FACTS block — would
close the loop, and is the change I would make next.

## Disclaimer

Educational project. Not investment advice, and not from a SEBI-registered
adviser. Rates are scraped from public web sources and may be stale or
misattributed. Verify anything before acting on it.
