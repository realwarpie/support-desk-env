# SupportDeskEnv

An OpenEnv-compatible, deterministic customer support simulation for benchmarking LLM agents on realistic multi-step service workflows.

## Motivation

LLM agents are easy to test on static QA tasks and surprisingly hard to evaluate in real workflows where decisions unfold across multiple turns.

Customer support is a high-value domain for this evaluation challenge because it combines:

- language understanding under ambiguity
- policy-compliant decision making
- sequencing of operational actions
- recovery from partial information and user frustration

SupportDeskEnv matters because it moves evaluation from one-shot prompt quality to end-to-end operational behavior: classification quality, policy compliance, resolution strategy, and efficiency under constraints.

## What This Environment Does

SupportDeskEnv exposes an OpenEnv-style interaction loop:

- `reset(task_id=None) -> Observation`
- `step(action) -> observation, reward, done, info`
- `state() -> full internal state`

An agent operates inside a simulated inbox where each step updates:

- ticket lifecycle statei 
- conversation history
- resolution stage
- reward components and progress signals

The environment models a realistic support flow from triage to resolution using structured actions and deterministic policy logic.

## Key Features

- Dynamic customer simulation with context-aware follow-ups
- Deterministic behavior for reproducible evaluation runs
- Multi-step task workflows with increasing complexity
- Dense reward shaping across the full episode (not terminal-only)
- Strict deterministic graders with 0.0-1.0 scoring
- Robust malformed/invalid action handling with safe penalties

## Action Space

The agent acts through a structured `Action` model:

- `tag_ticket(category, priority)`
  - Classifies the issue and sets urgency for downstream handling.
- `draft_reply(text)`
  - Sends a customer-facing response, expected to be actionable and policy-compliant.
- `request_info(field)`
  - Requests missing evidence or details needed to proceed (for example, damage proof).
- `escalate(team)`
  - Routes complex or high-risk cases to a specialist team.
- `close_ticket(reason)`
  - Closes the case with an explicit resolution status or waiting reason.

## Observation Space

Each step returns an `Observation` including:

- current ticket context
- full conversation history (customer, agent, system turns)
- policy snippets for decision guidance
- order metadata (order_id, status, price, delivery date)
- available action signatures
- step counters (`steps_taken`, `max_steps`)

## Task Design

SupportDeskEnv ships with three deterministic tasks with clear difficulty progression:

1. Easy: classification + prioritization
   - Agent must correctly categorize and prioritize a ticket.
2. Medium: policy-compliant response
   - Agent must classify, respond with proper policy language, and close with correct reason.
3. Hard: multi-step resolution
   - Agent must classify, request missing information, escalate appropriately, respond clearly, and close with correct waiting semantics.

Difficulty increases through longer dependency chains, stricter sequencing, and stronger policy requirements.

## Reward Design

Reward is dense and evolves every step, not just at terminal states.

Signals include:

- progress-based credit from improved task score
- intermediate milestone rewards for correct action order
- penalties for loops and repeated patterns
- penalties for redundant or invalid actions
- efficiency bonus for concise successful trajectories
- independent response-quality signals:
  - reply informativeness
  - keyword coverage
  - low-information reply penalties
  - action diversity incentives

All per-step rewards are bounded in `[-1.0, 1.0]`.

## Grading System

Graders are deterministic and strict, returning scores in `[0.0, 1.0]`.

They evaluate:

- correctness of classification and priority
- correctness of action sequence
- policy compliance in response text
- final outcome correctness (including close reasons)
- penalties for duplicate or unnecessary actions

This provides stable, reproducible comparisons across models and prompting strategies.

## Dynamic Environment

SupportDeskEnv is intentionally non-static.

The environment state evolves based on agent decisions:

- customer follow-ups depend on ticket status, resolution stage, and prior actions
- repeated requests or repeated responses increase frustration signals
- escalation and closure actions update lifecycle state
- ticket state transitions through statuses such as `open`, `waiting_for_customer`, `escalated`, `resolved`, and `closed`

This creates a realistic interaction loop closer to production support operations than scripted single-turn benchmarks.

## How to Run Locally

Install dependencies and run baseline inference:

```bash
pip install -r requirements.txt
python inference.py
```

Run the environment server:

```bash
uvicorn env:app --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## Environment Variables

`inference.py` supports OpenAI-compatible endpoints via:

- `API_BASE_URL`
  - Base URL for OpenAI-compatible chat completion API.
- `MODEL_NAME`
  - Model identifier used for inference (default: `gpt-4o-mini`).
- `HF_TOKEN`
  - API token used as client key for hosted providers.

Optional controls:

- `DEBUG` to enable debug logs
- `SEED` for deterministic run config
- `MAX_EPISODE_STEPS` for episode cap
- `MODEL_TIMEOUT_SECONDS` for request timeout

## Docker Usage

```bash
docker build -t supportdesk-env .
docker run -p 8000:8000 supportdesk-env
```

## Example Output

Example `python inference.py` output:

```json
{
  "task_id": "hard_resolution_001",
  "difficulty": "hard",
  "steps": 5,
  "max_steps_used": 8,
  "cumulative_reward": 0.8421,
  "mean_step_reward": 0.1684,
  "grader_score": 1.0,
  "final_outcome_quality": 1.0,
  "fallback_action_count": 0,
  "terminated_by_cap": false,
  "termination": "completed"
}
```

## Why This Stands Out

SupportDeskEnv stands out as an agent benchmark because it combines realism, determinism, and rigorous scoring in one environment:

- Realism: dynamic, stateful support interactions with evolving customer behavior
- Robustness: safe handling of malformed actions and loop control
- Evaluation quality: strict grading plus dense reward decomposition
- Benchmark utility: reproducible, policy-grounded, multi-step workflows that expose strengths and failure modes of LLM agents

For hackathon judging, this means stronger evidence of true agent capability than static prompt tests alone.
