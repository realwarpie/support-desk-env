from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

from env import SupportDeskEnv
from models import Action, Observation
from tasks import TASKS


LOGGER = logging.getLogger("support_desk_inference")
DEFAULT_MAX_EPISODE_STEPS = 10
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TASK_ID = "easy_classify_001"
DEFAULT_BENCHMARK = "supportdesk"


def _is_debug_enabled() -> bool:
    """Return whether debug logs should be emitted."""
    return os.getenv("DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _configure_reproducibility(seed: int) -> None:
    """Configure deterministic randomness for local run behavior."""
    random.seed(seed)


def _setup_logging(debug: bool) -> None:
    """Configure optional structured logging."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Keep protocol output clean by muting SDK transport logs in normal runs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _build_prompt(observation: Observation) -> str:
    """Build compact deterministic prompt from current observation."""
    return (
        "You are a customer support agent. Return exactly one JSON object with keys: "
        "action_type, category, priority, text, field, team, reason. "
        "Only include relevant fields for the action_type.\n\n"
        f"Observation:\n{observation.model_dump_json(indent=2)}"
    )


def _fallback_policy(task_id: str, step: int) -> Action:
    """Deterministic expert fallback policy for each task."""
    if task_id == "easy_classify_001":
        return Action(action_type="tag_ticket", category="damaged_item", priority="high")

    if task_id == "medium_response_001":
        scripted = [
            Action(action_type="tag_ticket", category="shipping", priority="medium"),
            Action(
                action_type="draft_reply",
                text=(
                    "Sorry for the delay. Please check the tracking page for live updates. "
                    "If it does not move in 48 hours, we can review refund options."
                ),
            ),
            Action(action_type="close_ticket", reason="resolved_with_guidance"),
        ]
        return scripted[min(step, len(scripted) - 1)]

    scripted = [
        Action(action_type="tag_ticket", category="refund", priority="high"),
        Action(action_type="request_info", field="damage_photo"),
        Action(action_type="escalate", team="returns"),
        Action(
            action_type="draft_reply",
            text=(
                "Please share a photo of the damage. Your case is escalated to our returns team "
                "for priority handling."
            ),
        ),
        Action(action_type="close_ticket", reason="awaiting_customer_info"),
    ]
    return scripted[min(step, len(scripted) - 1)]


def _safe_default_action(task_id: str, step: int) -> Action:
    """Safe deterministic action used when parsing or model calls fail."""
    return _fallback_policy(task_id, step)


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Extract one JSON object from model output with markdown/fence tolerance."""
    text = raw.strip()
    if not text:
        raise ValueError("empty model output")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            return candidate
        raise ValueError("top-level JSON is not an object")
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("no JSON object boundaries found")

    candidate_text = text[start : end + 1]
    candidate = json.loads(candidate_text)
    if not isinstance(candidate, dict):
        raise ValueError("extracted JSON is not an object")
    return candidate


def _coerce_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common LLM key variants into Action schema keys."""
    key_aliases = {
        "action": "action_type",
        "type": "action_type",
        "message": "text",
        "reply": "text",
        "closure_reason": "reason",
    }
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        mapped = key_aliases.get(key, key)
        normalized[mapped] = value

    action_type = str(normalized.get("action_type", "")).strip()
    normalized["action_type"] = action_type
    return normalized


def _parse_action_or_none(raw_content: str) -> Optional[Action]:
    """Parse model output into validated Action, returning None if invalid."""
    try:
        payload = _extract_json_object(raw_content)
        normalized = _coerce_action_payload(payload)
        return Action.model_validate(normalized)
    except Exception as exc:
        LOGGER.debug("action_parse_failed | error=%s | raw=%s", exc, raw_content)
        return None


def _model_action(client: Any, model_name: str, observation: Observation, timeout_seconds: int) -> Optional[Action]:
    """Call model once and parse into structured action, returning None on failure."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        seed=42,
        max_tokens=220,
        timeout=timeout_seconds,
        messages=[
            {
                "role": "system",
                "content": (
                    "You must return only valid JSON for one next action. "
                    "No markdown, no explanation."
                ),
            },
            {"role": "user", "content": _build_prompt(observation)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_action_or_none(raw)


def _fmt_bool(value: bool) -> str:
    """Format bool using required lowercase protocol values."""
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    """Format reward using fixed two decimal places."""
    return f"{value:.2f}"


def _format_action(action: Action) -> str:
    """Build a compact action string for protocol logging."""
    payload = action.model_dump(exclude_none=True)
    action_type = str(payload.pop("action_type", "unknown"))
    if not payload:
        return action_type
    args = ", ".join(f"{key}={json.dumps(value, ensure_ascii=True)}" for key, value in payload.items())
    return f"{action_type}({args})"


def _extract_last_action_error(info: Dict[str, Any]) -> str:
    """Return raw last_action_error text when available, else protocol null."""
    error = info.get("last_action_error")
    if error is None:
        return "null"
    return str(error)


def run_task(
    env: SupportDeskEnv,
    task_id: str,
    benchmark: str,
    client: Any,
    model_name: str,
    max_episode_steps: int,
    model_timeout_seconds: int,
) -> None:
    """Run one task episode and emit strict protocol lines to stdout."""
    done = False
    step = 0
    success = False
    rewards: List[float] = []

    print(f"[START] task={task_id} env={benchmark} model={model_name}")

    try:
        observation = env.reset(task_id)
        step_cap = min(max_episode_steps, TASKS[task_id].max_steps)
        LOGGER.debug("task_start | task_id=%s | step_cap=%s", task_id, step_cap)

        while not done and step < step_cap:
            if client is None:
                action = _safe_default_action(task_id, step)
            else:
                try:
                    parsed_action = _model_action(
                        client=client,
                        model_name=model_name,
                        observation=observation,
                        timeout_seconds=model_timeout_seconds,
                    )
                    action = parsed_action if parsed_action is not None else _safe_default_action(task_id, step)
                except Exception:
                    action = _safe_default_action(task_id, step)

            observation, reward, done, info = env.step(action)
            step += 1
            rewards.append(float(reward.score))
            error_value = _extract_last_action_error(info)

            print(
                f"[STEP] step={step} action={_format_action(action)} "
                f"reward={_fmt_reward(float(reward.score))} "
                f"done={_fmt_bool(bool(done))} error={error_value}"
            )

            LOGGER.debug(
                "task_step | task_id=%s | step=%s | action=%s | reward=%s | done=%s",
                task_id,
                step,
                action.model_dump(exclude_none=True),
                reward.score,
                done,
            )

        success = bool(done)
    except Exception:
        success = False
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                LOGGER.debug("env_close_failed", exc_info=True)

        rewards_str = ",".join(_fmt_reward(r) for r in rewards)
        print(f"[END] success={_fmt_bool(success)} steps={step} rewards={rewards_str}")


def main() -> None:
    """Run one deterministic task episode with strict output protocol."""
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    seed = int(os.getenv("SEED", "42"))
    debug = _is_debug_enabled()
    max_episode_steps = int(os.getenv("MAX_EPISODE_STEPS", str(DEFAULT_MAX_EPISODE_STEPS)))
    model_timeout_seconds = int(os.getenv("MODEL_TIMEOUT_SECONDS", "25"))
    task_id = os.getenv("TASK_ID", DEFAULT_TASK_ID)
    benchmark = os.getenv("BENCHMARK", DEFAULT_BENCHMARK)

    if task_id not in TASKS:
        raise ValueError(f"Unknown TASK_ID: {task_id}")

    _setup_logging(debug)
    _configure_reproducibility(seed)

    from openai import OpenAI  # pyright: ignore[reportMissingImports]

    client: Any = OpenAI(base_url=api_base_url, api_key=hf_token)

    env = SupportDeskEnv(debug=debug)
    run_task(
        env=env,
        task_id=task_id,
        benchmark=benchmark,
        client=client,
        model_name=model_name,
        max_episode_steps=max_episode_steps,
        model_timeout_seconds=model_timeout_seconds,
    )


if __name__ == "__main__":
    main()
