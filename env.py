from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graders import grade_task
from models import Action, ConversationTurn, Observation, Reward, Ticket
from tasks import POLICY_SNIPPETS, TASKS, TaskSpec


AVAILABLE_ACTIONS = [
    "tag_ticket(category, priority)",
    "draft_reply(text)",
    "request_info(field)",
    "escalate(team)",
    "close_ticket(reason)",
]


LOGGER = logging.getLogger("support_desk_env")


def _is_debug_enabled() -> bool:
    """Return whether debug logging is enabled via DEBUG env var."""
    return os.getenv("DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


class SupportDeskEnv:
    """OpenEnv-style customer support inbox simulator."""

    def __init__(self, default_task_id: str = "easy_classify_001", debug: Optional[bool] = None) -> None:
        """Create environment with optional debug tracing."""
        if default_task_id not in TASKS:
            raise ValueError(f"Unknown default task_id: {default_task_id}")
        self.default_task_id = default_task_id
        self._debug = _is_debug_enabled() if debug is None else debug
        if self._debug:
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._task: TaskSpec
        self._ticket: Ticket
        self._history: List[ConversationTurn]
        self._actions: List[Action]
        self._steps: int
        self._max_steps: int
        self._done: bool
        self._prev_score: float
        self._invalid_actions: int
        self._last_action_signatures: List[str]
        self._milestone_index: int
        self._ticket_status: str
        self._resolution_stage: str
        self._provided_info_fields: set[str]
        self._customer_frustration: int
        self._reset_internal(default_task_id)

    def _reset_internal(self, task_id: str) -> None:
        """Reset episode internals for the selected task."""
        self._task = deepcopy(TASKS[task_id])
        self._ticket = deepcopy(self._task.ticket)
        self._history = [ConversationTurn(role="customer", text=self._ticket.message)]
        self._actions = []
        self._steps = 0
        self._max_steps = self._task.max_steps
        self._done = False
        self._prev_score = 0.0
        self._invalid_actions = 0
        self._last_action_signatures = []
        self._milestone_index = 0
        self._ticket_status = "open"
        self._resolution_stage = "classification"
        self._provided_info_fields = set()
        self._customer_frustration = 0
        self._log_debug("episode_reset", {"task_id": task_id, "max_steps": self._max_steps})

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset environment and return initial observation."""
        selected = task_id or self.default_task_id
        if selected not in TASKS:
            raise ValueError(f"Unknown task_id: {selected}")
        self._reset_internal(selected)
        return self._build_observation()

    def step(self, action: Any) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Apply one action to the environment and return transition tuple.

        This method is resilient to malformed payloads and unknown actions by
        applying deterministic penalties and preserving consistent state.
        """
        if self._done:
            obs = self._build_observation()
            reward = Reward(score=0.0, components={"already_done": 0.0}, feedback=["Episode is done."])
            return obs, reward, True, {"message": "reset required"}

        self._steps += 1
        parsed_action, invalid_penalty, parse_feedback = self._normalize_action(action)

        if parsed_action is not None:
            self._actions.append(parsed_action)
            self._update_ticket_state(parsed_action)
            self._append_history(parsed_action)
            customer_turn = self._simulate_customer_response(parsed_action)
            if customer_turn is not None:
                self._history.append(customer_turn)
        else:
            self._history.append(
                ConversationTurn(
                    role="system",
                    text="Invalid action payload received and ignored.",
                )
            )

        grade = grade_task(self._task, self._actions)
        current_score = float(cast(float, grade["score"]))
        grade_delta = current_score - self._prev_score

        progress_reward = max(0.0, grade_delta)
        regression_penalty = min(0.0, grade_delta)
        intermediate_reward = self._intermediate_reward(parsed_action)
        incorrect_penalty = self._incorrect_action_penalty(parsed_action)
        redundant_penalty = self._redundant_action_penalty(parsed_action)
        loop_penalty = self._detect_loop_penalty(parsed_action)
        reply_quality_reward = self._meaningful_reply_reward(parsed_action)
        keyword_signal_reward = self._keyword_signal_reward(parsed_action)
        action_diversity_reward = self._action_diversity_reward(parsed_action)
        low_info_penalty = self._low_information_penalty(parsed_action)
        step_cost = -0.01

        done_by_grader = bool(grade["done"])
        done_by_limit = self._steps >= self._max_steps
        done_by_loop = loop_penalty <= -0.25
        self._done = done_by_grader or done_by_limit or done_by_loop

        efficiency_bonus = self._efficiency_bonus(done_by_grader)
        max_step_penalty = -0.2 if done_by_limit and not done_by_grader else 0.0
        invalid_limit_penalty = -0.1 if self._invalid_actions >= 3 else 0.0

        step_reward_value = (
            progress_reward
            + regression_penalty
            + intermediate_reward
            + incorrect_penalty
            + redundant_penalty
            + loop_penalty
            + reply_quality_reward
            + keyword_signal_reward
            + action_diversity_reward
            + low_info_penalty
            + step_cost
            + efficiency_bonus
            + max_step_penalty
            + invalid_penalty
            + invalid_limit_penalty
        )
        step_reward_value = max(-1.0, min(1.0, step_reward_value))

        self._prev_score = current_score
        feedback = cast(List[str], grade["feedback"]) + parse_feedback

        info: Dict[str, Any] = {
            "task_id": self._task.task_id,
            "grader_score": current_score,
            "feedback": feedback,
            "steps_taken": self._steps,
            "max_steps": self._max_steps,
            "final_outcome_quality": current_score,
            "invalid_actions": self._invalid_actions,
            "ticket_status": self._ticket_status,
            "resolution_stage": self._resolution_stage,
        }
        if done_by_limit and not done_by_grader:
            info["termination"] = "max_steps_reached"
        if done_by_loop:
            info["termination"] = "loop_detected"
        if parsed_action is None:
            info["termination"] = "invalid_action"

        reward_components = {
            "progress_reward": round(progress_reward, 6),
            "regression_penalty": round(regression_penalty, 6),
            "intermediate_reward": round(intermediate_reward, 6),
            "incorrect_penalty": round(incorrect_penalty, 6),
            "redundant_penalty": round(redundant_penalty, 6),
            "loop_penalty": round(loop_penalty, 6),
            "reply_quality_reward": round(reply_quality_reward, 6),
            "keyword_signal_reward": round(keyword_signal_reward, 6),
            "action_diversity_reward": round(action_diversity_reward, 6),
            "low_info_penalty": round(low_info_penalty, 6),
            "step_cost": round(step_cost, 6),
            "efficiency_bonus": round(efficiency_bonus, 6),
            "max_step_penalty": round(max_step_penalty, 6),
            "invalid_penalty": round(invalid_penalty, 6),
            "invalid_limit_penalty": round(invalid_limit_penalty, 6),
        }

        reward = Reward(
            score=step_reward_value,
            components=reward_components,
            feedback=feedback,
        )

        self._log_debug(
            "step_transition",
            {
                "task_id": self._task.task_id,
                "step": self._steps,
                "action": None if parsed_action is None else parsed_action.model_dump(exclude_none=True),
                "reward": reward.model_dump(),
                "done": self._done,
                "info": info,
            },
        )

        return self._build_observation(), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Expose complete internal state for debugging and evaluation."""
        return {
            "task": self._task.model_dump(mode="json"),
            "ticket": self._ticket.model_dump(mode="json"),
            "conversation_history": [turn.model_dump() for turn in self._history],
            "actions": [action.model_dump(exclude_none=True) for action in self._actions],
            "steps": self._steps,
            "max_steps": self._max_steps,
            "done": self._done,
            "score_so_far": self._prev_score,
            "invalid_actions": self._invalid_actions,
            "milestone_index": self._milestone_index,
            "ticket_state": {
                "status": self._ticket_status,
                "resolution_stage": self._resolution_stage,
            },
        }

    def _build_observation(self) -> Observation:
        """Construct current observation payload."""
        return Observation(
            task_id=self._task.task_id,
            task_difficulty=self._task.difficulty,
            current_ticket=self._ticket,
            conversation_history=self._history,
            available_actions=AVAILABLE_ACTIONS,
            policy_snippets=POLICY_SNIPPETS,
            order_metadata={
                **self._ticket.order.model_dump(mode="json"),
                "ticket_status": self._ticket_status,
                "resolution_stage": self._resolution_stage,
            },
            steps_taken=self._steps,
            max_steps=self._max_steps,
        )

    def _update_ticket_state(self, action: Action) -> None:
        """Apply deterministic ticket lifecycle transitions from an action."""
        if action.action_type == "tag_ticket":
            self._resolution_stage = "investigation"
            if self._ticket_status == "open":
                self._ticket_status = "open"
            return

        if action.action_type == "request_info":
            self._ticket_status = "waiting_for_customer"
            self._resolution_stage = "investigation"
            return

        if action.action_type == "escalate":
            self._ticket_status = "escalated"
            self._resolution_stage = "resolution"
            return

        if action.action_type == "draft_reply":
            if self._ticket_status not in {"escalated", "closed"}:
                self._ticket_status = "resolved"
            self._resolution_stage = "resolution"
            return

        if action.action_type == "close_ticket":
            self._ticket_status = "closed"
            self._resolution_stage = "resolution"

    def _simulate_customer_response(self, action: Action) -> Optional[ConversationTurn]:
        """Generate deterministic, context-aware customer follow-up messages.

        Behavior depends on ticket status, resolution stage, and recent agent action
        history. No randomness is used.
        """
        if action.action_type == "request_info" and action.field:
            if action.field in self._provided_info_fields:
                self._customer_frustration += 1
                return ConversationTurn(
                    role="customer",
                    text=(
                        f"I already shared {action.field}. Please check the previous upload "
                        "and move this forward."
                    ),
                )

            self._provided_info_fields.add(action.field)
            return ConversationTurn(
                role="customer",
                text=f"I have uploaded the {action.field}. Please confirm if anything else is needed.",
            )

        if action.action_type == "escalate":
            if self._has_repeated_recent_action("escalate"):
                self._customer_frustration += 1
                return ConversationTurn(
                    role="customer",
                    text="You already escalated this. I need an actionable update, not another transfer.",
                )
            return ConversationTurn(
                role="customer",
                text="Thanks for escalating. Please keep me updated with the next concrete step.",
            )

        if action.action_type == "close_ticket":
            if self._ticket_status == "closed" and self._resolution_stage == "resolution":
                if self._task.required_info_field and self._task.required_info_field not in self._provided_info_fields:
                    self._customer_frustration += 2
                    return ConversationTurn(
                        role="customer",
                        text=(
                            "Why was this closed before resolution? I still need help and will reopen this issue."
                        ),
                    )
                return ConversationTurn(
                    role="customer",
                    text="Thanks for resolving this. I will reopen if anything changes.",
                )

        if action.action_type != "draft_reply" or not action.text:
            return None

        reply_text = action.text.strip().lower()
        repeated_reply = self._has_repeated_recent_action("draft_reply")
        low_info = self._is_low_information_reply(reply_text)

        if low_info:
            self._customer_frustration += 2
            return ConversationTurn(
                role="customer",
                text="I still do not understand the resolution. Can you give clear next steps?",
            )

        if repeated_reply:
            self._customer_frustration += 1
            return ConversationTurn(
                role="customer",
                text="This sounds like the same response as before. What is actually changing on my case?",
            )

        if self._ticket_status == "waiting_for_customer":
            return ConversationTurn(
                role="customer",
                text="I provided what you asked for. Can you continue with the resolution now?",
            )

        if self._resolution_stage == "classification":
            return ConversationTurn(
                role="customer",
                text="Can you confirm the exact issue category and expected turnaround time?",
            )

        if self._resolution_stage == "investigation":
            if self._task.required_info_field and self._task.required_info_field not in self._provided_info_fields:
                return ConversationTurn(
                    role="customer",
                    text=(
                        "I am confused about what evidence is needed. Please specify exactly what to provide."
                    ),
                )
            return ConversationTurn(
                role="customer",
                text="Thanks, I shared the requested details. Please confirm the investigation outcome.",
            )

        if self._ticket_status == "escalated":
            return ConversationTurn(
                role="customer",
                text="I appreciate the escalation. Please share an ETA from the specialist team.",
            )

        if self._customer_frustration >= 2:
            return ConversationTurn(
                role="customer",
                text="This is taking too long and remains unresolved. I need a concrete resolution now.",
            )

        return ConversationTurn(
            role="customer",
            text="Thanks for the update. Please keep me posted on the final resolution.",
        )

    def _has_repeated_recent_action(self, action_type: str) -> bool:
        """Return True when the same action type repeats in recent turns."""
        recent = [a.action_type for a in self._actions[-3:]]
        if len(recent) < 2:
            return False
        return recent[-1] == action_type and recent[-2] == action_type

    @staticmethod
    def _is_low_information_reply(reply_text: str) -> bool:
        """Classify low-information replies for deterministic customer behavior."""
        low_info_tokens = {"ok", "done", "noted", "thanks", "resolved", "wait"}
        normalized = " ".join(reply_text.split())
        if normalized in low_info_tokens:
            return True
        return len(normalized.split()) <= 4

    def _normalize_action(self, action: Any) -> Tuple[Optional[Action], float, List[str]]:
        """Validate action payload and map malformed actions to safe penalties."""
        if isinstance(action, Action):
            return action, 0.0, []

        feedback: List[str] = []
        try:
            parsed = Action.model_validate(action)
            return parsed, 0.0, []
        except Exception as exc:
            self._invalid_actions += 1
            feedback.append(f"Invalid action payload: {exc}")
            return None, -0.2, feedback

    def _append_history(self, action: Action) -> None:
        """Append system or agent turn based on action semantics."""
        if action.action_type == "draft_reply" and action.text:
            self._history.append(ConversationTurn(role="agent", text=action.text))
        elif action.action_type == "request_info" and action.field:
            self._history.append(
                ConversationTurn(
                    role="agent",
                    text=f"Please share {action.field} so we can continue.",
                )
            )
        elif action.action_type == "escalate" and action.team:
            self._history.append(
                ConversationTurn(
                    role="system",
                    text=f"Ticket escalated to {action.team} team. Status is now escalated.",
                )
            )
        elif action.action_type == "close_ticket" and action.reason:
            self._history.append(
                ConversationTurn(
                    role="system",
                    text=f"Ticket closed: {action.reason}. Status is now closed.",
                )
            )
        elif action.action_type == "tag_ticket":
            self._history.append(
                ConversationTurn(
                    role="system",
                    text=f"Ticket tagged as {action.category}/{action.priority}.",
                )
            )

    def _meaningful_reply_reward(self, action: Optional[Action]) -> float:
        """Reward sufficiently informative reply length in draft responses."""
        if action is None or action.action_type != "draft_reply" or not action.text:
            return 0.0

        length = len(action.text.strip())
        if length >= 80:
            return 0.05
        if length >= 35:
            return 0.03
        if length >= 15:
            return 0.01
        return -0.04

    def _keyword_signal_reward(self, action: Optional[Action]) -> float:
        """Reward direct policy keyword coverage without depending solely on grader score."""
        if action is None or action.action_type != "draft_reply" or not action.text:
            return 0.0

        required_keywords = [kw.lower() for kw in self._task.required_reply_keywords]
        if not required_keywords:
            return 0.0

        lowered = action.text.lower()
        matched = sum(1 for kw in required_keywords if kw in lowered)
        coverage = matched / len(required_keywords)
        return 0.06 * coverage

    def _action_diversity_reward(self, action: Optional[Action]) -> float:
        """Reward non-repetitive action type progression."""
        if action is None or len(self._actions) < 2:
            return 0.0

        prev_type = self._actions[-2].action_type
        if prev_type == action.action_type:
            return -0.03
        return 0.02

    def _low_information_penalty(self, action: Optional[Action]) -> float:
        """Penalize shallow low-information replies that do not help resolution."""
        if action is None or action.action_type != "draft_reply" or not action.text:
            return 0.0

        short = action.text.strip().lower()
        low_info_phrases = {"ok", "done", "noted", "thanks", "please wait", "resolved"}
        if short in low_info_phrases:
            return -0.08

        if len(short.split()) <= 3:
            return -0.05
        return 0.0

    def _detect_loop_penalty(self, action: Optional[Action]) -> float:
        """Penalize repeated action loops, including A-A-A and A-B-A-B patterns."""
        if action is None:
            return 0.0

        signature = self._canonical_action(action)
        self._last_action_signatures.append(signature)
        if len(self._last_action_signatures) > 6:
            self._last_action_signatures = self._last_action_signatures[-6:]

        if len(self._last_action_signatures) >= 3:
            if (
                self._last_action_signatures[-1]
                == self._last_action_signatures[-2]
                == self._last_action_signatures[-3]
            ):
                return -0.25

        if len(self._last_action_signatures) >= 4:
            a, b, c, d = self._last_action_signatures[-4:]
            if a == c and b == d and a != b:
                return -0.15

        return 0.0

    def _redundant_action_penalty(self, action: Optional[Action]) -> float:
        """Penalize exact duplicate actions to discourage redundant work."""
        if action is None:
            return 0.0

        current = self._canonical_action(action)
        seen = sum(1 for existing in self._actions[:-1] if self._canonical_action(existing) == current)
        if seen == 0:
            return 0.0
        return -0.06 * min(3, seen)

    def _incorrect_action_penalty(self, action: Optional[Action]) -> float:
        """Penalize task-inappropriate actions and wrong closure behavior."""
        if action is None:
            return -0.05

        if self._task.difficulty == "easy" and action.action_type != "tag_ticket":
            return -0.12

        if self._task.difficulty == "medium" and action.action_type in {"escalate", "request_info"}:
            return -0.08

        if action.action_type == "close_ticket":
            has_reply = any(a.action_type == "draft_reply" for a in self._actions)
            has_tag = any(a.action_type == "tag_ticket" for a in self._actions)
            if not has_tag:
                return -0.15
            if self._task.difficulty != "easy" and not has_reply:
                return -0.12

        return 0.0

    def _intermediate_reward(self, action: Optional[Action]) -> float:
        """Give dense reward for correct intermediate sequence progression."""
        if action is None:
            return 0.0

        expected_sequence = self._expected_sequence()
        if self._milestone_index >= len(expected_sequence):
            return 0.0

        expected = expected_sequence[self._milestone_index]
        if action.action_type == expected:
            self._milestone_index += 1
            return 0.08

        if action.action_type in expected_sequence[self._milestone_index + 1 :]:
            return -0.04

        return 0.0

    def _efficiency_bonus(self, done_by_grader: bool) -> float:
        """Reward solving tasks in fewer steps once episode reaches valid completion."""
        if not done_by_grader:
            return 0.0

        optimal_steps = {"easy": 1, "medium": 3, "hard": 5}[self._task.difficulty]
        if self._steps <= optimal_steps:
            return 0.1

        over = self._steps - optimal_steps
        return max(0.0, 0.1 - 0.02 * over)

    def _expected_sequence(self) -> List[str]:
        """Return expected action order by difficulty."""
        if self._task.difficulty == "easy":
            return ["tag_ticket"]
        if self._task.difficulty == "medium":
            return ["tag_ticket", "draft_reply", "close_ticket"]
        return ["tag_ticket", "request_info", "escalate", "draft_reply", "close_ticket"]

    @staticmethod
    def _canonical_action(action: Action) -> str:
        """Create deterministic canonical representation of an action."""
        payload = action.model_dump(exclude_none=True)
        return str(sorted(payload.items()))

    def _log_debug(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit debug logs when DEBUG is enabled."""
        if not self._debug:
            return
        LOGGER.debug("%s | %s", event, payload)


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""

    task_id: Optional[str] = None


class StepRequest(BaseModel):
    """Request model for step endpoint.

    Accept raw dict payload to support robust malformed-action handling.
    """

    action: Dict[str, Any]


app = FastAPI(title="SupportDeskEnv", version="1.0.0")
ENV = SupportDeskEnv()


@app.get("/tasks")
def list_tasks() -> Dict[str, Dict[str, str]]:
    return {
        k: {
            "difficulty": v.difficulty,
            "title": v.title,
            "description": v.description,
        }
        for k, v in TASKS.items()
    }


@app.post("/reset")
def api_reset(req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reset endpoint that accepts empty/missing JSON body safely."""
    task_id = req.get("task_id") if isinstance(req, dict) else None
    if task_id is not None and not isinstance(task_id, str):
        task_id = None

    try:
        obs = ENV.reset(task_id)
    except Exception:
        # Fallback to default task to avoid validator failures on malformed body.
        obs = ENV.reset()

    return {"observation": obs.model_dump(mode="json")}


@app.post("/step")
def api_step(req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Step endpoint that accepts malformed input without crashing."""
    action_payload = req.get("action") if isinstance(req, dict) else None

    try:
        obs, reward, done, info = ENV.step(action_payload)
    except Exception as exc:
        obs = ENV._build_observation()
        reward = Reward(
            score=-0.2,
            components={"endpoint_error_penalty": -0.2},
            feedback=["step failed safely"],
        )
        done = False
        info = {"termination": "step_error", "error": str(exc)}

    return {
        "observation": obs.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": json.loads(json.dumps(info, default=str)),
    }


@app.get("/state")
def api_state() -> Dict[str, Any]:
    """Return fully JSON-serializable environment state."""
    return json.loads(json.dumps(ENV.state(), default=str))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("env:app", host="0.0.0.0", port=8000, reload=False)
