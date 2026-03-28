from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import Action
from tasks import TaskSpec


def _clamp_01(value: float) -> float:
    """Clamp score into [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _keyword_match_score(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """Return keyword coverage ratio and missing keywords list."""
    if not keywords:
        return 1.0, []

    lowered = text.lower()
    found = [kw for kw in keywords if kw in lowered]
    missing = [kw for kw in keywords if kw not in lowered]
    return len(found) / len(keywords), missing


def _first_index(actions: List[Action], action_type: str) -> Optional[int]:
    """Find first index of an action type."""
    return next((i for i, a in enumerate(actions) if a.action_type == action_type), None)


def _first_action(actions: List[Action], action_type: str) -> Optional[Action]:
    """Return first action of given type if present."""
    idx = _first_index(actions, action_type)
    return None if idx is None else actions[idx]


def _duplicate_penalty(actions: List[Action]) -> float:
    """Penalize repeated identical action payloads deterministically."""
    seen: Dict[str, int] = {}
    penalty = 0.0
    for action in actions:
        key = str(sorted(action.model_dump(exclude_none=True).items()))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            penalty += 0.03
    return penalty


def _efficiency_credit(steps: int, optimal_steps: int) -> float:
    """Reward concise trajectories with deterministic decay after optimum."""
    if steps <= optimal_steps:
        return 1.0
    overshoot = steps - optimal_steps
    return max(0.0, 1.0 - 0.2 * overshoot)


def _is_ordered(indices: List[Optional[int]]) -> bool:
    """Check whether all indices exist and are strictly increasing."""
    if any(i is None for i in indices):
        return False
    values = [int(i) for i in indices if i is not None]
    return values == sorted(values) and len(set(values)) == len(values)


def _policy_violation_penalty(text: str, banned_phrases: List[str]) -> Tuple[float, List[str]]:
    """Check banned policy phrases and return penalty with violations."""
    lowered = text.lower()
    hits = [phrase for phrase in banned_phrases if phrase in lowered]
    return (0.05 * len(hits), hits)


def grade_easy(task: TaskSpec, actions: List[Action]) -> Dict[str, object]:
    """Strict deterministic grader for easy classification task."""
    score = 0.0
    feedback: List[str] = []

    tag = _first_action(actions, "tag_ticket")
    if tag is not None:
        if tag.category == task.expected_category:
            score += 0.55
        else:
            feedback.append("Incorrect category for easy task.")

        if tag.priority == task.expected_priority:
            score += 0.25
        else:
            feedback.append("Incorrect priority for easy task.")
    else:
        feedback.append("Missing required tag_ticket action.")

    if actions and actions[0].action_type == "tag_ticket":
        score += 0.1
    elif actions:
        feedback.append("First action should be tag_ticket for easy task.")

    efficiency = _efficiency_credit(len(actions), optimal_steps=1)
    score += 0.1 * efficiency
    if efficiency < 1.0:
        feedback.append("Too many actions for easy task.")

    non_tag_count = sum(1 for action in actions if action.action_type != "tag_ticket")
    if non_tag_count > 0:
        score -= 0.1 * non_tag_count
        feedback.append("Easy task should avoid non-tag actions.")

    tag_count = sum(1 for action in actions if action.action_type == "tag_ticket")
    if tag_count > 1:
        score -= 0.05 * (tag_count - 1)
        feedback.append("Repeated tag_ticket actions are redundant.")

    score -= _duplicate_penalty(actions)

    return {
        "score": _clamp_01(score),
        "feedback": feedback,
        "done": tag is not None,
    }


def grade_medium(task: TaskSpec, actions: List[Action]) -> Dict[str, object]:
    """Strict deterministic grader for medium policy-compliance response task."""
    score = 0.0
    feedback: List[str] = []

    tag_idx = _first_index(actions, "tag_ticket")
    reply_idx = _first_index(actions, "draft_reply")
    close_idx = _first_index(actions, "close_ticket")

    tag = _first_action(actions, "tag_ticket")
    reply = _first_action(actions, "draft_reply")
    close_action = _first_action(actions, "close_ticket")

    if tag is not None:
        if tag.category == task.expected_category:
            score += 0.2
        else:
            feedback.append("Medium task category is incorrect.")
        if tag.priority == task.expected_priority:
            score += 0.1
        else:
            feedback.append("Medium task priority is incorrect.")
    else:
        feedback.append("Missing tag_ticket action.")

    if reply is not None:
        kw_score, missing = _keyword_match_score(reply.text or "", task.required_reply_keywords)
        score += 0.25 * kw_score
        if missing:
            feedback.append(f"Missing reply keywords: {', '.join(missing)}")

        violation_penalty, violations = _policy_violation_penalty(
            reply.text or "",
            ["immediate refund", "refund now", "guaranteed refund today"],
        )
        score += 0.1
        score -= violation_penalty
        if violations:
            feedback.append(f"Policy-violating promises in reply: {', '.join(violations)}")
    else:
        feedback.append("Missing draft_reply action.")

    if close_action is not None:
        if close_action.reason == task.expected_close_reason:
            score += 0.2
        else:
            feedback.append("Incorrect close reason for medium task.")
        if reply_idx is not None and close_idx is not None and close_idx > reply_idx:
            score += 0.05
        elif reply_idx is not None:
            feedback.append("close_ticket must happen after draft_reply.")
    else:
        feedback.append("Missing close_ticket action.")

    if _is_ordered([tag_idx, reply_idx, close_idx]):
        score += 0.15
    elif any(i is not None for i in [tag_idx, reply_idx, close_idx]):
        feedback.append("Action sequence must be tag -> reply -> close.")

    score += 0.1 * _efficiency_credit(len(actions), optimal_steps=3)

    forbidden = sum(
        1
        for a in actions
        if a.action_type
        in {"request_info", "escalate"}
    )
    if forbidden > 0:
        score -= 0.08 * forbidden
        feedback.append("Medium task should not request info or escalate.")

    score -= _duplicate_penalty(actions)

    done = tag is not None and reply is not None and close_action is not None
    return {
        "score": _clamp_01(score),
        "feedback": feedback,
        "done": done,
    }


def grade_hard(task: TaskSpec, actions: List[Action]) -> Dict[str, object]:
    """Strict deterministic grader for hard multi-step resolution task."""
    score = 0.0
    feedback: List[str] = []

    tag_idx = _first_index(actions, "tag_ticket")
    info_idx = _first_index(actions, "request_info")
    esc_idx = _first_index(actions, "escalate")
    reply_idx = _first_index(actions, "draft_reply")
    close_idx = _first_index(actions, "close_ticket")

    tag = _first_action(actions, "tag_ticket")
    info_action = _first_action(actions, "request_info")
    esc_action = _first_action(actions, "escalate")
    reply = _first_action(actions, "draft_reply")
    close_action = _first_action(actions, "close_ticket")

    if tag is not None:
        if tag.category == task.expected_category:
            score += 0.12
        else:
            feedback.append("Hard task category is incorrect.")
        if tag.priority == task.expected_priority:
            score += 0.08
        else:
            feedback.append("Hard task priority is incorrect.")
    else:
        feedback.append("Missing tag_ticket action.")

    if info_action is not None:
        if info_action.field == task.required_info_field:
            score += 0.15
        else:
            feedback.append("Wrong info field requested for hard task.")
    else:
        feedback.append("Missing request_info action.")

    if esc_action is not None:
        if esc_action.team == task.required_escalation_team:
            score += 0.15
        else:
            feedback.append("Wrong escalation team for hard task.")
    else:
        feedback.append("Missing escalate action.")

    if reply is not None:
        kw_score, missing = _keyword_match_score(reply.text or "", task.required_reply_keywords)
        score += 0.15 * kw_score
        if missing:
            feedback.append(f"Missing reply keywords: {', '.join(missing)}")

        violation_penalty, violations = _policy_violation_penalty(
            reply.text or "",
            ["refund approved", "full refund processed now", "instant refund issued"],
        )
        score += 0.08
        score -= violation_penalty
        if violations:
            feedback.append(f"Policy-violating refund promise: {', '.join(violations)}")
    else:
        feedback.append("Missing draft_reply action.")

    if close_action is not None:
        if close_action.reason == task.expected_close_reason:
            score += 0.12
        else:
            feedback.append("Incorrect close reason for hard task.")
    else:
        feedback.append("Missing close_ticket action.")

    if _is_ordered([tag_idx, info_idx, esc_idx, reply_idx, close_idx]):
        score += 0.15
    elif any(i is not None for i in [tag_idx, info_idx, esc_idx, reply_idx, close_idx]):
        feedback.append("Expected sequence: tag -> request_info -> escalate -> reply -> close.")

    score += 0.05 * _efficiency_credit(len(actions), optimal_steps=5)
    score -= _duplicate_penalty(actions)

    done = all(
        present is not None
        for present in [tag, info_action, esc_action, reply, close_action]
    )

    return {
        "score": _clamp_01(score),
        "feedback": feedback,
        "done": done,
    }


def grade_task(task: TaskSpec, actions: List[Action]) -> Dict[str, object]:
    """Route grading to the task-specific deterministic grader."""
    if task.difficulty == "easy":
        return grade_easy(task, actions)
    if task.difficulty == "medium":
        return grade_medium(task, actions)
    return grade_hard(task, actions)
