from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal

from pydantic import BaseModel

from models import OrderMetadata, Ticket


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    ticket: Ticket
    expected_category: str
    expected_priority: str
    expected_close_reason: str
    required_reply_keywords: List[str]
    required_info_field: str | None = None
    required_escalation_team: str | None = None
    max_steps: int = 8


POLICY_SNIPPETS: Dict[str, str] = {
    "classification": (
        "Classify each ticket into one category and set priority. "
        "Use high priority for payment issues, damaged deliveries, or time-sensitive failures."
    ),
    "shipping_policy": (
        "If an order is in transit but delayed, apologize, share tracking guidance, "
        "and ask customer to wait 48 hours before refund eligibility."
    ),
    "refund_policy": (
        "Damaged item refunds require customer proof image before final approval. "
        "Escalate to returns team when damage is reported and ticket value exceeds $100."
    ),
    "closure_policy": (
        "Close a ticket only when resolution is provided or when waiting on customer information "
        "is clearly communicated."
    ),
}


TASKS: Dict[str, TaskSpec] = {
    "easy_classify_001": TaskSpec(
        task_id="easy_classify_001",
        difficulty="easy",
        title="Classify damaged delivery",
        description="Classify and prioritize a ticket where a customer received a broken item.",
        ticket=Ticket(
            ticket_id="TCK-1001",
            message=(
                "My blender arrived cracked and leaking. I need help quickly because this was a gift."
            ),
            order=OrderMetadata(
                order_id="ORD-7010",
                status="delivered",
                price=89.99,
                delivery_date=date(2026, 3, 26),
            ),
        ),
        expected_category="damaged_item",
        expected_priority="high",
        expected_close_reason="classified",
        required_reply_keywords=[],
        max_steps=4,
    ),
    "medium_response_001": TaskSpec(
        task_id="medium_response_001",
        difficulty="medium",
        title="Shipping delay policy response",
        description="Tag correctly, respond with policy-compliant guidance, and close.",
        ticket=Ticket(
            ticket_id="TCK-2001",
            message=(
                "My package has been stuck for three days. Where is it and can I get a refund now?"
            ),
            order=OrderMetadata(
                order_id="ORD-8452",
                status="in_transit",
                price=44.5,
                delivery_date=date(2026, 3, 24),
            ),
        ),
        expected_category="shipping",
        expected_priority="medium",
        expected_close_reason="resolved_with_guidance",
        required_reply_keywords=["sorry", "tracking", "48 hours"],
        max_steps=6,
    ),
    "hard_resolution_001": TaskSpec(
        task_id="hard_resolution_001",
        difficulty="hard",
        title="Damaged high-value item with escalation",
        description=(
            "Handle a high-value damaged item with missing evidence, proper escalation, "
            "and an appropriate holding close reason."
        ),
        ticket=Ticket(
            ticket_id="TCK-3001",
            message=(
                "The laptop I received has a shattered screen. I want a refund immediately."
            ),
            order=OrderMetadata(
                order_id="ORD-9901",
                status="delivered",
                price=1299.0,
                delivery_date=date(2026, 3, 27),
            ),
        ),
        expected_category="refund",
        expected_priority="high",
        expected_close_reason="awaiting_customer_info",
        required_reply_keywords=["photo", "escalated", "returns team"],
        required_info_field="damage_photo",
        required_escalation_team="returns",
        max_steps=8,
    ),
}
