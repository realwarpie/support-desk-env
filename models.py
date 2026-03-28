from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ActionType = Literal[
    "tag_ticket",
    "draft_reply",
    "request_info",
    "escalate",
    "close_ticket",
]

PriorityType = Literal["low", "medium", "high"]
TicketCategory = Literal[
    "refund",
    "shipping",
    "damaged_item",
    "order_change",
    "billing",
    "other",
]


class OrderMetadata(BaseModel):
    order_id: str
    status: str
    price: float
    delivery_date: Optional[date] = None


class Ticket(BaseModel):
    ticket_id: str
    message: str
    category: Optional[TicketCategory] = None
    priority: Optional[PriorityType] = None
    order: OrderMetadata


class ConversationTurn(BaseModel):
    role: Literal["customer", "agent", "system"]
    text: str


class Action(BaseModel):
    action_type: ActionType
    category: Optional[TicketCategory] = None
    priority: Optional[PriorityType] = None
    text: Optional[str] = None
    field: Optional[str] = None
    team: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "Action":
        if self.action_type == "tag_ticket" and (not self.category or not self.priority):
            raise ValueError("tag_ticket requires category and priority")
        if self.action_type == "draft_reply" and not self.text:
            raise ValueError("draft_reply requires text")
        if self.action_type == "request_info" and not self.field:
            raise ValueError("request_info requires field")
        if self.action_type == "escalate" and not self.team:
            raise ValueError("escalate requires team")
        if self.action_type == "close_ticket" and not self.reason:
            raise ValueError("close_ticket requires reason")
        return self


class Reward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    feedback: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    task_difficulty: str
    current_ticket: Ticket
    conversation_history: List[ConversationTurn]
    available_actions: List[str]
    policy_snippets: Dict[str, str]
    order_metadata: Dict[str, Any]
    steps_taken: int
    max_steps: int
