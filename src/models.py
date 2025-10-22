from enum import Enum
from typing import Any, List, Set

from pydantic import BaseModel, conset

ROLE_KEY = "role"
CONTENT_KEY = "content"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class Risk(str, Enum):
    HARM = "harm"
    SOCIAL_BIAS = "social_bias"
    JAILBREAKING = "jailbreak"
    PROFANITY = "profanity"
    UNETHICAL_BEHAVIOR = "unethical_behavior"
    VIOLENCE = "violence"
    HARM_ENGAGEMENT = "harm_engagement"
    EVASIVENESS = "evasiveness"


class Dialogue(BaseModel):
    prompt: str
    response: str = None

    def to_message(self) -> List:
        result = [{ROLE_KEY: USER_ROLE, CONTENT_KEY: self.prompt}]
        if self.response is not None:
            result.append({ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: self.response})
        return result


class ModerationInput(BaseModel):
    risks_to_detect: Set[Risk] = conset(item_type=Risk, min_length=1)
    dialogue_history: List[Dialogue]

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)

    def to_messages(self):
        result = []
        for dialogue in self.dialogue_history:
            result.extend(dialogue.to_message())
        return result


class ModerationResult(BaseModel):
    risk_name: str
    detected: bool
    confidence: str
    probability: float


class ModerationOutput(BaseModel):
    risks: List[ModerationResult]
