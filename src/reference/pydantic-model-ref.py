#!/usr/bin/env python3
"""
Practical Pydantic (v2) reference with runnable examples.
Run: python3 src/reference/pydantic-model-ref.py
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, EmailStr, ValidationError, field_validator


# ---------------------------------------------------------------------------
# 1. Basic model: fields become typed, validated attributes automatically.
# ---------------------------------------------------------------------------
class User(BaseModel):
    id: int
    name: str
    is_active: bool = True          # default value -> field becomes optional in constructor


def basic_example():
    u = User(id=1, name="Alice")
    print(u)                        # id=1 name='Alice' is_active=True
    print(u.model_dump())           # -> plain dict
    print(u.model_dump_json())      # -> JSON string

    # Pydantic coerces compatible types, e.g. "2" -> 2
    u2 = User(id="2", name="Bob")
    print(u2.id, type(u2.id))       # 2 <class 'int'>


# ---------------------------------------------------------------------------
# 2. Validation errors: bad data raises ValidationError with clear details.
# ---------------------------------------------------------------------------
def validation_error_example():
    try:
        User(id="not-a-number", name="Carl")
    except ValidationError as e:
        print(e.errors())           # list of dicts: loc, msg, type per bad field


# ---------------------------------------------------------------------------
# 3. Optional fields, Field() metadata/constraints, and EmailStr.
# ---------------------------------------------------------------------------
class Profile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    email: EmailStr
    bio: str = Field(default="", max_length=280, description="Short bio")


def field_constraints_example():
    p = Profile(username="siva", email="siva@example.com", age=30)
    print(p)

    try:
        Profile(username="ab", email="siva@example.com")  # username too short
    except ValidationError as e:
        print(e.errors())

    try:
        Profile(username="siva", email="not-an-email")    # malformed email
    except ValidationError as e:
        print(e.errors())


# ---------------------------------------------------------------------------
# 4. Enums restrict a field to a fixed set of values.
# ---------------------------------------------------------------------------
class Role(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    GUEST = "guest"


class Account(BaseModel):
    username: str
    role: Role = Role.MEMBER


def enum_example():
    a = Account(username="siva", role="admin")   # string is coerced to enum
    print(a.role, a.role.value)

    try:
        Account(username="x", role="owner")      # not a valid Role
    except ValidationError as e:
        print(e.errors())


# ---------------------------------------------------------------------------
# 5. Nested models: models can contain other models / lists of models.
# ---------------------------------------------------------------------------
class Address(BaseModel):
    city: str
    zip_code: str


class Customer(BaseModel):
    name: str
    addresses: list[Address] = []


def nested_model_example():
    c = Customer(
        name="Priya",
        addresses=[{"city": "Chennai", "zip_code": "600001"}],
    )
    print(c)
    print(c.addresses[0].city)      # nested attribute access

    # Also works building from a raw dict (e.g. parsed JSON / API payload)
    payload = {"name": "Ravi", "addresses": [{"city": "Pune", "zip_code": "411001"}]}
    c2 = Customer.model_validate(payload)
    print(c2)


# ---------------------------------------------------------------------------
# 6. Custom validators for logic Field() constraints can't express.
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    password: str
    password_confirm: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


def custom_validator_example():
    s = SignupRequest(password="supersecret", password_confirm="supersecret")
    print(s)

    try:
        SignupRequest(password="short", password_confirm="short")
    except ValidationError as e:
        print(e.errors())


# ---------------------------------------------------------------------------
# 7. Practical tie-in: use a Pydantic model to define/validate the shape of
#    an LLM response, e.g. for structured output from agent.py's OpenAI calls.
# ---------------------------------------------------------------------------
class AgentAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = []


def llm_structured_output_example():
    # Imagine this dict came from json.loads(model_response_text)
    raw = {"answer": "Paris is the capital of France.", "confidence": 0.97, "sources": []}
    result = AgentAnswer.model_validate(raw)
    print(result)


# ---------------------------------------------------------------------------
# 8. ConfigDict: model-wide settings, incl. extra="forbid" to reject unknown
#    fields (vs. the default "ignore", or "allow" to keep them).
# ---------------------------------------------------------------------------
class StrictOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    item: str
    quantity: int = Field(gt=0)


class LenientOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")  # default behavior, shown explicitly

    item: str


def config_dict_example():
    # str_strip_whitespace=True trims incoming strings automatically.
    o = StrictOrder(item="  Pen  ", quantity=3)
    print(o)                       # item='Pen' quantity=3

    try:
        # "discount" is not a declared field -> rejected because extra="forbid"
        StrictOrder(item="Pen", quantity=3, discount=10)
    except ValidationError as e:
        print(e.errors())

    # extra="ignore" silently drops unknown fields instead of erroring
    l = LenientOrder(item="Pen", discount=10)
    print(l)                       # discount is dropped, no error


if __name__ == "__main__":
    print("\n--- 1. basic_example ---")
    basic_example()

    print("\n--- 2. validation_error_example ---")
    validation_error_example()

    print("\n--- 3. field_constraints_example ---")
    field_constraints_example()

    print("\n--- 4. enum_example ---")
    enum_example()

    print("\n--- 5. nested_model_example ---")
    nested_model_example()

    print("\n--- 6. custom_validator_example ---")
    custom_validator_example()

    print("\n--- 7. llm_structured_output_example ---")
    llm_structured_output_example()

    print("\n--- 8. config_dict_example ---")
    config_dict_example()
