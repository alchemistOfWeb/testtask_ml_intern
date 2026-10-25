import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DATE_PATTERN = r"^\d{2}\.\d{2}\.\d{4}$"


class PassportFields(BaseModel):
    """The 10 fields in exactly the format of train_labels.jsonl."""

    Last_name_ru: str = Field(min_length=1)
    First_name_ru: str = Field(min_length=1)
    Middle_name_ru: str  # empty when the person has no patronymic
    Sex_ru: Literal["МУЖ.", "ЖЕН."]
    Birth_date: str = Field(pattern=DATE_PATTERN)
    Birth_place_ru: str = Field(min_length=1)
    Issue_date: str = Field(pattern=DATE_PATTERN)
    Issue_organisation_code: str = Field(pattern=r"^\d{3}-\d{3}$")
    Licence_number: str = Field(pattern=r"^\d{2} \d{2} \d{6}$")
    Issue_organization_ru: str = Field(min_length=1)

    @field_validator("Birth_date", "Issue_date")
    @classmethod
    def real_date(cls, value):
        datetime.strptime(value, "%d.%m.%Y")
        return value

    def birth(self):
        return datetime.strptime(self.Birth_date, "%d.%m.%Y").date()

    def issue(self):
        return datetime.strptime(self.Issue_date, "%d.%m.%Y").date()


class Answer(BaseModel):
    doc_id: str
    decision: Literal["accept", "reject"]
    fields: PassportFields | None = None
    reasons: list[str] | None = None

    def to_json(self, with_reasons=False):
        data = self.model_dump(exclude_none=True, exclude=None if with_reasons else {"reasons"})
        return json.dumps(data, ensure_ascii=False)
