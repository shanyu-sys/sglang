import dataclasses
from typing import Optional


@dataclasses.dataclass
class Request:
    req_id: str
    prompt: str
    prompt_len: int
    output_len: int
    arrival_time: float
    model: str
    tokenizer: Optional[str] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclasses.dataclass
class ReplaceRequest:
    old_model_path: str
    new_model_path: str
    new_tokenizer_path: str
    load_format: str

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ReplaceResponse:
    success: bool
    error: Optional[str] = None


@dataclasses.dataclass
class DeleteRequest:
    model_path: str

