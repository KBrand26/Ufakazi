"""Model-provider adapters behind one interface.

Each provider implements :class:`Provider`, isolating per-provider differences (notably
logprob availability: OpenAI exposes it, Anthropic does not) behind a capability flag.
Likely litellm-backed; the interface keeps providers swappable.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Response:
    """A single model response to a trial prompt."""

    text: str
    choice_logprob: float | None = None  # None when the provider exposes no logprobs


class Provider(Protocol):
    """Common surface every provider adapter must satisfy."""

    supports_logprobs: bool

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        capture_logprobs: bool = False,
    ) -> Response: ...
