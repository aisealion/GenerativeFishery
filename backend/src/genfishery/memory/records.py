"""Memory record encoding (build spec §3).

Concordia's `AssociativeMemoryBank` stores only raw `text` + `embedding` --
no timestamp, importance, or kind columns. Rather than keep a side-table that
could desync (writes can be silently deduped), metadata travels *inline* in
the stored text itself, so it survives the memory bank's own
get_state()/set_state() serialization for free -- which is exactly what
portability across fisheries (§3) needs.
"""

import re
from dataclasses import dataclass

_PATTERN = re.compile(
    r"^\[seq=(?P<seq>\d+)\|round=(?P<round>\d+)\|importance=(?P<importance>[\d.]+)"
    r"\|kind=(?P<kind>\w+)\] (?P<content>.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class MemoryRecord:
    seq: int
    round: int
    importance: float
    kind: str  # "observation" | "reflection" | "plan"
    content: str


def encode(*, seq: int, round: int, importance: float, kind: str, content: str) -> str:
    clean_content = content.replace("\n", " ")
    return f"[seq={seq}|round={round}|importance={importance:.2f}|kind={kind}] {clean_content}"


def decode(text: str) -> MemoryRecord:
    match = _PATTERN.match(text)
    if not match:
        raise ValueError(f"malformed memory record (missing metadata prefix): {text!r}")
    return MemoryRecord(
        seq=int(match["seq"]),
        round=int(match["round"]),
        importance=float(match["importance"]),
        kind=match["kind"],
        content=match["content"],
    )
