"""Memory stream for patient NPC (Layer 2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    role: str    # "doctor" | "patient"
    text: str
    turn: int


@dataclass
class MemoryStream:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add_doctor(self, text: str, turn: int) -> None:
        self.entries.append(MemoryEntry("doctor", text, turn))

    def add_patient(self, text: str, turn: int) -> None:
        self.entries.append(MemoryEntry("patient", text, turn))

    def recent_context(self, n: int = 6) -> str:
        recent = self.entries[-n:]
        lines = []
        for e in recent:
            prefix = "医生" if e.role == "doctor" else "患者"
            lines.append(f"{prefix}[T{e.turn}]: {e.text}")
        return "\n".join(lines)

    def full_transcript(self) -> list[dict]:
        return [{"role": e.role, "text": e.text, "turn": e.turn} for e in self.entries]
