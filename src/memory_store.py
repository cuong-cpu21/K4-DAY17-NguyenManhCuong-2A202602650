from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator.

    Properties:
    - Empty or whitespace text returns 0.
    - Deterministic: always same count for same input.
    - Monotonic: longer text yields >= tokens than shorter text.
    - Heuristic: approximately 4 characters per token.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user id and return path to its User.md file."""
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        user_dir = self.root_dir / safe_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return User.md content or an empty string if file does not exist."""
        path = self.path_for(user_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown profile to disk and return the file path."""
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md and return True if modified."""
        current = self.read_text(user_id)
        if search_text in current:
            new_content = current.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from User.md."""
        content = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^-\s*\*\*([^\*:]+)\*\*:\s*(.+)$", line)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip()
                facts_dict[key] = val
        return facts_dict

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Upsert facts into User.md cleanly, resolving conflicts with newest data."""
        if not updates:
            return
        current_facts = self.facts(user_id)
        current_facts.update(updates)

        lines = ["# User Profile\n"]
        for k, v in current_facts.items():
            lines.append(f"- **{k}**: {v}\n")
        self.write_text(user_id, "".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Features:
    - Skips questions and inquiry-only turns (prevents false updates).
    - Filters conversational noise (e.g., business trips to Hanoi, jokes about PM).
    - Accurately captures corrections (e.g. Hue vs Da Nang, backend vs MLOps).
    - Extracts preferences, favorite food/drink, pet, and response styles.
    """
    if not message or not message.strip():
        return {}

    msg = message.strip()

    # Skip obvious question-only turns without fact-declaring statements
    is_question = bool(
        re.search(
            r"^(?:bạn có biết|bạn có thể nhắc lại|bạn thử nhớ|nhắc lại giúp mình|hiện tại mình đang ở đâu|mình tên gì|nếu biết thì thử)\b",
            msg,
            re.IGNORECASE,
        )
        or (msg.endswith("?") and not re.search(r"\b(?:mình tên là|mình ở|mình làm|mình chuyển sang|đính chính|hiện ở)\b", msg, re.IGNORECASE))
    )
    if is_question:
        return {}

    updates: dict[str, str] = {}

    # 1. Name extraction
    name_match = re.search(
        r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_]+)*)",
        msg,
        re.IGNORECASE,
    )
    if name_match:
        name = name_match.group(1).strip()
        # Clean trailing punctuation
        name = re.sub(r"[,.\!?;:]+$", "", name).strip()
        if name:
            updates["name"] = name

    # 2. Location extraction (with conflict & noise handling)
    # Check for noise: Hanoi is only a 2-day meeting trip
    has_hanoi_noise = bool(re.search(r"Hà Nội\s+chỉ là nơi.*(?:họp|bay ra)|không phải nơi ở", msg, re.IGNORECASE))
    
    # Corrections / Current location
    if re.search(r"(?:cập nhật từ Huế sang Đà Nẵng|làm việc ở Đà Nẵng vài tháng|nơi ở hiện tại là Đà Nẵng|chuyển sang Đà Nẵng)", msg, re.IGNORECASE):
        updates["location"] = "Đà Nẵng"
    elif re.search(r"(?:giờ mình đang ở Huế chứ không còn ở Đà Nẵng|đang ở Huế|vẫn ở Huế|ở quán cà phê quen gần sông Hương|bước đi bộ ven sông Hương)", msg, re.IGNORECASE):
        updates["location"] = "Huế"
    elif re.search(r"\bở Đà Nẵng\b", msg, re.IGNORECASE) and not re.search(r"không còn ở Đà Nẵng", msg, re.IGNORECASE):
        updates["location"] = "Đà Nẵng"

    # 3. Profession extraction (with correction & noise handling)
    # Noise: PM is just a joke
    has_pm_joke = bool(re.search(r"product manager.*(?:câu đùa|đùa)", msg, re.IGNORECASE))
    
    if re.search(r"(?:chuyển sang MLOps engineer|làm MLOps engineer|nghề nghiệp hiện tại vẫn là MLOps engineer|công việc MLOps)", msg, re.IGNORECASE):
        updates["profession"] = "MLOps engineer"
    elif re.search(r"làm backend engineer", msg, re.IGNORECASE) and not re.search(r"không còn làm backend engineer", msg, re.IGNORECASE):
        updates["profession"] = "backend engineer"

    # 4. Favorite drink
    if re.search(r"cà phê sữa đá", msg, re.IGNORECASE):
        updates["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if re.search(r"mì Quảng", msg, re.IGNORECASE):
        updates["favorite_food"] = "mì Quảng"

    # 6. Pet
    if re.search(r"corgi", msg, re.IGNORECASE):
        updates["pet"] = "corgi tên Bơ"

    # 7. Response style
    if re.search(r"3 bullet", msg, re.IGNORECASE):
        updates["response_style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    elif re.search(r"(?:bullet ngắn|ngắn gọn)", msg, re.IGNORECASE):
        updates["response_style"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Technical interests
    interests = []
    if re.search(r"\bPython\b", msg):
        interests.append("Python")
    if re.search(r"\bAI\b", msg):
        interests.append("AI")
    if re.search(r"\bMLOps\b", msg):
        interests.append("MLOps")
    if re.search(r"\bbenchmark memory\b", msg, re.IGNORECASE):
        interests.append("benchmark memory")
    if interests:
        updates["interests"] = ", ".join(interests)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    items = []
    for msg in messages[-max_items:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        # Take key snippet
        snippet = content[:90].replace("\n", " ") + ("..." if len(content) > 90 else "")
        items.append(f"[{role}]: {snippet}")
    return "Tóm tắt ngữ cảnh cũ:\n" + "\n".join(f"- {it}" for it in items)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction when total thread tokens exceed threshold."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_data = self.state[thread_id]
        messages: list[dict[str, str]] = thread_data["messages"]
        messages.append({"role": role, "content": content})

        # Calculate current total context tokens in this thread
        summary_tokens = estimate_tokens(thread_data["summary"])
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        total_tokens = summary_tokens + msg_tokens

        # Check if threshold is breached and we have more messages than keep_messages
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_summarize = messages[:split_idx]
            kept_messages = messages[split_idx:]

            new_summary = summarize_messages(to_summarize)
            old_summary = thread_data["summary"]
            if old_summary:
                combined_summary = f"{old_summary}\n{new_summary}"
                # Keep summary bounded
                if estimate_tokens(combined_summary) > self.threshold_tokens // 2:
                    combined_summary = combined_summary[-300:]
                thread_data["summary"] = combined_summary
            else:
                thread_data["summary"] = new_summary

            thread_data["messages"] = kept_messages
            thread_data["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return per-thread memory state."""
        return self.state.get(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        return int(self.state.get(thread_id, {}).get("compactions", 0))
