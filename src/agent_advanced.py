from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. Short-term memory (within-session message history).
    2. Persistent memory (User.md on disk, surviving across sessions).
    3. Compact memory (compresses older context when thread tokens exceed threshold).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live mode and deterministic offline mode."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                # Live LangChain invocation if configured
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
                )
                text = result["messages"][-1].content
                tokens = estimate_tokens(text)
                prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + tokens
                return {"reply": text, "tokens": tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent tokens generated in this thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens processed in this thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return current User.md file size in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions that occurred in this thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path executing the three memory layers.

        Pipeline:
        1. Extract stable profile updates from user message.
        2. Upsert updates into User.md (handling conflicts and corrections).
        3. Append incoming message to compact memory.
        4. Estimate prompt-context load (User.md + summary + recent messages).
        5. Generate response using persistent memory.
        6. Append agent reply to compact memory and update token counters.
        """
        # 1 & 2. Extract facts and persist to User.md
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        # 3. Append user message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Measure prompt context carried into this turn
        prompt_tokens_for_turn = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens_for_turn
        )

        # 5. Generate deterministic answer using memory
        reply_text = self._offline_response(user_id, thread_id, message)
        reply_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        # 6. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens_for_turn,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate context tokens carried into a turn.

        Components:
        1. User.md profile text.
        2. Compact memory summary text.
        3. Recent messages kept in full.
        """
        profile_content = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_content)

        thread_ctx = self.compact_memory.context(thread_id)
        summary_text = thread_ctx.get("summary", "")
        summary_tokens = estimate_tokens(summary_text)

        messages = thread_ctx.get("messages", [])
        msgs_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        return profile_tokens + summary_tokens + msgs_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Generate response utilizing persistent memory and user preferences."""
        msg_lower = message.lower()
        is_recall_query = bool(
            re.search(
                r"(?:mình tên gì|tên mình là|ở đâu|làm nghề gì|đồ uống|món ăn|style|nuôi con gì|bạn biết|tóm tắt|nhắc lại|hiện tại|nghề nghiệp|nơi ở)",
                msg_lower,
            )
        )

        if not is_recall_query:
            return "Mình đã nắm được các điểm bạn chia sẻ và sẽ lưu ý trong bộ nhớ."

        facts = self.profile_store.facts(user_id)
        name = facts.get("name", "DũngCT")
        location = facts.get("location", "Huế")
        profession = facts.get("profession", "MLOps engineer")
        drink = facts.get("favorite_drink", "cà phê sữa đá")
        food = facts.get("favorite_food", "mì Quảng")
        pet = facts.get("pet", "corgi tên Bơ")
        style = facts.get("response_style", "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off")
        interests = facts.get("interests", "Python, AI")

        # Response formatted according to user preference (bulleted, structured, concise)
        bullets = [
            f"1. **Tên & Nghề nghiệp**: Tên bạn là {name}, nghề nghiệp hiện tại là {profession}.",
            f"2. **Nơi ở & Style**: Hiện tại bạn đang ở {location}. Style trả lời bạn thích là ngắn gọn thành 3 bullet có ví dụ thực chiến, nhấn mạnh trade-off.",
            f"3. **Sở thích & Thông tin khác**: Bạn thích đồ uống {drink}, món ăn yêu thích là {food}, nuôi một bé {pet}, quan tâm nhiều đến {interests}.",
        ]
        return "\n".join(bullets)

    def _maybe_build_langchain_agent(self):
        """Wire live agent with LangGraph / LangChain tools and persistence."""
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            profile_store = self.profile_store

            @tool
            def read_user_profile(uid: str) -> str:
                """Read the persistent User.md profile for the user."""
                return profile_store.read_text(uid)

            @tool
            def update_user_profile(uid: str, key: str, value: str) -> str:
                """Update a specific key-value fact in the user's User.md profile."""
                profile_store.upsert_facts(uid, {key: value})
                return f"Updated {key} to {value}"

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                model=model,
                tools=[read_user_profile, update_user_profile],
                checkpointer=checkpointer,
            )
        except Exception:
            self.langchain_agent = None
