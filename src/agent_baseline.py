from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session short-term memory only (keyed strictly by thread_id).
    - No persistent memory (no User.md).
    - No compact memory (compaction count is always 0).
    - Forgets long-term facts across new threads/sessions.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live agent if available and configured, otherwise use deterministic offline path."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                # Live invocation if configured
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                text = result["messages"][-1].content
                tokens = estimate_tokens(text)
                prompt_tokens = self.prompt_token_usage(thread_id) + estimate_tokens(message)
                return {"reply": text, "tokens": tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate cumulative prompt context this baseline agent processed."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory; compactions count is always 0."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline response generation within a single session."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Calculate prompt context for this turn: all prior messages in session + current message
        prior_context = "".join(m["content"] for m in session.messages) + message
        prompt_tokens_for_turn = estimate_tokens(prior_context)
        session.prompt_tokens_processed += prompt_tokens_for_turn

        # Add user message to session history
        session.messages.append({"role": "user", "content": message})

        # Generate response based ONLY on messages in THIS thread
        reply_text = self._generate_session_reply(session.messages, message)
        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens

        # Record assistant reply
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens_for_turn,
        }

    def _generate_session_reply(self, history: list[dict[str, str]], current_message: str) -> str:
        """Answer queries strictly from the messages present in this session."""
        # Check if user is asking recall questions
        msg_lower = current_message.lower()
        is_recall_query = bool(
            re.search(
                r"(?:mình tên gì|tên mình là gì|ở đâu|làm nghề gì|đồ uống|món ăn|style|nuôi con gì|bạn biết|tóm tắt)",
                msg_lower,
            )
        )

        if is_recall_query:
            # Look backwards in this thread's history for any declared facts
            # Note: history already includes current_message as last element
            all_text = " ".join(m["content"] for m in history[:-1])

            # Check if name was mentioned in this session
            name_match = re.search(r"(?:tên là|mình tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+)", all_text)
            if not all_text.strip() or not name_match:
                return "Chào bạn, đây là phiên làm việc mới nên mình chưa có bất kỳ thông tin nào về bạn."

            # If user did provide facts in this session, answer using them
            name = name_match.group(1) if name_match else "bạn"
            return f"Trong phiên này, bạn có chia sẻ tên bạn là {name}."

        return "Mình đã nhận được thông tin từ bạn và ghi nhận trong phiên hiện tại."

    def _maybe_build_langchain_agent(self):
        """Optionally build a LangGraph / LangChain agent with InMemorySaver."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                model=model,
                tools=[],
                checkpointer=checkpointer,
            )
        except Exception:
            self.langchain_agent = None
