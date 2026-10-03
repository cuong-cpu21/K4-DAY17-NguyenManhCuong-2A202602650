from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, edited, and measured."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_01"

    # Initially file does not exist
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # Write profile
    initial_content = "# User Profile\n- **location**: Hue\n- **profession**: backend engineer\n"
    written_path = store.write_text(user_id, initial_content)
    assert written_path.is_file()
    assert store.read_text(user_id) == initial_content
    assert store.file_size(user_id) > 0

    # Edit profile (simulate correction)
    success = store.edit_text(user_id, "Hue", "Da Nang")
    assert success is True
    updated_content = store.read_text(user_id)
    assert "Da Nang" in updated_content
    assert "Hue" not in updated_content

    # Edit non-existing text
    failed = store.edit_text(user_id, "NonExistingText", "Replacement")
    assert failed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction in CompactMemoryManager."""
    cm = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread_id = "thread_compact_test"

    assert cm.compaction_count(thread_id) == 0

    # Append enough messages with tokens exceeding threshold
    for i in range(8):
        cm.append(
            thread_id,
            "user",
            f"Message {i}: Đây là một tin nhắn có độ dài đáng kể nhằm làm tăng số lượng token trong thread.",
        )

    assert cm.compaction_count(thread_id) > 0
    ctx = cm.context(thread_id)
    assert len(ctx["messages"]) <= 3
    assert bool(ctx["summary"]) is True


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced Agent remembers across new threads while Baseline forgets."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "user_cross_test"

    # Session 1: User introduces themselves
    intro_message = "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng và làm backend engineer."
    baseline.reply(user_id=user_id, thread_id="session_1", message=intro_message)
    advanced.reply(user_id=user_id, thread_id="session_1", message=intro_message)

    # Session 2: User asks recall in a BRAND NEW thread
    recall_query = "Mình tên gì và hiện tại mình làm nghề gì?"
    base_reply = baseline.reply(user_id=user_id, thread_id="session_2", message=recall_query)["reply"]
    adv_reply = advanced.reply(user_id=user_id, thread_id="session_2", message=recall_query)["reply"]

    # Advanced MUST remember
    assert "DũngCT" in adv_reply
    assert advanced.memory_file_size(user_id) > 0

    # Baseline MUST NOT remember across sessions
    assert "DũngCT" not in base_reply
    assert baseline.compaction_count("session_2") == 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "user_long_thread"
    thread_id = "thread_stress_comparison"

    for i in range(12):
        msg = (
            f"Lượt {i}: Đây là một chuỗi hội thoại dài với rất nhiều chi tiết ngữ cảnh, "
            f"thông tin công nghệ, bài học vận hành và các ví dụ minh họa phong phú."
        )
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    # Compact must have activated for advanced agent
    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0

    # Prompt load for Advanced must be substantially lower than Baseline
    adv_prompt = advanced.prompt_token_usage(thread_id)
    base_prompt = baseline.prompt_token_usage(thread_id)

    assert adv_prompt < base_prompt, f"Expected advanced ({adv_prompt}) < baseline ({base_prompt})"
