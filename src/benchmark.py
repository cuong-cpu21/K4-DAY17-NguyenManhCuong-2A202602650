from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from tabulate import tabulate


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for item in expected if item.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    elif matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Calculate a lightweight quality score for offline mode (0.0 - 1.0)."""
    if not answer or not answer.strip():
        return 0.0
    score = 0.5
    # Reward structured response (bullets, lists, paragraphs)
    if "\n" in answer or "bullet" in answer.lower() or "1." in answer or "- " in answer:
        score += 0.2
    # Reward presence of expected factual answers
    if expected:
        ans_lower = answer.lower()
        match_ratio = sum(1 for it in expected if it.lower() in ans_lower) / len(expected)
        score += 0.3 * match_ratio
    return min(1.0, round(score, 2))


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions."""
    thread_ids: list[str] = []
    user_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        user_ids.add(user_id)
        conv_thread = f"{conv['id']}_main"
        thread_ids.append(conv_thread)

        # 1. Feed turns sequentially
        for turn in conv["turns"]:
            agent.reply(user_id=user_id, thread_id=conv_thread, message=turn)

        # 2. Ask recall questions in a FRESH thread (measuring cross-session recall)
        recall_questions = conv.get("recall_questions", [])
        if recall_questions:
            recall_thread = f"{conv['id']}_recall"
            thread_ids.append(recall_thread)
            for rq in recall_questions:
                question = rq["question"]
                expected = rq["expected_contains"]
                resp = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
                answer = resp["reply"]

                r_score = recall_points(answer, expected)
                q_score = heuristic_quality(answer, expected)
                recall_scores.append(r_score)
                quality_scores.append(q_score)

    total_agent_tokens = sum(agent.token_usage(t) for t in thread_ids)
    total_prompt_tokens = sum(agent.prompt_token_usage(t) for t in thread_ids)
    total_compactions = sum(agent.compaction_count(t) for t in thread_ids)

    # Memory growth in bytes
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in user_ids)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 4),
        response_quality=round(avg_quality, 4),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table with required columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality:.2f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("================================================================================")
    print("                      AI AGENT MEMORY SYSTEMS BENCHMARK                         ")
    print("================================================================================\n")

    # 1. Standard Benchmark
    print("### Standard Benchmark (`data/conversations.json`)\n")
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    std_conversations = load_conversations(standard_path)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    baseline_std_row = run_agent_benchmark("Baseline", baseline_std, std_conversations, config)

    advanced_std = AdvancedAgent(config=config, force_offline=True)
    advanced_std_row = run_agent_benchmark("Advanced", advanced_std, std_conversations, config)

    print(format_rows([baseline_std_row, advanced_std_row]))
    print("\n")

    # 2. Long-Context Stress Benchmark
    print("### Long-Context Stress Benchmark (`data/advanced_long_context.json`)\n")
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    stress_conversations = load_conversations(stress_path)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    baseline_stress_row = run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config)

    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    advanced_stress_row = run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config)

    print(format_rows([baseline_stress_row, advanced_stress_row]))
    print("\n================================================================================")


if __name__ == "__main__":
    main()
