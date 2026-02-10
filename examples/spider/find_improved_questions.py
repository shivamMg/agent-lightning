"""Parse spider output.log to find questions that had reward 0.0 in an earlier rollout
and later achieved reward 1.0, indicating the agent improved on that question.

Usage:
    python find_improved_questions.py [logfile] [outfile]

Defaults:
    logfile = output.log
    outfile = improved_questions.txt
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Rollout:
    rollout_id: str
    question: str = ""
    ground_truth: str = ""
    generated_query: str = ""
    reward: float | None = None
    timestamp: str = ""  # timestamp from the Generated Query log line
    line_order: int = 0  # to preserve chronological order


def parse_log(logfile: str) -> dict[str, list[Rollout]]:
    """Parse the log file and group rollouts by question."""

    # Patterns for extracting rollout fields
    re_question = re.compile(r"\[Rollout (ro-[0-9a-f]+)\] Question:\s*(.*)")
    re_ground_truth = re.compile(r"\[Rollout (ro-[0-9a-f]+)\] Ground Truth:\s*(.*)")
    re_generated = re.compile(r"\[Rollout (ro-[0-9a-f]+)\] Generated Query:\s*(.*)")
    re_reward = re.compile(r"\[Rollout (ro-[0-9a-f]+)\] Reward:\s*([\d.]+)")
    # Trailing source-file references like "  sql_agent.py:503"
    re_source_suffix = re.compile(r"\s+\S+\.py:\d+\s*$")
    # Continuation lines: indented text without a timestamp or log-level prefix
    # These are lines that belong to the previous multi-line field
    re_rollout_line = re.compile(r"\[Rollout (ro-[0-9a-f]+)\]")
    # Timestamp at the start of a log line, e.g. [02/10/26 04:09:37]
    re_timestamp = re.compile(r"^\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]")

    def clean(text: str) -> str:
        """Strip trailing source-file references like '  sql_agent.py:503'."""
        return re_source_suffix.sub("", text).strip()

    rollouts: dict[str, Rollout] = {}  # rollout_id -> Rollout
    line_counter = 0
    current_timestamp = ""  # track the most recent timestamp seen

    # Track which field was last set, for multi-line continuation
    last_field: dict[str, str] = {}  # rollout_id -> "question"|"ground_truth"|"generated_query"
    last_rollout_id: str | None = None

    with open(logfile, encoding="utf-8", errors="replace") as f:
        for line in f:
            line_counter += 1

            # Update the current timestamp if this line has one
            ts_match = re_timestamp.match(line)
            if ts_match:
                current_timestamp = ts_match.group(1)

            # Check for Reward (single line, always)
            m = re_reward.search(line)
            if m:
                rid, reward_str = m.group(1), m.group(2)
                if rid not in rollouts:
                    rollouts[rid] = Rollout(rollout_id=rid, line_order=line_counter)
                rollouts[rid].reward = float(reward_str)
                last_rollout_id = rid
                last_field[rid] = ""
                continue

            # Check for Question
            m = re_question.search(line)
            if m:
                rid, text = m.group(1), clean(m.group(2))
                if rid not in rollouts:
                    rollouts[rid] = Rollout(rollout_id=rid, line_order=line_counter)
                rollouts[rid].question = text
                last_rollout_id = rid
                last_field[rid] = "question"
                continue

            # Check for Ground Truth
            m = re_ground_truth.search(line)
            if m:
                rid, text = m.group(1), clean(m.group(2))
                if rid not in rollouts:
                    rollouts[rid] = Rollout(rollout_id=rid, line_order=line_counter)
                rollouts[rid].ground_truth = text
                last_rollout_id = rid
                last_field[rid] = "ground_truth"
                continue

            # Check for Generated Query
            m = re_generated.search(line)
            if m:
                rid, text = m.group(1), clean(m.group(2))
                if rid not in rollouts:
                    rollouts[rid] = Rollout(rollout_id=rid, line_order=line_counter)
                rollouts[rid].generated_query = text
                rollouts[rid].timestamp = current_timestamp
                last_rollout_id = rid
                last_field[rid] = "generated_query"
                continue

            # If this line contains a [Rollout ro-...] marker for a different field, reset continuation
            if re_rollout_line.search(line):
                last_rollout_id = None
                continue

            # Handle continuation lines (multi-line ground truth / generated query)
            # These are indented lines that don't start with a timestamp or log marker
            if last_rollout_id and last_rollout_id in last_field and last_field[last_rollout_id]:
                stripped = line.rstrip()
                # Skip lines that look like other log entries (timestamps, AgentOps, ERROR, INFO, etc.)
                if stripped and not re.match(r"^\[", stripped) and not stripped.startswith("🖇"):
                    # This is likely a continuation of the previous field
                    field_name = last_field[last_rollout_id]
                    continuation_text = clean(stripped)
                    if continuation_text and not continuation_text.startswith("INFO") and not continuation_text.startswith("ERROR") and not continuation_text.startswith("WARNING"):
                        ro = rollouts[last_rollout_id]
                        if field_name == "question":
                            ro.question += " " + continuation_text
                        elif field_name == "ground_truth":
                            ro.ground_truth += " " + continuation_text
                        elif field_name == "generated_query":
                            ro.generated_query += " " + continuation_text
                        continue
                # If it doesn't look like a continuation, reset
                last_rollout_id = None

    # Group rollouts by question text
    by_question: dict[str, list[Rollout]] = defaultdict(list)
    for ro in rollouts.values():
        if ro.question:
            by_question[ro.question].append(ro)

    # Sort each group by timestamp (ascending), fall back to line_order
    for q in by_question:
        by_question[q].sort(key=lambda r: (r.timestamp, r.line_order))

    return by_question


def find_improved(by_question: dict[str, list[Rollout]]) -> dict[str, list[Rollout]]:
    """Find questions that had reward 0.0 before later getting reward 1.0."""
    improved: dict[str, list[Rollout]] = {}

    for question, rollout_list in by_question.items():
        seen_zero = False
        has_improvement = False
        for ro in rollout_list:
            if ro.reward == 0.0:
                seen_zero = True
            elif ro.reward == 1.0 and seen_zero:
                has_improvement = True
                break

        if has_improvement:
            # Include only rollouts that have a reward
            relevant = [ro for ro in rollout_list if ro.reward is not None]
            if relevant:
                improved[question] = relevant

    return improved


def write_output(improved: dict[str, list[Rollout]], outfile: str) -> None:
    """Write improved questions to a text file."""
    # Sort questions alphabetically for consistent output
    sorted_questions = sorted(improved.keys())

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Questions that improved from reward 0.0 to 1.0 ({len(sorted_questions)} found)\n")
        f.write("=" * 80 + "\n\n")

        for i, question in enumerate(sorted_questions):
            rollout_list = improved[question]
            f.write(f"Question: {question}\n\n")

            for ro in rollout_list:
                f.write(f"Rollout: {ro.rollout_id}\n")
                f.write(f"Timestamp: {ro.timestamp}\n")
                f.write(f"Reward: {ro.reward}\n")
                f.write(f"Ground Truth: {ro.ground_truth}\n")
                f.write(f"Generated Query: {ro.generated_query}\n\n")

            if i < len(sorted_questions) - 1:
                f.write("\n")


def main() -> None:
    logfile = sys.argv[1] if len(sys.argv) > 1 else "output.log"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "improved_questions.txt"

    print(f"Parsing {logfile}...")
    by_question = parse_log(logfile)
    print(f"Found {len(by_question)} unique questions across all rollouts.")

    improved = find_improved(by_question)
    print(f"Found {len(improved)} questions that improved from 0.0 to 1.0.")

    write_output(improved, outfile)
    print(f"Results written to {outfile}")


if __name__ == "__main__":
    main()
