# Copyright (c) Microsoft. All rights reserved.

"""CLI for invoking the SQL agent and generating predictions for spider_eval.

Usage:
    # Invoke a single question
    python sql_agent_cli.py invoke --llm http://localhost:8088 --question "show me all unique singer names" --db-id concert_singer

    # Generate predictions file for spider_eval
    python sql_agent_cli.py generate_predictions --llm http://localhost:8088 --input data/test_dev.parquet --output data/pred.sql
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests

from sql_agent import SQLAgent, evaluate_query


def get_model_name(llm_endpoint: str, model: Optional[str] = None) -> str:
    """Resolve the model name from the server if not explicitly provided."""
    if model:
        return model
    try:
        resp = requests.get(f"{llm_endpoint}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            resolved = models[0]["id"]
            print(f"Auto-detected model: {resolved}")
            return resolved
    except Exception as e:
        print(f"Warning: could not auto-detect model from {llm_endpoint}/v1/models: {e}")
    return "default"


def resolve_db_path(data_dir: str, db_id: str) -> Optional[str]:
    """Find the SQLite database file for a given db_id."""
    for subdir in ("test_database", "database"):
        path = os.path.join(data_dir, subdir, db_id, f"{db_id}.sqlite")
        if os.path.exists(path):
            return path
    return None


def cmd_invoke(args: argparse.Namespace) -> None:
    """Handle the 'invoke' subcommand: run a single question."""
    db_path = resolve_db_path(args.data_dir, args.db_id)
    if not db_path:
        print(f"Error: Database '{args.db_id}' not found in {args.data_dir}")
        sys.exit(1)

    model_name = get_model_name(args.llm, args.model)
    agent = SQLAgent(
        db=f"sqlite:///{db_path}",
        endpoint=f"{args.llm}/v1",
        verl_replacement={"model": model_name, "temperature": args.temperature},
        max_turns=args.max_turns,
        debug=args.debug,
    )

    print(f"Question: {args.question}")
    print(f"Database: {args.db_id} ({db_path})")
    print()

    result = agent.graph().invoke({"question": args.question})
    print(f"Generated SQL: {result['query']}")

    if args.gold:
        score = evaluate_query(result["query"], args.gold, db_path, raise_on_error=False)
        print(f"Score (exec match): {score}")


def _process_record(
    row: dict,
    idx: int,
    total: int,
    args: argparse.Namespace,
    model_name: str,
) -> tuple[int, str, str]:
    """Process a single record and return (index, query, db_id)."""
    db_id = row["db_id"]
    db_path = resolve_db_path(args.data_dir, db_id)

    if not db_path:
        print(f"[{idx + 1}/{total}] SKIP - Database not found: {db_id}")
        return idx, "ERROR DB NOT FOUND", db_id

    try:
        agent = SQLAgent(
            db=f"sqlite:///{db_path}",
            endpoint=f"{args.llm}/v1",
            verl_replacement={"model": model_name, "temperature": args.temperature},
            max_turns=args.max_turns,
        )
        result = agent.graph().invoke({"question": row["question"]})
        query = result["query"]
    except Exception as e:
        print(f"[{idx + 1}/{total}] ERROR - {db_id}: {e}")
        query = "ERROR AGENT INVOCATION"

    # Collapse multi-line SQL into a single line so the evaluator
    # (which reads one prediction per line) parses it correctly.
    query = " ".join(query.split())
    return idx, query, db_id


def cmd_generate_predictions(args: argparse.Namespace) -> None:
    """Handle the 'generate_predictions' subcommand: produce pred.sql from a parquet dataset."""
    model_name = get_model_name(args.llm, args.model)

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    records = df.to_dict(orient="records")
    total = len(records)

    print(f"Generating predictions for {total} questions -> {args.output} (concurrency={args.concurrency})")
    start = time.time()

    # Pre-allocate results to preserve input order in the output file.
    results: list[tuple[str, str] | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_process_record, row, i, total, args, model_name): i
            for i, row in enumerate(records)
        }

        for future in as_completed(futures):
            idx, query, db_id = future.result()
            results[idx] = (query, db_id)
            completed += 1

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start
                print(f"[{completed}/{total}] {elapsed:.0f}s elapsed")

    with open(args.output, "w") as f:
        for query, db_id in results:  # type: ignore[misc]
            f.write(f"{query}\t{db_id}\n")

    print(f"Done. Predictions saved to {args.output} ({time.time() - start:.0f}s)")

def main() -> None:
    parser = argparse.ArgumentParser(description="SQL Agent CLI")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("VERL_SPIDER_DATA_DIR", "data"),
                        help="Root directory for Spider data (default: data/)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- invoke subcommand ---
    p_invoke = subparsers.add_parser("invoke", help="Run a single question against the SQL agent")
    p_invoke.add_argument("--llm", type=str, required=True)
    p_invoke.add_argument("--question", type=str, required=True)
    p_invoke.add_argument("--db-id", type=str, required=True, help="Spider database ID (e.g. concert_singer)")
    p_invoke.add_argument("--gold", type=str, default=None, help="Gold SQL query to score against")
    p_invoke.add_argument("--model", type=str, default=None)
    p_invoke.add_argument("--temperature", type=float, default=0.0)
    p_invoke.add_argument("--max-turns", type=int, default=5)
    p_invoke.add_argument("--debug", action="store_true")

    # --- generate_predictions subcommand ---
    p_gen = subparsers.add_parser("generate_predictions", help="Generate pred.sql from a parquet dataset")
    p_gen.add_argument("--llm", type=str, required=True)
    p_gen.add_argument("--input", type=str, default="data/test_dev.parquet")
    p_gen.add_argument("--output", type=str, default="pred.sql")
    p_gen.add_argument("--model", type=str, default=None)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--max-turns", type=int, default=5)
    p_gen.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    p_gen.add_argument("--concurrency", type=int, default=3, help="Number of concurrent agent invocations (default: 4)")

    args = parser.parse_args()

    if args.command == "invoke":
        cmd_invoke(args)
    elif args.command == "generate_predictions":
        cmd_generate_predictions(args)


if __name__ == "__main__":
    main()
