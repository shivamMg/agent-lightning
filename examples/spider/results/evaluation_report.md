# Spider Evaluation Report

## Evaluations

| Model | Customization | Easy (470) | Medium (857) | Hard (463) | Extra (357) | All (2147) |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | None | 62.6% | 46.4% | 33.9% | 22.4% | **43.3%** |
| Qwen2.5-Coder-1.5B-Instruct | RL checkpoint step 32 | 76.6% | 62.8% | 51.6% | 37.0% | **59.1%** |
| Qwen2.5-Coder-1.5B-Instruct | RL checkpoint step 64  | 78.3% | 68.1% | 59.4% | 48.7% | **65.3%** |

## Key Takeaways

- Reinforcement learning on Qwen2.5-Coder-1.5B-Instruct with Agent Lightning improved overall execution accuracy from **43.3% → 65.3%** (+22 percentage points).
- The biggest gains are on harder queries — the "extra hard" category more than doubled from 22.4% to 48.7%.
- Longer training (step 192 vs step 32) continues to improve results, especially on hard/extra categories.

## Overview

These results measure **SQL execution accuracy** — the fraction of predicted SQL queries that produce the same result set as the gold (correct) SQL query when executed against the actual database.

## Difficulty Levels

| Difficulty | Description |
|---|---|
| **Easy** | ≤1 component (WHERE/GROUP BY/ORDER BY/JOIN), no aggregations, no subqueries |
| **Medium** | A few components (e.g., 1–2 clauses) |
| **Hard** | Multiple components or 1 subquery |
| **Extra** | Many components and/or multiple subqueries |
| **All** | Weighted average across all test questions |


