# Spider Example

[![spider CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml)

This example demonstrates how to train a text-to-SQL agent on the Spider dataset using Agent-Lightning with reinforcement learning. It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example depends on LangChain v0.x and several SQL-related libraries. Install the required dependencies with:

```bash
pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" "sqlparse" "nltk"
```

Additionally, follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and VERL-related dependencies.

## Dataset

Detailed dataset preparation instructions are available in the [How to Train a SQL Agent](../../docs/how-to/train-sql-agent.md) guide.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `train_sql_agent.py` | Training script for SQL agents with support for multiple model configurations (Qwen, LLaMA, fast mode for CI) |
| `sql_agent.py` | SQL agent implementation using LangGraph and LangChain, with debugging capabilities |
| `data/` | Directory containing the Spider dataset files |
| `spider_eval/` | Evaluation utilities for assessing SQL agent performance |

## Running Examples

### Training

Train a SQL agent using the Qwen2.5-Coder-1.5B-Instruct model with the following command. This requires a single node with at least one 40GB GPU:

```bash
python train_sql_agent.py qwen
```

If you want to use an NPU for training, please refer to the **Launch Training with NPUS** section in [How to Train a SQL Agent](../../docs/how-to/train-sql-agent.md).

### Debugging

To test and debug the SQL agent interactively:

```bash
python sql_agent.py
```

This command requires an OpenAI-compatible API service. Configure your service endpoint and credentials using the `OPENAI_API_BASE` and `OPENAI_API_KEY` environment variables.


### Serve Checkpoint

```shell
python fsdp_to_hf.py --fsdp-model-path checkpoints/AgentLightning/spider/global_step_32/actor/model_world_size_1_rank_0.pt --hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct --hf-tokenizer-path checkpoints/AgentLightning/spider/global_step_32/actor/huggingface --output-dir checkpoints/AgentLightning/spider/global_step_32/hf_model

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model checkpoints/AgentLightning/spider/global_step_32/hf_model --host 0.0.0.0 --port 8088
# For base model
# CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-1.5B-Instruct --host 0.0.0.0 --port 8088

python sql_agent_cli.py invoke --llm http://localhost:8088 --question "show me all unique singer names" --db-id concert_singer

python sql_agent_cli.py generate_predictions --llm http://localhost:8088 --input data/test.parquet --output data/pred.sql --concurrency 3

python -c "import nltk; nltk.download('punkt_tab')"

python -m spider_eval.evaluation --gold data/test_gold.sql --pred data/pred.sql --db data/test_database --etype exec
```

### Base model

```
                     easy                 medium               hard                 extra                all
count                470                  857                  463                  357                  2147
=====================   EXECUTION ACCURACY     =====================
execution            0.626                0.464                0.339                0.224                0.433
```

### FT model (checkpoint step 32)

```
                     easy                 medium               hard                 extra                all
count                470                  857                  463                  357                  2147
=====================   EXECUTION ACCURACY     =====================
execution            0.766                0.628                0.516                0.370                0.591
```

### FT model (checkpoint step 192)

```
                     easy                 medium               hard                 extra                all
count                470                  857                  463                  357                  2147
=====================   EXECUTION ACCURACY     =====================
execution            0.783                0.681                0.594                0.487                0.653
```