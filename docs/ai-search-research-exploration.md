# AI Search Research Exploration

## Overview

This document summarizes our exploration of learned search agents and agentic retrieval systems, with a focus on training small models to iteratively search until they have sufficient context for answer generation.

## The Problem with Traditional RAG

Traditional RAG has limitations:
- **Static retrieval**: Single query, single retrieval, no iteration
- **Chunk size matters more than algorithm**: 512T → 4096T drops HIT@1 by ~15pp
- **Chunker algorithm differences are small**: Only ~2-5pp between different chunkers
- **Reranking just reshuffles**: If the right chunk wasn't retrieved, reranking won't help
- **Hybrid search diminishing returns**: Modern embedding models handle keywords well

## The Opportunity: Learned Search Agents

Train a model to iteratively search until it has sufficient context, then hand off to a generator.

### Proposed Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Search Agent      │         │     Generator       │
│   (Small, 3-7B)     │────────▶│    (Large, 70B+)    │
│                     │ handoff │                     │
│ - Iterative search  │         │ - Final answer      │
│ - Query formulation │         │ - One-shot          │
│ - "Do I have enough?"│        │                     │
└─────────────────────┘         └─────────────────────┘
```

### Key Insight: Reward Without Generation

For QA datasets with ground truth answers, we can compute rewards without generation:

```python
# Reward = does collected context contain the answer?
reward = 1.0 if answer in collected_context else 0.0
```

This sidesteps credit assignment problems and makes training cheaper.

---

## Research Landscape

### Survey Papers

| Paper | Link | Key Contribution |
|-------|------|------------------|
| Agentic RAG Survey | [arXiv:2501.09136](https://arxiv.org/abs/2501.09136) | Taxonomy of agentic RAG architectures |
| Reasoning Agentic RAG | [arXiv:2506.10408](https://arxiv.org/html/2506.10408v1) | System 1 vs System 2 framing |

### RL-Trained Search Agents

| Paper | Approach | Reward Signal |
|-------|----------|---------------|
| **Search-R1** | Interleaves reasoning + search, masks retrieved tokens | Outcome (answer correctness) |
| **DeepRetrieval** | Single-turn query rewriting via RL | Retrieval recall |
| **HiPRAG** | Hierarchical process rewards per search decision | Fine-grained step rewards |

### Iterative Retrieval (Not Jointly Trained)

| Paper | Mechanism |
|-------|-----------|
| FLARE | Predicts next sentence, retrieves when confidence low |
| Self-RAG | Reflection tokens decide when to retrieve |
| DRAGIN | Token-level entropy triggers retrieval |
| IRCoT | Retrieves at each chain-of-thought step |

### Two-Model Architectures

| Paper | Architecture |
|-------|--------------|
| Speculative RAG | Small drafter + large verifier (parallel drafts, not iterative) |
| DRAG | Distills RAG from large → small (replaces large model entirely) |

---

## Search-R1 Deep Dive

### How It Works

```
LLM generates → hits <search>query</search> → system retrieves top-3 passages
→ wraps in <information>...</information> → appends to context → LLM continues
→ repeats up to 4 search calls → terminates at <answer>...</answer>
```

### Key Technical Details

| Component | Value |
|-----------|-------|
| Objective | `max E[r(x,y)] - β·KL[π_θ \|\| π_ref]` |
| Reward | Exact match: `r = EM(pred, gold)` |
| RL Algorithm | PPO or GRPO |
| Learning Rate | 1e-6 (policy), 1e-5 (value) |
| Retriever | E5 embeddings, top-k=3 |
| Max Search Calls | 4 per question |

### Critical Innovation: Retrieved Token Masking

```python
I(y_t) = 1  # for LLM-generated tokens → included in loss
I(y_t) = 0  # for retrieved text → masked out
```

Without masking: 0.343 EM → With masking: 0.431 EM (**+25.7%**)

### Results

| Model | Avg EM | vs RAG Baseline |
|-------|--------|-----------------|
| Qwen2.5-3B | 0.303 | +20% |
| Qwen2.5-7B | 0.431 | +41% |
| Qwen2.5-14B | 0.479 | +47% |

### Ablation Findings

1. **PPO vs GRPO**: GRPO converges faster but can collapse; PPO more stable
2. **Base vs Instruct**: Base catches up after RL training
3. **Passages**: top-k=3 optimal; k=5 introduces noise
4. **Response length**: Shortens early (removing filler), lengthens later (more search calls)

---

## DeepRetrieval Deep Dive

### Comparison to Search-R1

| | DeepRetrieval | Search-R1 |
|---|---------------|-----------|
| **Turns** | Single-turn | Multi-turn iterative |
| **What it learns** | Query reformulation | Query + when to stop |
| **Reward** | Retrieval recall | Answer correctness |
| **Output** | Better query | Sufficient context |

### Training Setup

| Spec | Value |
|------|-------|
| GPUs | 2× NVIDIA A100 80GB |
| Epochs | 5 |
| Batch size | 64 samples/iteration |
| Training time | ~5 days |
| Convergence | ~1400-1800 steps |

### Reward Function

```python
recall >= 0.7  →  +5.0
recall >= 0.5  →  +4.0
recall >= 0.4  →  +3.0
recall >= 0.3  →  +1.0
recall >= 0.1  →  +0.5
recall >= 0.05 →  +0.1
recall < 0.05  →  -3.5
format error   →  -4.0
```

### Results

- 3B model beats GPT-4o on 11/13 retrieval datasets
- PubMed: 65% recall (vs 25% baseline)

---

## DeepRetrieval Codebase Analysis

### Repository Structure

```
DeepRetrieval/
├── code/
│   ├── verl/                    # Modified veRL framework
│   │   ├── trainer/main_ppo.py  # Training loop integration
│   │   ├── utils/reward_score/  # Reward functions per dataset
│   │   └── workers/             # Actor, critic, rollout workers
│   │
│   ├── src/                     # Retrieval implementations
│   │   ├── Dense/               # Dense retrieval + indexing
│   │   └── Lucene/              # BM25 search
│   │
│   └── data_preprocess/         # Multi-domain data preparation
```

### Training Loop Integration

```python
class RewardManager():
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'])

        for i in range(len(data)):
            # Decode tokens to string
            sequences_str = self.tokenizer.decode(sequences)

            # Get ground truth and compute reward
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(sequences_str, ground_truth, api)

            # Place reward at last token
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
```

### Adding a Custom Task

1. Create `verl/utils/reward_score/your_task.py` with `compute_score()` function
2. Add `elif` in `_select_rm_score_fn()` to return your function
3. Prepare data with `ground_truth` field

---

## Our Proposed Approach

### Architecture

```
Small Search Agent (3-7B)          Large Generator (70B+)
├─ Iterative search                ├─ Final answer only
├─ Query formulation               ├─ Sees collected context
├─ "Do I have enough?" decision    └─ One-shot generation
└─ Handoff when sufficient
```

### Key Differences from Existing Work

| Aspect | DeepRetrieval | Search-R1 | Our Approach |
|--------|---------------|-----------|--------------|
| Turns | Single | Multi | Multi |
| Generator | Same model | Same model | Separate large model |
| Reward | Recall | Answer EM | Answer in context |
| Training | Train query rewriter | Train full pipeline | Train search agent only |

### Reward Function

```python
def compute_reward(collected_context, ground_truth_answer):
    if ground_truth_answer.lower() in collected_context.lower():
        return 1.0  # Answer found in context
    return 0.0      # Answer not found
```

### Advantages

1. **No generation during training** - cheaper, faster
2. **Clean reward signal** - no credit assignment issues
3. **Decoupled** - can swap generator at inference
4. **Efficient** - small model does iterations, large model runs once

---

## Implementation Plan

### Phase 1: Learn from DeepRetrieval

- Clone and understand codebase ✓
- Run on a simple dataset
- Understand veRL integration

### Phase 2: Adapt for Multi-Turn

- Modify rollout to support iterative search
- Add stopping policy (special token or confidence)
- Implement answer-containment reward

### Phase 3: Two-Model Architecture

- Separate search agent training
- Implement handoff mechanism
- Integrate with large generator for inference

---

## Resources

### Code Repositories

- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval) - Cloned to `/Users/bhavnick/Workspace/DeepRetrieval`
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [veRL](https://github.com/volcengine/verl)
- [Agentic RAG Survey](https://github.com/asinghcsu/AgenticRAG-Survey)

### Papers

- Search-R1: [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- DeepRetrieval: [arXiv:2503.00223](https://arxiv.org/abs/2503.00223)
- HiPRAG: [arXiv:2510.07794](https://arxiv.org/abs/2510.07794)
- Speculative RAG: [arXiv:2407.08223](https://arxiv.org/abs/2407.08223)

### Training Logs

- [DeepRetrieval W&B Report](https://wandb.ai/patjj/literature_search/reports/DeepRetrieval-Training-Report-on-PubMed-Search--VmlldzoxMTU5MDk1MA)

---

## Deployment Latency Analysis

### Small Model Inference Speed

Based on Qwen 3B-4B benchmarks on consumer/prosumer hardware:

| Hardware | Tokens/sec | Notes |
|----------|------------|-------|
| RTX 4090 | ~200-300 | FP16 inference |
| A100 40GB | ~300-400 | Production server |
| M2 Max | ~100-150 | Apple Silicon |

### Query Generation Latency

Search queries are typically short (10-20 tokens):

```
Query length: 10-20 tokens
Generation speed: 200-300 tok/s
Query generation time: ~40-80ms
```

**Key insight**: Unlike long-form generation, search queries are quick because they're short.

### Full Iteration Breakdown

| Component | Latency | Notes |
|-----------|---------|-------|
| TTFT (time to first token) | 20-50ms | Depends on context length |
| Query generation | 40-80ms | 10-20 tokens at 200-300 tok/s |
| Retrieval (local) | 10-50ms | Vector DB lookup |
| Retrieval (API) | 100-500ms | Network latency dominant |
| **Total (local retrieval)** | **~100-200ms** | Per search iteration |
| **Total (API retrieval)** | **~200-600ms** | Per search iteration |

### End-to-End Estimates

| Scenario | Iterations | Local Retrieval | API Retrieval |
|----------|------------|-----------------|---------------|
| Simple query | 1 | ~100-200ms | ~200-600ms |
| Multi-hop (typical) | 3 | ~300-600ms | ~600ms-1.8s |
| Complex (max) | 4-5 | ~400-800ms | ~1-3s |

### Training vs Inference Note

DeepRetrieval's 5-day training time was dominated by **PubMed API latency**, not model inference:
- Each training sample requires real API calls to compute recall
- API rate limiting (10 req/s) creates bottleneck
- With local retrieval index, training would be much faster

**Implication**: For our approach, using a pre-built local index during training would significantly reduce training time.

---

## Open Questions

1. ~~**Latency at deployment**~~ - Answered above: ~100-200ms per iteration with local retrieval
2. **Stopping policy** - How to train "I have enough" decision?
3. **Tool calling format** - Can we use native tool calling instead of XML?
4. **Local vs API retrieval** - Trade-off between training speed and real-world performance
