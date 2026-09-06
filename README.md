# GenPark AI Agent Skill - Speculative Decoding Draft Acceptance Evaluator

Implements rejection sampling and token verification routines to accelerate large LLM inference using fast draft models.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[Small Draft Model Proposals: K Tokens] --> B[Target Model Single-Pass Parallel Scoring]
    B --> C[Compute Acceptance Ratio: min 1, p / q]
    C --> D{Ratio >= Acceptance Criterion?}
    D -->|Yes| E[Accept Draft Token Instantly]
    D -->|No, Rejection| F[Sample Corrected Token & End Speculative Burst]
    E --> G[3-5x Generative Inference Acceleration]
```

## Features
- **Rejection Sampling Correctness**: Guarantees target distribution output mathematical equivalence.
- **Zero External Dependencies**: Pure Python 3.9+ standard library.
