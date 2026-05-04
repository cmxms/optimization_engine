# Optimization Engine

An autonomous, multi-agent quantitative strategy development lab.

## Core Pipeline
For a detailed explanation of every step in the optimization process, see the [Detailed Pipeline Documentation](PIPELINE.md).

## Project Overview
The Optimization Engine uses a stack of specialized LLM agents and a high-performance vectorized quant engine to parse, audit, optimize, and stress-test Pine Script strategies.

- **Developer**: Parses Pine Script into Intermediate Representation (IR).
- **Logic Critic**: Audits code for repainting and future leaks.
- **Quant Engine**: Runs 5-Fold Walk-Forward Analysis and Monte Carlo simulations.
- **Strategist**: Synthesizes logic and math into a final "Hard Truth" verdict.
