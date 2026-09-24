# DDQN<sub>DS</sub>-GA: Efficient Mutation Test Data Generation via DDQN with Dual-Stream Feature Encoding

A framework for mutation test data generation that combines **Double Deep Q-Network (DDQN)** with **dual-stream feature encoding** and a **Genetic Algorithm (GA)** with adaptive operator selection.

## Overview

Mutation testing evaluates test suite quality by injecting small syntactic changes (mutants) into programs and checking whether tests can detect them. This framework addresses the computational cost bottleneck by:

1. **Group-level optimization**: Organizing homologous mutants as the optimization unit, reducing redundant computation
2. **K-Means clustering**: Preprocessing all groups, extracting feature vectors, and clustering to select representative groups for RL training
3. **Adaptive operator selection**: Using DDQN to dynamically select GA operators (crossover/mutation strategies) based on real-time search state
4. **Dual-stream state representation**: Encoding both search progress features and mutant characteristics for informed decision-making
5. **STPE strategy**: Stagnation-Triggered Potential Exploration via SimHash to escape local optima

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DDQN_DS-GA Framework                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Preprocessing & Clustering                            │  │
│  │  1. Run g' generations of baseline GA on each group    │  │
│  │  2. Extract Unkill = [Rate_kill, div, imp_dis, Rate_stag] │
│  │  3. K-Means clustering → select representative groups  │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           ▼                                  │
│  ┌──────────────┐    ┌──────────────────────────────┐        │
│  │  Search State │    │   Mutant Feature Vector      │        │
│  │  (5-dim)      │    │   (7-dim)                    │        │
│  │  ┌──────────┐ │    │   ┌───────────────────────┐  │        │
│  │  │stagnation│ │    │   │avg_complexity         │  │        │
│  │  │min_dist  │ │    │   │operator_ratio (×6)    │  │        │
│  │  │avg_dist  │ │    │   └───────────────────────┘  │        │
│  │  │diversity │ │    └──────────┬───────────────────┘        │
│  │  │progress  │ │               │                            │
│  │  └────┬─────┘ │               │                            │
│  └───────┼───────┘               │                            │
│          │         ┌─────────────┘                            │
│          ▼         ▼                                          │
│     ┌─────────────────────┐                                   │
│     │   Dual-Stream DDQN  │                                   │
│     │   Q-Network         │──► Action (a0/a1/a2)              │
│     └─────────────────────┘         │                         │
│                                     ▼                         │
│     ┌──────────────────────────────────────────┐              │
│     │  Genetic Algorithm (per mutant group)    │              │
│     │  a0: SBX crossover (exploitation)        │              │
│     │  a1: Uniform + Gaussian (balanced)       │              │
│     │  a2: Large Gaussian (exploration)        │              │
│     │  τ(a): τ_long for a0,a1; τ_short for a2 │              │
│     └──────────────────────────────────────────┘              │
│                          │                                    │
│                          ▼                                    │
│     ┌──────────────────────────────────────────┐              │
│     │  STPE: SimHash-based exploration bonus   │              │
│     │  (activated on search stagnation;        │              │
│     │   visit counter resets each trigger)     │              │
│     └──────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
DDQN_DS-GA/
├── ddqn_ds_ga/              # Core framework library
│   ├── __init__.py
│   ├── config.py            # Centralized configuration & ablation switches
│   ├── subject.py           # Abstract interface for test subjects
│   ├── network.py           # Q-network architectures (dual/single stream)
│   ├── agent.py             # Double DQN agent & replay buffer
│   ├── operators.py         # GA operators (SBX, uniform crossover, Gaussian)
│   ├── features.py          # Mutant feature extraction
│   ├── curiosity.py         # STPE strategy & SimHash encoder
│   ├── clustering.py        # K-Means clustering & preprocessing
│   ├── ga.py                # Group-level GA optimizer
│   └── runner.py            # Two-phase experiment orchestration
├── examples/
│   └── triangle/            # Triangle classification example
│       ├── program.py       # Original program & 16 strong mutants
│       ├── subject.py       # SubjectInterface implementation
│       └── run.py           # Entry point with CLI arguments
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DDQN_DS-GA.git
cd DDQN_DS-GA

# Install dependencies
pip install numpy torch
```

## Quick Start

```bash
# Run with full framework (DDQN + dual-stream + STPE)
python -m examples.triangle.run

# Run baseline (GA only, no RL)
python -m examples.triangle.run --no-rl

# Ablation: single-stream Q-network
python -m examples.triangle.run --single-stream

# Ablation: without STPE
python -m examples.triangle.run --no-stpe

# Custom parameters
python -m examples.triangle.run --seed 42 --pop-size 30 --max-gen 5000

# Custom clustering parameters
python -m examples.triangle.run --preprocess-gen 200 --num-clusters 3
```

## Experiment Pipeline

The full framework follows this pipeline:

1. **Preprocessing**: Run g' generations of baseline GA on every mutant group
2. **Feature Extraction**: Compute a 4-dimensional feature vector per group:
   - `Rate_kill`: ratio of killed mutants
   - `div`: population diversity (σ_i / Span_i averaged over dimensions)
   - `imp_dis`: average killing distance improvement
   - `Rate_stag`: proportion of stagnation generations
3. **Clustering**: K-Means on feature vectors → select representative group (nearest to centroid) per cluster
4. **Phase 1 (Training)**: Train DDQN agent on representative groups with online RL
5. **Phase 2 (Inference)**: Freeze agent, apply to remaining groups (parameter transfer within clusters)

## Key Design Choices

- **Action-dependent decision interval τ(a)**: `τ_long` for gentle operators (a0: SBX, a1: Uniform+Gaussian); `τ_short` for destructive exploration (a2: large Gaussian). This ensures rapid re-evaluation after large perturbations.
- **Reward**: R = (fit_before − fit_after) / τ — per-generation average fitness improvement, normalized by τ for fair comparison across actions.
- **Normalization**: All distances and features use x/(x+1) mapping to [0,1).
- **STPE lifecycle**: Visit counters are initialized fresh each time stagnation triggers STPE (Algorithm 3), not accumulated globally.

## Ablation Configurations

| Configuration | `USE_RL` | `USE_DUAL_STREAM` | `ENABLE_STPE` | Description |
|---|---|---|---|---|
| GA<sub>Trad</sub>-Group | `False` | — | `False` | Traditional GA baseline |
| DDQN-GA | `True` | `False` | `False` | Single-stream RL |
| DDQN<sub>DS</sub>-GA-noSTPE | `True` | `True` | `False` | Dual-stream, no STPE |
| **DDQN<sub>DS</sub>-GA** | `True` | `True` | `True` | **Full framework** |

## Adding a New Test Subject

To apply the framework to a new program under test:

1. **Implement `SubjectInterface`** in a new file (see `examples/triangle/subject.py`):

```python
from ddqn_ds_ga.subject import SubjectInterface, ParamSpec

class MySubject(SubjectInterface):
    def param_spec(self):
        return [ParamSpec("x", 0, 100), ParamSpec("y", -50, 50)]

    def mutant_metadata(self):
        return {0: {"operator_type": "ROR", "complexity": 5, "description": "..."}, ...}

    def mutant_groups(self):
        return [{"group_id": 0, "mutant_ids": [0, 1, 2]}, ...]

    def equivalent_mutants(self):
        return set()

    def run_original(self, params):
        # Execute original program, return output
        ...

    def run_mutant(self, params, mutant_id):
        # Execute mutant, return (killed, output)
        ...

    def mutation_distance(self, params, mutant_id):
        # Compute strong mutation distance
        ...
```

2. **Run the experiment**:

```python
from ddqn_ds_ga import run_experiment, Config
from my_subject import MySubject

results = run_experiment(subject=MySubject(), config=Config())
```

## Key References

- DeMillo, R.A., Lipton, R.J., Sayward, F.G. "Hints on Test Data Selection: Help for the Practicing Programmer." *IEEE Computer*, 1978.
- Jia, Y., Harman, M. "An Analysis and Survey of the Development of Mutation Testing." *IEEE TSE*, 2011.
- Papadakis, M., Malevris, N. "Searching and Generating Test Inputs for Mutation Testing." *Springer*, 2013.
- Tang, H. et al. "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning." *NeurIPS*, 2017.

## License

This project is released for academic research purposes.
