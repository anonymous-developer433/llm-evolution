# llm-evolution

A professional-grade Python library combining Large Language Models (LLMs) with Evolutionary Algorithms (EA) to optimize programs, systems, and strategies across multiple domains (e.g., CUDA kernels, RISC-V assembly, algorithmic trading).

`llm-evolution` provides a robust, protocol-based framework for building complex evolutionary pipelines where LLMs can act as intelligent mutators, crossovers, or evaluators.

## 🚀 Features

- **Library-first layout**: `src/`-based packaging for reliable imports
- **Modern Python**: Python 3.12+
- **Protocol-based design**: swap strategies via clear interfaces

## 📋 Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## 🛠️ Installation

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install as a dependency (recommended)

```bash
# Add this library from Git (PEP 508 URL)
uv add "llm-evolution @ git+https://github.com/alexfdez1010/llm-evolution.git"

# Add from a branch or tag
uv add "llm-evolution @ git+https://github.com/alexfdez1010/llm-evolution.git@main"
```

### Install locally (if you are working from a checkout)

```bash
# Clone the repository
git clone https://github.com/alexfdez1010/llm-evolution
cd llm-evolution

# Sync dependencies (creates .venv and installs packages)
uv sync

# Install in editable mode
uv pip install -e .
```

## 🎯 Usage

`llm-evolution` provides a flexible framework for implementing evolutionary algorithms. It uses a protocol-based design, allowing you to easily swap out different strategies for population initialization, evaluation, selection, crossover, and mutation.

### Core Components

The library is built around several key interfaces (Protocols):

- **`InitialPopulation[T]`**: Generates the starting set of individuals.
- **`Evaluation[T]`**: Calculates the fitness score for an individual.
- **`Selection[T]`**: Decides which individuals survive to the next generation.
- **`Crossover[T]`**: Combines parents to create offspring (optional).
- **`Mutation[T]`**: Introduces random variations into individuals (optional). Can return `None` if no mutation is performed.
- **`FinishCondition[T]`**: Determines when the evolutionary process should stop.

### Basic Example

Here is how you can set up and run a simple evolutionary algorithm:

```python
import random
from llm_evolution.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from llm_evolution.interfaces.initial_population import initial_population_fn
from llm_evolution.interfaces.evaluation import evaluation_fn
from llm_evolution.interfaces.selection import selection_fn
from llm_evolution.interfaces.finish_condition import finish_condition_fn

# 1. Define your population initialization
@initial_population_fn
def my_initial_pop(size: int) -> list[int]:
    return [random.randint(0, 100) for _ in range(size)]

# 2. Define how to evaluate individuals (higher is better)
@evaluation_fn
def my_evaluation(instance: int) -> float:
    return float(instance)  # Simple maximization of the integer value

# 3. Define survivor selection
@selection_fn
def my_selection(population, offspring, fitness_scores):
    # Keep the best individuals from the combined pool
    combined = population + offspring
    indexed = list(enumerate(fitness_scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [combined[i] for i, _ in indexed[:len(population)]]

# 4. Define when to stop
@finish_condition_fn
def my_finish(population, generation, fitness_scores):
    return generation >= 50 or max(fitness_scores) >= 100

# 5. Initialize and run the algorithm
ea = EvolutionaryAlgorithm(
    initial_population=my_initial_pop,
    evaluation=my_evaluation,
    selection=my_selection,
    finish_condition=my_finish,
    population_size=20
)

result = ea.run(log=True)
print(f"Best instance: {result.best_instance} with fitness {result.best_fitness}")
```

## 🧬 How the Algorithm Works

The `EvolutionaryAlgorithm` orchestrates a standard evolutionary cycle:

1. **Initialization**: The `initial_population` strategy generates an initial set of `population_size` individuals.
2. **Evaluation**: Each individual in the current population is evaluated using the `evaluation` strategy to determine its fitness.
3. **Check Stop Condition**: The `finish_condition` is checked. If it returns `True`, the evolution stops.
4. **Reproduction**:
    - **Crossover**: If a `crossover` strategy is provided, pairs of parents are selected and combined to create offspring.
    - **Mutation**: If a `mutation` strategy is provided, random variations are applied to a subset of the population and offspring.
5. **Selection**: The `selection` strategy chooses which individuals from the current population and the new offspring will survive to the next generation.
6. **Iteration**: Steps 2-5 are repeated until the stop condition is met.

The library's use of Generics (`T`) ensures that you can evolve any type of object, from simple numbers to complex LLM-generated code or system configurations.

## 📦 Dependency Management

### Adding Dependencies (uv)

```bash
# Add a runtime dependency
uv add <package-name>

# Add this library from Git (PEP 508 URL)
uv add "llm-evolution @ git+https://github.com/alexfdez1010/llm-evolution.git"

# Add from a branch or tag
uv add "llm-evolution @ git+https://github.com/alexfdez1010/llm-evolution.git@main"

# Example: Add requests library
uv add requests

# Example: Add rich
uv add rich
```

### Updating Dependencies

```bash
# Update a specific package
uv lock --upgrade-package <package-name>

# Update all packages
uv lock --upgrade

# Sync after updating
uv sync
```

### Removing Dependencies

```bash
uv remove <package-name>
```

## 📁 Project Structure

```text
.
├── src/
│   └── llm_evolution/            # Main package source code
│       ├── __init__.py
│       ├── ai/                    # LLM/embedding interfaces + implementations
│       ├── algorithm/             # Evolutionary algorithm logic
│       ├── implementations/       # Concrete evolution implementations
│       ├── interfaces/            # Evolution interfaces (mutation, crossover, etc.)
│       └── version.py
├── tests/
│   ├── unit/                     # Unit tests with mocks
│   └── integration/              # Integration tests (real APIs/services)
├── .python-version               # Python version (3.12)
├── AGENTS.md                     # AI coding assistant guidelines
├── pyproject.toml                # Project metadata & dependencies
├── uv.lock                       # Locked dependencies (DO NOT edit manually)
├── .gitignore                    # Git ignore patterns
├── LICENSE                       # MIT license
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

For sensitive configuration, create a `.env` file (already in `.gitignore`):

```bash
# .env
API_KEY=your-secret-key
DATABASE_URL=postgresql://localhost/db
```

Load with `python-dotenv` (already included):

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
```

## 📝 License

MIT License. See [LICENSE](LICENSE).
