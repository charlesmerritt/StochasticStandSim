# agents.md

## Purpose

This document defines how AI coding agents should operate in this repository.

The goal is to:
- Preserve correctness over cleverness
- Avoid architectural drift
- Enforce explicit reasoning and contracts
- Prevent silent assumption creep
- Enable safe collaboration between humans and agents
- Ensure compatibility with a modern Python tooling stack

Agents must treat this file as binding.

---

## 1. Operating Modes

Agents operate in one of three explicit modes. The mode must be stated or inferred from the task.

### 1.1 Build Mode

Used when implementing or modifying code.

**Allowed**
- Iterative changes
- Incremental refactors
- Adding tests
- Expanding functionality

**Required behavior**
- Respect existing architecture
- Do not invent abstractions unless required
- Do not silently change interfaces
- Prefer small, reversible changes

**Disallowed**
- Large refactors without approval
- Rewriting modules for style
- Changing data models without justification

---

### 1.2 Review Mode

Used when evaluating code, design, or logic.

**Required behavior**
- Assume nothing
- Identify hidden assumptions
- Flag ambiguity
- Point out failure modes
- Ask clarifying questions before suggesting changes

**Disallowed**
- Writing new code unless explicitly asked
- Optimizing prematurely
- Accepting existing design as correct

---

### 1.3 Design Mode

Used when planning architecture or systems.

**Required behavior**
- Start from first principles
- Identify invariants
- Explicitly define inputs, outputs, and failure modes
- Prefer simple composable systems
- State tradeoffs clearly

**Disallowed**
- Implementation details
- Framework decisions without justification
- Vague abstractions

---

## 2. Context Management Rules

### 2.1 Context Is Not Truth

Long conversation history may contain:
- Outdated assumptions
- Incorrect conclusions
- Partial designs
- Abandoned directions

Agents must not treat prior context as authoritative.

---

### 2.2 When to Request a Reset

Agents should recommend a new session if:
- The task changes domains
- Architecture is being reconsidered
- Conflicting assumptions appear
- The agent feels constrained by earlier context

---

### 2.3 Cold Start Protocol

When starting fresh:
- Restate the problem in your own words
- Identify unknowns
- Ask for constraints if missing
- Do not reuse earlier design decisions unless restated

---

## 3. Tooling and Environment Standards

This repository uses a modern Python tooling stack. Agents must align with it.

### 3.1 Tooling Assumptions

Unless explicitly stated otherwise, assume:

- **uv** for environment and dependency management
- **ruff** for linting and formatting
- **pytest** for testing
- **pyproject.toml** as the source of truth
- Python ≥ 3.10

Agents must not:
- Introduce pip/venv instructions when uv is in use
- Add black, isort, or flake8 unless explicitly requested
- Add tools that duplicate existing functionality

---

### 3.2 Tool Usage Expectations

When relevant, agents should:

- Use `uv add` or `uv pip` semantics when describing dependency changes
- Assume `ruff` handles formatting and linting
- Write tests compatible with `pytest`
- Respect existing configuration in `pyproject.toml`

If tooling is unclear, the agent must ask before proceeding.

---

## 4. Change Discipline: Smallest Possible Change

### 4.1 Default Rule

**Always make the smallest change that solves the problem.**

Agents must assume:
- Existing structure is intentional
- Refactors are risky
- Simpler and cleaner is preferred

### 4.2 Prohibited Without Explicit Approval

- Large-scale rewrites
- Moving files across modules
- Introducing new abstractions
- Changing data models
- Renaming public interfaces

### 4.3 Required Mindset

Before making a change, the agent must ask:

1. What is the minimum change that fixes this?
2. Can this be solved locally?
3. Can this be solved without altering interfaces?
4. Is this change reversible?

If the answer to any is “no,” the agent must stop and ask.

---

## 5. Reusable Design Pattern Guidance

Agents should prefer **simple, well-known, reusable design patterns** when solving problems, instead of inventing custom abstractions.

The goal is:
- Reusability
- Predictability
- Low cognitive overhead
- Minimal architectural churn

Patterns should be used **only when they reduce complexity**, not to demonstrate sophistication.

---

### 5.1 Preferred Patterns

#### Generators and Iterators

Use when:
- Producing streams of data
- Processing large datasets
- Modeling stepwise or incremental computation

Example:
```python
def batch_iterator(items: list[int], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
```

Preferred over:
- Manual indexing
- Precomputing full lists
- Stateful loop classes

#### Factory Functions

Use when:
- Object construction varies by config
- You want to avoid complex __init__ logic
- Behavior depends on runtime parameters

Example:
```python
def make_optimizer(name: str, **kwargs):
    if name == "adam":
        return Adam(**kwargs)
    if name == "sgd":
        return SGD(**kwargs)
    raise ValueError(f"Unknown optimizer: {name}")
```

Preferred over:
- Large conditional blocks in constructors
- Inheritance trees for configuration differences

#### Singleton or Module-level instances

Use sparingly for:
- Configuration
- Logging
- Shared read-only state
- Cached resources
```python
settings = load_settings()
```
Avoid:
- Mutable global state
- Singletons with hidden side effects

#### Strategy Pattern

Use when:
- You have a family of algorithms
- Algorithms can be selected at runtime
- Algorithms are independent of their use

Prefer over:
- Deep if/else trees
- Flag-driven logic

#### Adapter Pattern

Use when:
- Integrating external libraries
- Normalizing inconsistent interfaces
- Wrapping unstable APIs

Goal:
- Isolate third-party changes
- Keep core logic clean

```python
class Adapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def request(self):
        return self.adaptee.specific_request()
```

#### Facade Pattern

Use when:
- Simplifying complex interfaces
- Providing a simplified interface to a complex subsystem
- Hiding implementation details

### 5.2 Patterns to Avoid by Default

Avoid unless explicitly justified:
- Abstract base class hierarchies
- Deep inheritance trees
- Metaclasses
- Dynamic attribute injection
- Reflection-based logic
- Overuse of decorators
- Frameworks built on top of frameworks

If one of these is necessary, the agent must explain why.

### 5.3 Pattern Selection Rule

Before introducing a pattern, the agent must ask:
- Does this reduce duplication?
- Does this improve readability?
- Does this make behavior easier to test?
- Is this simpler than the alternative?

If any answer is no, do not introduce the pattern.

### 5.4 Minimal Change Principle (Reinforced)

When modifying existing code:
- Prefer extending existing patterns over adding new ones
- Prefer local fixes over structural changes
- Prefer explicit code over abstraction
- Prefer clarity over cleverness

If a change requires:
- Moving multiple files
- Renaming public APIs
- Introducing new base classes

The agent must stop and ask for approval.

### 5.5 Design Smell Warnings

Treat the following as red flags:

- “This will be cleaner if we rewrite…”
- “Let’s generalize this now”
- “This might be useful later”
- “We should future-proof this”

These are not valid reasons for change.