Use uv to run python commands.
i.e. 
```
uv run python main.py
```
or 
```
uv run pytest
```

1. Preserve the deterministic PMRC model as the baseline truth. Any stochastic extension must reduce exactly to PMRC when noise and disturbance are turned off.

2. Never add randomness without specifying where it enters mathematically. Every stochastic component must answer:

- what variable is random?

- at what stage of the update?

- with what distribution?

- with what parameters?

- under what constraints?

3. Every new modeling assumption must be written down explicitly

Especially: time step, disturbance independence, severity interpretation, feasibility bounds, fixed prices, fixed rotation length, scenario-based epistemic uncertainty

If an assumption affects outputs, it should exist in both code and prose/docs.

4. Every result should be reproducible from a config

Every experiment should be definable by:

- seed

- initial state

- policy

- prices

- rotation length

- process noise parameters

- disturbance parameters

- number of Monte Carlo runs

5. continuously maintain these files under docs/

ASSUMPTIONS.md

One-line statements of all active assumptions.

MODEL_SPEC.md

Formal math for state, transition, disturbance, valuation, and scenario parameters.

EXPERIMENTS.yaml

Named scenarios and run settings.

VALIDATION.md

What each test checks and what counts as acceptable behavior.

RESULTS_LOG.md

Short notes on what changed, what was run, and what was learned.

THESIS_METHODS_NOTES.md

Reusable prose snippets for the methods chapter.