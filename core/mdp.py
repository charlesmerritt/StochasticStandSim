""" Transition function that composes growth, disturbances, economics. No RL library imports.

transition(state, action, cfg, params) -> tuple[new_state, reward, info]

Order: apply action effects that happen immediately, call growth.project_one_year, sample disturbance, apply ADSR and salvage, compute reward, advance age. """



