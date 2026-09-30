from __future__ import annotations

from agent1 import SOPDesignAgent


class Agent2(SOPDesignAgent):
    agent_id = "Agent2"
    role_hint = (
        "Design the second candidate independently and competitively. The design "
        "must be meaningfully different from Agent1."
    )
    difference_hint = (
        "Prefer a different literature family, a different fusion strategy, or a "
        "different search schedule from Agent1. If Agent1 is DE-like, consider "
        "EDA/CMA-style sampling, swarm leadership, relay optimization, or another "
        "clearly distinct metaheuristic combination. If feedback indicates useful "
        "components, cross them only when it is justified by the observed results."
    )

