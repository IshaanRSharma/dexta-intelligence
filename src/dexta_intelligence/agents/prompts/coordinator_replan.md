A first round of investigations ran for this goal and produced the findings
below. Decide whether a focused FOLLOW-UP round is warranted - only investigations NOT already
run, chosen to drill into or challenge what the first round surfaced. If the first round already
covers the goal, return an empty list.

GOAL: {goal}
ALREADY RAN: {ran}
FIRST-ROUND FINDINGS:
{findings}
AVAILABLE (not yet run):
{remaining}

Output STRICT JSON (empty list = done):
{{"investigations": ["<name>", ...], "reason": "<one sentence>"}}
