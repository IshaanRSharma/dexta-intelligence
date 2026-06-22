Your hypothesis: {claim}
You called {tool}({args}) and it returned:
{result}

The statistic is computed; you only JUDGE. Decide:
- "claim": the effect looks real and worth formally testing (interpretation
  moderate/large, groups adequately sized).
- "wonder": suggestive but underpowered or ambiguous - bank it as an open
  question for a future run rather than claiming it now.
- "drop": no meaningful effect.

Output STRICT JSON: {{"verdict": "claim"|"wonder"|"drop", "reason": "<one sentence>"}}
