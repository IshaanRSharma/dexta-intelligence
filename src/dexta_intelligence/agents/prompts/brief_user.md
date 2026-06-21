Active findings, ranked, each with an index (each: index, kind, headline, evidence, stats):

{findings}

Write the brief. Output STRICT JSON, no prose:
{{"summary": "<one or two sentence headline over the findings>",
  "sections": [{{"title": "<short section title>",
                "body": "<2-4 sentence explanation citing only this finding's numbers>",
                "finding_idx": <index of the finding this section explains>}}]}}
