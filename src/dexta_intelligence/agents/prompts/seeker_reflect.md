The patient's goal was:
  "{goal}"

You just produced this answer:
  "{answer}"

Tools you called this round: {tools}

Did this fully answer the goal? Judge strictly: if the goal asked about a
specific spike/window/period you never zoomed or narrowed into, you are NOT
satisfied. Output STRICT JSON, no prose:
{{"satisfied": true|false,
  "missing": "<what is still unanswered>",
  "next_tool_hint": "<the EXACT name of one available tool to call next, e.g. zoom_event>",
  "reason": "<one sentence on why the answer falls short>"}}
