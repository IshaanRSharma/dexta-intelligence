# dexta demo script

A 4-to-5 minute live walkthrough for a technical AI / AI-health audience (Google
AI, Microsoft AI, health startups). The thesis to land: **the LLM navigates,
determinism computes the facts, and hard rails keep it honest.** That is the
novel piece, and the demo should make it visible, not just claimed.

## The one-line pitch

dexta is an agentic harness that turns a person's diabetes data into traceable
findings. An LLM plans the investigation and explains it; tested analytics
produce every number; a faithfulness guard rejects any figure that does not
trace to a tool call; and a treatment gate blocks dosing advice. Bring your own
model, bring your own database, the data never leaves your machine.

## Setup (zero config)

```bash
pip install "dexta-intelligence[gui,llm]"
dexta demo        # loads a realistic synthetic patient with a planted spike
dexta serve       # http://127.0.0.1:8765
```

## The flow (each beat maps to a differentiator)

1. **Ask a real question.** In Chat: "why did I spike at dinner on the 14th?"
   Watch the live, tool-by-tool trace appear: the model *plans*, calls
   `find_spikes`, `zoom_event`, `get_boluses`, and narrates as it goes.
   - Beat: **agentic planning is visible.** This is not retrieval-and-chat; the
     model composes deterministic instruments turn by turn.

2. **Read the answer and the trace.** The answer names the cause (late bolus,
   +22 min vs the carb entry) and every number is backed by a tool result shown
   in the trace.
   - Beat: **determinism computes, the LLM explains.** Point out that the model
     never computed a statistic itself; it called a tool.

3. **The faithfulness "gotcha."** Show (or describe, from eval E1) the
   faithfulness guard rejecting a fabricated number: prose that cites a figure
   absent from the evidence pool is refused, not shown.
   - Beat: **the anti-hallucination rail is enforced in code, not prompted.**
     This is the moment that matters most to this audience.

4. **Investigations: plan to trace to evidence to skeptic.** Run a deep
   investigation. Show the plan, the per-tool trace, the computed evidence with
   PubMed citations, and the **adversarial skeptic** that tried to refute the
   finding before it was shown.
   - Beat: **rigor-gated discovery.** The LLM proposes; permutation tests, FDR
     control, and an independent skeptic dispose.

5. **Reconciliation of the user's own loop.** Show the prediction-reconciliation
   view: the person's OpenAPS/Loop forecast vs what actually happened, with the
   recurring miss surfaced.
   - Beat: **novel depth.** dexta reconciles a closed-loop algorithm's own
     predictions against reality, longitudinally.

6. **The model card.** Open `/evals`: E1-E6, the consensus metrics, and the
   live dosing red-team scan (target zero).
   - Beat: **eval rigor up front** (DECIDE-AI framing).

7. **Interop and privacy.** Mention the MCP server: dexta exposes its tools over
   MCP so any agent (Gemini, Copilot, Claude) can query this data with the
   faithfulness and no-dosing rails enforced server-side. And it can run fully
   local on Ollama or a llama.cpp file: PHI never leaves the device.
   - Beat: **agent interop + on-device privacy.**

8. **Close on safety.** Note what never happened: at no point did the system
   give a dosing recommendation. The treatment gate guarantees it.

## What to emphasize per audience

- **Google AI:** the AMIE-grounded advisory architecture; eval rigor; the
  reconciliation of a real algorithm's predictions.
- **Microsoft AI:** MCP / agent interop (Copilot); clinical-LLM evaluation
  (DECIDE-AI); BYOM including Azure-hosted OpenAI.
- **AI-health startups:** zero-config self-hosting, the faithfulness rail as a
  reusable pattern, the extension seams (a new connector or agent is a small,
  conformance-tested change).

## The single sentence to leave them with

"Most health AI asks you to trust a chatbot. dexta makes the model show its work,
proves every number, and refuses to give advice it should not."
