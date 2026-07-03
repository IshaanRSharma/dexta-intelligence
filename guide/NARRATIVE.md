# dexta: a trustworthy intelligence layer over your medical data

Positioning note (internal). This is the load-bearing story dexta tells: what it is, why
it is built the way it is, and the one line it will not cross. It is the source the README,
the LinkedIn post, and any talk draw from. Keep it honest; the audience checks.

## The one-sentence version

dexta is a self-hosted intelligence layer that turns your own diabetes data into answers you
can trace to the arithmetic that produced them, and it never tells you what to dose.

## The problem it exists for

A person with Type 1 diabetes generates an enormous, high-stakes time series: a glucose reading
every five minutes, plus every bolus, basal change, carb entry, and forecast. The obvious thing
to do in 2026 is paste it into a language model and ask "why are my mornings high?" The model
answers fluently and is often confidently wrong about the numbers, because exact arithmetic over
thousands of readings is not what language models do. It will narrate summing six thousand
readings and hand back an overnight average that is off by 25 mg/dL. At the dinner table that is
a bad guess. For someone deciding how to correct a low, a wrong number is a safety event.

This is not a niche observation anymore. It is a named, quantified failure mode across the
clinical-AI literature: models confabulate clinical numbers, and the fix the field has converged
on is to stop letting the model do the math. The model should reason. Deterministic code should
compute. dexta is a hardened, open implementation of exactly that split, with the safety rails
and the receipts that turn it from an architecture diagram into something you can run and break.

## What dexta actually is

Three layers, bottom to top, each doing only its job.

1. **Determinism computes the facts.** Read-only connectors pull your CGM, insulin, pump, and
   wearable history into a store you control (SQLite for zero-setup, Postgres for scale). Tested
   analytics produce every number: time in range, coefficient of variation, GMI, oref0 IOB and
   COB reconciliation, permutation tests, false-discovery control, error grids. No number in an
   answer is ever invented by the model.

2. **The model reasons on top.** An agent plans an investigation, calls the deterministic tools
   one at a time, ranks competing hypotheses, and explains what it found. It decides freely which
   instruments to run and in what order. What it cannot do is produce a figure that did not come
   out of a tool.

3. **Two rails bound the output, in code.** A faithfulness guard rejects any prose whose numbers
   do not trace back to a tool call. A treatment gate blocks dosing, basal, carb-ratio, and
   correction instructions, always. Both live below the safety line, in tested code, not in a
   prompt that can be talked around.

The result is an answer that carries its work: a visible plan, a tool-by-tool trace, the evidence
behind each number, the hypotheses it ruled out, and an honest note on what it could not check.

## Why "trustworthy and traceable" is the whole point, not a feature

Trust in medical AI is not a vibe; it is a mechanism. The two most valuable companies in clinical
AI right now sell exactly one thing underneath the product: the ability to drill from a claim down
to its source. dexta applies that same posture to your personal data. Every number is traceable to
the tool call that computed it, so "your overnight average was 149" is not the model's opinion; it
is a figure you can follow to the reading window it came from. That is what verifiability means
when the stakes are a person's health, and it is the reason the faithfulness guard is the center
of the design rather than a wrapper around it.

## The line it will not cross

dexta shows you the facts you tune from. It never prescribes the setting or the dose. That line is
absolute, and it is enforced in code because we tested the alternative and the alternative failed.
A well-written system prompt telling the model "never give dosing advice" does not hold; the
published literature agrees, and so did our own testing. So the treatment gate is a code rail, not
an instruction, and it is the one part of the system the agent has no authority over.

This is also the honest posture for the regulatory moment: informational and wellness software that
shows you your own data is well inside the lines, while software that drives treatment is a medical
device. dexta is firmly the former, by construction, on purpose.

## The receipt

Claims about trustworthiness are cheap, so dexta carries proof. We ran it against LLM-CGM, a
peer-reviewed benchmark (Healey and Kohane, PSB 2025) whose questions and ground-truth formulas are
published. Same model both ways. The plain model got the raw glucose record in-context and answered
from it. dexta computed through its tools. On the numeric tasks, the plain model's mean absolute
error was 14.7 and dexta's was 0.15, roughly a hundredfold difference, and every one of dexta's
numbers traces to a tool call.

The scope is stated plainly and travels with the number: our run used dexta's own synthetic patient
generator (the benchmark's real data is restricted), one patient, a single pass, 14 of the 30 tasks.
This is a controlled external-validity probe, not a clinical claim, and it is deliberately not
"language models are bad." Exact arithmetic over long sequences is a known model limitation; the
point is what confident, plausible, untraceable error looks like at clinical stakes, and that a
verification harness removes it.

## What it is not

It is not a medical device. It is not a diagnosis or a treatment tool. It is not a claim to beat a
clinician, and it does not frame itself against one. It is not novel in its core architecture: the
"model reasons, code computes" split is now published consensus, independently validated in the
same window by academic work on CGM agents. dexta's contribution is the hardening: the two code
rails, the every-number-traceable guarantee, the external-benchmark receipt, the honest negatives,
and a thing you can actually run on your own data on your own machine.

## The heritage it owes

dexta stands on the #WeAreNotWaiting lineage: OpenAPS, Nightscout, the DIY-loop community that built
open diabetes tooling years before industry did. It reuses their algorithms (oref0's IOB and COB
math, dose-free) and speaks their interchange formats. It is not a member of that community by
posturing; it owes that community a debt and says so. It answers a different question than Nightscout
or an AGP report does: natural-language interrogation of your own history, with the math kept
deterministic and the safety line kept in code.
