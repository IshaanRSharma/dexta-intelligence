"""Plain-model baseline on the LLM-CGM tasks (head-to-head vs the dexta harness).

The honest baseline: the SAME subject model (claude-sonnet-4-6), given the SAME
patient's COMPLETE CGM record in-context as CSV, NO tools, asked the SAME 15
questions, scored by the SAME ground-truth formulas. This measures the gap the
harness actually closes - on questions a model must compute over 6,048 readings
it cannot hold precisely. We give the raw data (not a precomputed summary) so the
test is the model's, not ours.

dexta's answers are read from bench/results/llmcgm_raw.json (same data/seed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench import run_llmcgm as llmcgm  # QUESTIONS, ground_truth, score

from dexta_intelligence.llm.factory import ModelSpec, get_model
from dexta_intelligence.testing.synthetic import scenario_sensitivity_shift

RAW_SYSTEM = (
    "You are a careful diabetes data assistant for a person with type 1 diabetes. "
    "Below is the patient's COMPLETE continuous glucose record as CSV rows "
    "(timestamp, glucose mg/dL, 5-minute cadence). Answer the question by computing "
    "PRECISELY from this data - compute the actual value, do not estimate or guess. "
    "Give the specific number, time, or category the question asks for. Never give "
    "dosing or treatment advice."
)


def _text_of(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def main() -> int:
    model = get_model(ModelSpec("anthropic", "claude-sonnet-4-6", 0.0, 1800))
    # IDENTICAL data to the dexta harness run (same scenario/seed/effect/days).
    events, _ = scenario_sensitivity_shift(seed=5, n_days=21, effect_size=60.0)
    g = events["glucose"]
    ts = [e.ts for e in g]
    cgm = [float(e.mg_dl) for e in g]
    gt = llmcgm.ground_truth(ts, cgm)
    csv = "timestamp,glucose_mg_dl\n" + "\n".join(
        f"{e.ts.strftime('%Y-%m-%d %H:%M')},{int(e.mg_dl)}" for e in g
    )

    dexta = {it["q"]: it for it in json.loads(
        (Path(__file__).resolve().parent / "results" / "llmcgm_raw.json").read_text()
    )["items"]}

    out: dict[str, Any] = {
        "baseline": "plain claude-sonnet-4-6, full CGM CSV in-context, NO tools",
        "data": (f"sensitivity_shift seed=5 e60, 21 days, "
                 f"{len(cgm)} readings ({len(csv)} chars CSV)"),
        "items": [],
    }
    rb = rc = hb = hc = 0  # raw scored/correct, harness scored/correct
    for key, (qtext, gtkey, stype) in llmcgm.QUESTIONS.items():
        t0 = time.time()
        try:
            resp = model.invoke([
                {"role": "system", "content": f"{RAW_SYSTEM}\n\n{csv}"},
                {"role": "user", "content": qtext},
            ])
            ans = _text_of(resp)
        except Exception as exc:
            ans = f"ERROR: {exc}"
        rv = llmcgm.score(stype, gt.get(gtkey), ans)
        hv = dexta.get(key, {}).get("verdict")
        if rv is not None:
            rb += 1
            rc += int(rv)
        if hv is not None:
            hb += 1
            hc += int(hv)
        out["items"].append({
            "q": key, "question": qtext, "gt": gt.get(gtkey), "type": stype,
            "raw_verdict": rv, "harness_verdict": hv,
            "raw_seconds": round(time.time() - t0, 1), "raw_answer": ans,
        })
        gv = gt.get(gtkey)
        gvs = round(gv, 2) if isinstance(gv, float) else gv
        print(f"  {key:4s} raw={rv!s:5s} harness={hv!s:5s}  gt={gvs}", flush=True)
    out["raw_auto"] = f"{rc}/{rb}"
    out["harness_auto"] = f"{hc}/{hb}"
    print(f"\nRAW (no tools):  {rc}/{rb}", flush=True)
    print(f"HARNESS (dexta): {hc}/{hb}", flush=True)

    dest = Path(__file__).resolve().parent / "results" / "llmcgm_baseline.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"WROTE {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
