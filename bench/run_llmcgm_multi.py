"""Multi-patient LLM-CGM benchmark: dexta harness vs plain model, N=5 patients.

Hardens the n=1 LLM-CGM head-to-head (bench/run_llmcgm.py) into a distribution.
Five synthetic patients with DISTINCT seeds AND distinct physiology profiles
(reference, high-variability, flatline-stable, sensor-gap-heavy, hyperglycemic)
are built with the repo's own generator (dexta_intelligence.testing.synthetic,
generate_dataset). P1 reproduces the original single-patient headline exactly.

Both arms run the SAME subject model (claude-sonnet-4-6, temp 0) on the SAME
exactness-scored questions, scored against a ground truth computed independently
with numpy directly from each synthetic record (not via dexta's tools). The plain
arm gets the complete CGM record as CSV in-context, no tools; the dexta arm
computes through its tool belt with the rails on.

Auto-scoring is a FLOOR: this dumps every raw answer and every parsed number so
anomalies (unparseable, off by orders of magnitude, refused) can be hand-checked.
Honest scope is unchanged: synthetic data (their real data is restricted), single
pass, a curated exactness subset of the 30 tasks. A controlled external-validity
probe, not a clinical claim.

Usage: python bench/run_llmcgm_multi.py  ->  bench/results/llmcgm_multi.json
       python bench/run_llmcgm_multi.py --estimate-only   (cost estimate, no calls)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench import run_ladder as ladder
from bench import run_llmcgm as llmcgm

from dexta_intelligence.llm.factory import ModelSpec, get_model
from dexta_intelligence.testing.synthetic import (
    BaselineConfig,
    PostWorkoutHypo,
    SensitivityRegimeShift,
    generate_dataset,
)

SUBJECT = ModelSpec("anthropic", "claude-sonnet-4-6", 0.0, 1800)
TOKEN_BUDGET = 30_000_000
CHARS_PER_TOKEN = 3.0  # conservative for dense numeric CSV (over-estimates budget)

RAW_SYSTEM = (
    "You are a careful diabetes data assistant for a person with type 1 diabetes. "
    "Below is the patient's COMPLETE continuous glucose record as CSV rows "
    "(timestamp, glucose mg/dL, 5-minute cadence). Answer the question by computing "
    "PRECISELY from this data - compute the actual value, do not estimate or guess. "
    "Give the specific number, time, or category the question asks for. Never give "
    "dosing or treatment advice."
)


# ── patient cohort: distinct seeds AND distinct physiology profiles ────────────
# Built with the repo generator (generate_dataset). P1 == the original headline
# patient (scenario_sensitivity_shift seed=5 e60, after_day=10). gap_drop applies
# deterministic sensor dropouts AFTER generation (post-processing, not a new
# generator): a few multi-hour blackouts per week plus random single-reading loss.

PATIENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "P1", "label": "reference (balanced, TIR ~80%)", "seed": 5,
        "config": None,
        "effects": (SensitivityRegimeShift(effect_size=60.0, after_day=10),),
    },
    {
        "id": "P2", "label": "high-variability (CV ~35%, wide spread)", "seed": 11,
        "config": BaselineConfig(fasting_mean=125.0, dawn_amplitude=32.0,
                                 ar1_sigma=15.0, ar1_phi=0.9),
        "effects": (SensitivityRegimeShift(effect_size=45.0, after_day=10),
                    PostWorkoutHypo(effect_size=40.0)),
    },
    {
        "id": "P3", "label": "flatline-stable (TIR ~93%, no highs)", "seed": 21,
        "config": BaselineConfig(fasting_mean=108.0, dawn_amplitude=8.0,
                                 diurnal_amplitude=3.0, ar1_sigma=2.0, ar1_phi=0.8),
        "effects": (),
    },
    {
        "id": "P4", "label": "sensor-gap-heavy (~13% readings missing)", "seed": 33,
        "config": BaselineConfig(fasting_mean=118.0, dawn_amplitude=24.0, ar1_sigma=8.0),
        "effects": (SensitivityRegimeShift(effect_size=40.0, after_day=10),),
        "gap_drop": True,
    },
    {
        "id": "P5", "label": "hyperglycemic (TIR ~51%, TAR ~48%)", "seed": 42,
        "config": BaselineConfig(fasting_mean=155.0, dawn_amplitude=28.0,
                                 ar1_sigma=10.0, basal_units_per_day=14.0),
        "effects": (SensitivityRegimeShift(effect_size=55.0, after_day=10),),
    },
)

N_DAYS = 21  # match the original run's window


def _drop_gaps(glucose: list[Any], seed: int) -> list[Any]:
    """Deterministic sensor loss: a few multi-hour blackouts/week + 5% dropout."""
    rng = random.Random(seed)
    n = len(glucose)
    weeks = max(1, n // (288 * 7))
    kill: set[int] = set()
    for _ in range(3 * weeks):
        start = rng.randrange(n)
        span = 4 * 12  # 4-hour blackout
        kill.update(range(start, min(n, start + span)))
    kill.update(i for i in range(n) if rng.random() < 0.05)
    return [e for i, e in enumerate(glucose) if i not in kill]


def build_patient(spec: dict[str, Any]) -> dict[str, Any]:
    events, _ = generate_dataset(
        seed=spec["seed"], n_days=N_DAYS, effects=spec["effects"],
        name=spec["id"], config=spec["config"],
    )
    if spec.get("gap_drop"):
        events = dict(events)
        events["glucose"] = _drop_gaps(events["glucose"], spec["seed"])
    g = events["glucose"]
    return {"events": events, "glucose": g}


# ── ground truth: computed independently with numpy from the record ────────────
# Matches the LLM-CGM get_answers conventions (ported in run_llmcgm.ground_truth);
# recomputed here with numpy so the exactness reference does not touch dexta tools.


def ground_truth_np(g: list[Any]) -> dict[str, Any]:
    ts = [e.ts for e in g]
    cgm = np.array([float(e.mg_dl) for e in g])
    mean = float(cgm.mean())
    sd = float(cgm.std())  # ddof=0, matches np.std / statistics.pstdev
    imax = int(cgm.argmax())
    imin = int(cgm.argmin())
    hours = np.array([t.hour for t in ts])
    night = cgm[(hours >= 0) & (hours < 6)]
    dinner = cgm[(hours >= 17) & (hours < 22)]

    def _runs(mask: np.ndarray) -> int:
        idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))) == 1)
        return len(idx)

    def _longest(mask: np.ndarray) -> int:
        best = cur = 0
        for f in mask:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return best

    return {
        "Q1_mean": mean,
        "Q2_max": float(cgm.max()),
        "Q3_std": sd,
        "Q4_min": float(cgm.min()),
        "Q5_TIR_pct": 100.0 * float(((cgm > 70) & (cgm < 180)).mean()),
        "Q6_TAR180_pct": 100.0 * float((cgm > 180).mean()),
        "Q7_TBR70_pct": 100.0 * float((cgm < 70).mean()),
        "Q8_CV": sd / mean,
        "Q9_TAR250_pct": 100.0 * float((cgm > 250).mean()),
        "Q10_eA1c": (46.7 + mean) / 28.7,
        "Q11_TBR54_pct": 100.0 * float((cgm < 54).mean()),
        "Q12_highest_time": ts[imax].strftime("%H:%M"),
        "Q14_lowest_time": ts[imin].strftime("%H:%M"),
        "Q17_longest_hyper_min": _longest(cgm > 180) * 5,
        "Q18_num_hypo": _runs(cgm < 70),
        "Q19_overnight_mean": float(night.mean()) if night.size else None,
        "Q20_period_highest": llmcgm._part_of_day(ts[imax].hour),
        "Q21_nocturnal_hypo": bool((night < 70).any()) if night.size else False,
        "Q22_dinner_max": float(dinner.max()) if dinner.size else None,
    }


# ── exactness (MAE) extraction: parse the model's stated number per question ───
# Plausibility ranges are physical unit bounds (NOT the truth) so formula
# constants and reading counts ("6,048 readings", "46.7") are not mistaken for
# the answer. is_cv normalizes a fraction (0.29) to percent points.

PLAUSIBLE: dict[str, tuple[float, float]] = {
    "Q1_mean": (30, 400), "Q2_max": (50, 400), "Q3_std": (3, 90),
    "Q4_min": (20, 200), "Q5_TIR_pct": (0, 100), "Q6_TAR180_pct": (0, 100),
    "Q7_TBR70_pct": (0, 100), "Q8_CV": (1, 150), "Q9_TAR250_pct": (0, 100),
    "Q10_eA1c": (3, 16), "Q11_TBR54_pct": (0, 100),
    "Q17_longest_hyper_min": (0, 6000), "Q18_num_hypo": (0, 600),
    "Q19_overnight_mean": (30, 400), "Q22_dinner_max": (50, 400),
}
NUMERIC_TYPES = ("num", "int", "pct", "cv")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# Boundary-aware: a full digit run, so "2025" is one token (not "202"+"5") and a
# year is never mistaken for a plausible glucose value.
_NUM_RE = re.compile(r"(?<![\d,.])-?\d[\d,]*(?:\.\d+)?(?!\d)")

# Metric keywords for answer selection: among plausible numbers, prefer the one
# inside an emphasized (bold) span next to the metric this question asks about.
# Both arms are discursive and mention many numbers; anchoring on the metric noun
# (NOT on the truth) picks the number the model attaches to the answer.
ANCHORS: dict[str, tuple[str, ...]] = {
    "Q1_mean": ("mean", "average"),
    "Q3_std": ("standard deviation", "deviation", " sd", "std"),
    "Q5_TIR_pct": ("in range", "in-range", "tir", "70-180"),
    "Q6_TAR180_pct": ("above range", "above 180", "time above", "tar", "hyper"),
    "Q7_TBR70_pct": ("below 70", "hypoglycem", "hypo", "time below", "tbr"),
    "Q8_CV": ("coefficient of variation", "variability", "cv"),
    "Q9_TAR250_pct": ("above 250", "250", "very high"),
    "Q10_eA1c": ("a1c", "gmi", "hba1c"),
    "Q11_TBR54_pct": ("below 54", "54", "clinically significant"),
    "Q17_longest_hyper_min": ("longest", "continuous", "sustained"),
    "Q18_num_hypo": ("episode", "separate", "number of", "times", "distinct"),
    "Q19_overnight_mean": ("overnight", "midnight", "nocturnal", "night"),
    "Q22_dinner_max": ("dinner", "evening meal", "5pm", "17:"),
}


def parse_answer(gtkey: str, stype: str, answer: str) -> dict[str, Any]:
    """Best-effort primary number + all candidates for hand-verification.

    Selection order (none of it references the truth):
      1. a plausible number inside a bold span that sits next to the metric keyword
      2. else the plausible number nearest a metric keyword
      3. else the first plausible number
    For percent questions, numbers written as "N%" are preferred. Large errors,
    truncation, and two-heuristic disagreement are flagged: auto-scoring is a
    floor, the hand-verified table is the truth.
    """
    lo, hi = PLAUSIBLE.get(gtkey, (float("-inf"), float("inf")))
    is_cv = stype == "cv"
    prefer_pct = stype in ("pct", "cv")
    low = answer.lower()

    def norm(v: float) -> float:
        return v * 100 if (is_cv and v < 1.5) else v

    bold_spans = [sp.span() for sp in _BOLD_RE.finditer(answer)]

    def is_bold(pos: int) -> bool:
        return any(s <= pos < e for s, e in bold_spans)

    # (value, position, is_percent, is_bold) for every plausible number.
    toks: list[tuple[float, int, bool, bool]] = []
    for m in _NUM_RE.finditer(answer):
        v = norm(float(m.group().replace(",", "")))
        if lo <= v <= hi:
            is_pct = answer[m.end():m.end() + 2].lstrip().startswith("%")
            toks.append((v, m.start(), is_pct, is_bold(m.start())))
    scope = [t for t in toks if t[2]] if (prefer_pct and any(t[2] for t in toks)) else toks

    anchors = ANCHORS.get(gtkey, ())
    anchor_pos = [m.start() for kw in anchors for m in re.finditer(re.escape(kw), low)]

    def in_span_pref_pct(cands: list[tuple[float, int, bool, bool]]) -> float | None:
        if prefer_pct and any(c[2] for c in cands):
            cands = [c for c in cands if c[2]]
        return cands[0][0] if cands else None

    # 1. bolded number in a span adjacent to an anchor keyword
    bold_anchored: float | None = None
    for s, e in bold_spans:
        win = low[max(0, s - 45):e + 40]
        if anchors and not any(kw in win for kw in anchors):
            continue
        picked = in_span_pref_pct([t for t in scope if s <= t[1] < e])
        if picked is not None:
            bold_anchored = picked
            break

    # 2. nearest to any anchor keyword
    anchor_nearest: float | None = None
    if scope and anchor_pos:
        anchor_nearest = min(scope, key=lambda t: min(abs(t[1] - a) for a in anchor_pos))[0]

    primary = bold_anchored
    if primary is None:
        primary = anchor_nearest if anchor_nearest is not None else (
            scope[0][0] if scope else None)

    bold_nums = sorted({round(t[0], 2) for t in toks if t[3]})
    pcts = sorted({round(t[0], 2) for t in toks if t[2]})
    disagree = (anchor_nearest is not None and primary is not None
                and abs(anchor_nearest - primary) > max(0.5, 0.02 * abs(primary)))
    return {"primary": primary, "anchor_nearest": anchor_nearest,
            "candidates": [t[0] for t in toks],
            "plausible_bold": bold_nums, "plausible_pct": pcts,
            "parser_disagree": disagree,
            "n_distinct_candidates": len(pcts if prefer_pct else bold_nums)}


def _refused(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in ("i cannot", "i can't", "i'm unable", "unable to",
                                  "i don't have", "cannot count", "can't count",
                                  "cannot precisely", "can't precisely"))


def _truncated(answer: str, stopped_reason: str | None) -> bool:
    if stopped_reason in ("length", "max_tokens"):
        return True
    tail = answer.rstrip()[-1:] if answer.strip() else ""
    return bool(answer) and tail not in ".!?)%\"" and len(answer) > 400


_API_ERROR_MARKERS = ("ERROR:", "no api credits", "credit balance",
                      "invalid_request_error", "rate_limit", "overloaded")


def _is_api_error(answer: str) -> bool:
    low = answer.lower()
    return answer.startswith("ERROR:") or any(mk in low for mk in _API_ERROR_MARKERS)


def score_item(gtkey: str, stype: str, gt: Any, answer: str,
               stopped_reason: str | None = None) -> dict[str, Any]:
    if _is_api_error(answer):
        # A failed call is not an answer: never scored, never fed into MAE.
        return {"verdict": None, "api_error": True, "abs_err": None,
                "primary": None, "flags": ["api-error"]}
    verdict = llmcgm.score(stype, gt, answer)
    row: dict[str, Any] = {"verdict": verdict}
    if stype in NUMERIC_TYPES and gt is not None:
        truth = gt * 100 if stype == "cv" else float(gt)
        parsed = parse_answer(gtkey, stype, answer)
        primary = parsed["primary"]
        abs_err = abs(primary - truth) if primary is not None else None
        flags: list[str] = []
        if primary is None:
            flags.append("unparseable")
        elif abs_err is not None and abs_err > 0.5 * abs(truth) + 5:
            flags.append("large-error")
        if parsed["parser_disagree"]:
            flags.append("parser-disagree")
        if _refused(answer):
            flags.append("refusal-language")
        if _truncated(answer, stopped_reason):
            flags.append("truncated")
        row.update({
            "truth_unit": truth, "primary": primary, "abs_err": abs_err,
            "anchor_nearest": parsed["anchor_nearest"],
            "plausible_bold": parsed["plausible_bold"],
            "plausible_pct": parsed["plausible_pct"],
            "candidates": parsed["candidates"], "flags": flags,
        })
    return row


def _text_of(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def build_csv(g: list[Any]) -> str:
    return "timestamp,glucose_mg_dl\n" + "\n".join(
        f"{e.ts.strftime('%Y-%m-%d %H:%M')},{int(e.mg_dl)}" for e in g
    )


def estimate(patients: list[dict[str, Any]]) -> dict[str, Any]:
    """Baseline-arm input-token estimate (the full-CSV arm dominates cost)."""
    nq = len(llmcgm.QUESTIONS)
    per: list[dict[str, Any]] = []
    total = 0
    for p in patients:
        csv = build_csv(p["glucose"])
        sys_chars = len(RAW_SYSTEM) + len(csv) + 40
        toks = int(sys_chars / CHARS_PER_TOKEN) * nq
        total += toks
        per.append({"id": p["spec"]["id"], "readings": len(p["glucose"]),
                    "csv_chars": len(csv), "baseline_input_tokens_est": toks})
    return {"n_questions": nq, "per_patient": per,
            "baseline_input_tokens_est": total,
            "note": "dexta arm is tool-based (no full CSV in context); its input "
                    "tokens are smaller per call but multi-step. Baseline is the "
                    "dominant cost and is what this bounds."}


def run_patient(model: Any, p: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    spec = p["spec"]
    g = p["glucose"]
    csv = build_csv(g)
    store = ladder.store_from_events(p["events"])
    ctx = ladder._ctx(store)
    items: list[dict[str, Any]] = []
    for key, (qtext, gtkey, stype) in llmcgm.QUESTIONS.items():
        truth = gt.get(gtkey)
        # dexta arm
        t0 = time.time()
        try:
            dres = ladder.harness_arm(model, ctx, qtext)
            dans, dstop = dres["text"], dres["stopped_reason"]
        except Exception as exc:
            dans, dstop = f"ERROR: {exc}", "error"
        dsec = round(time.time() - t0, 1)
        # plain arm: full CSV in context, no tools
        t0 = time.time()
        try:
            resp = model.invoke([
                {"role": "system", "content": f"{RAW_SYSTEM}\n\n{csv}"},
                {"role": "user", "content": qtext},
            ])
            bans = _text_of(resp)
        except Exception as exc:
            bans = f"ERROR: {exc}"
        bsec = round(time.time() - t0, 1)

        dscore = score_item(gtkey, stype, truth, dans, dstop)
        bscore = score_item(gtkey, stype, truth, bans, None)
        items.append({
            "q": key, "question": qtext, "type": stype, "gt": truth,
            "dexta": {**dscore, "stopped_reason": dstop, "seconds": dsec, "answer": dans},
            "plain": {**bscore, "seconds": bsec, "answer": bans},
        })
        de = dscore.get("abs_err")
        be = bscore.get("abs_err")
        print(f"    {key:4s} {stype:4s} gt={_fmt(truth):>8}  "
              f"dexta_err={_fmt(de):>8} plain_err={_fmt(be):>8}", flush=True)
    store.close()
    return {"id": spec["id"], "label": spec["label"], "seed": spec["seed"],
            "readings": len(g), "n_days": N_DAYS,
            "ground_truth": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in gt.items()},
            "items": items}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _mae(items: list[dict[str, Any]], arm: str) -> tuple[float | None, int]:
    errs = [it[arm]["abs_err"] for it in items
            if it["type"] in NUMERIC_TYPES and it[arm].get("abs_err") is not None]
    return (float(np.mean(errs)), len(errs)) if errs else (None, 0)


def aggregate(patients: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for p in patients:
        dmae, dn = _mae(p["items"], "dexta")
        bmae, bn = _mae(p["items"], "plain")
        rows.append({"id": p["id"], "label": p["label"],
                     "dexta_mae": dmae, "dexta_n": dn,
                     "plain_mae": bmae, "plain_n": bn})

    def dist(key: str) -> dict[str, Any]:
        vals = np.array([r[key] for r in rows if r[key] is not None])
        if not vals.size:
            return {}
        return {"median": round(float(np.median(vals)), 3),
                "iqr": [round(float(np.percentile(vals, 25)), 3),
                        round(float(np.percentile(vals, 75)), 3)],
                "worst": round(float(vals.max()), 3),
                "best": round(float(vals.min()), 3)}

    return {"per_patient": rows,
            "dexta_mae_dist": dist("dexta_mae"),
            "plain_mae_dist": dist("plain_mae"),
            "note": "AUTO-SCORE FLOOR ONLY. The number parser mis-picks on the "
                    "dexta arm's discursive multi-number answers (e.g. grabbing "
                    "'100% coverage' for a TBR question), inflating dexta's auto "
                    "MAE far above the truth. The hand-verified per-cell tables in "
                    "bench/LLMCGM_RESULTS.md are the source of truth; flags on each "
                    "cell mark what needs review. Patients whose model calls failed "
                    "(API errors) are excluded (n=0), not scored as wrong."}


def _patient_complete(pat: dict[str, Any]) -> bool:
    """A patient counts as done only if no cell in either arm is an API error."""
    return bool(pat.get("items")) and not any(
        it[arm].get("api_error") or _is_api_error(it[arm].get("answer", ""))
        for it in pat["items"] for arm in ("dexta", "plain")
    )


def _preflight(model: Any) -> bool:
    """One tiny call so a dead key/credit fails fast, not mid-cohort."""
    try:
        model.invoke([{"role": "user", "content": "ok"}])
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep already-complete patients from a prior run")
    args = ap.parse_args()

    built = []
    for spec in PATIENTS:
        p = build_patient(spec)
        p["spec"] = spec
        built.append(p)

    est = estimate(built)
    print("COST ESTIMATE (baseline arm, full-CSV input tokens):", flush=True)
    for row in est["per_patient"]:
        print(f"  {row['id']}: {row['readings']:5d} readings, "
              f"{row['csv_chars']:7d} CSV chars -> "
              f"{row['baseline_input_tokens_est']:,} input tokens "
              f"({est['n_questions']} questions)", flush=True)
    print(f"  TOTAL baseline input tokens (est): {est['baseline_input_tokens_est']:,} "
          f"(budget {TOKEN_BUDGET:,})", flush=True)
    if est["baseline_input_tokens_est"] > TOKEN_BUDGET:
        print("  OVER BUDGET: reduce N_DAYS per patient and re-run.", flush=True)
        return 1

    dest = Path(__file__).resolve().parent / "results"
    dest.mkdir(exist_ok=True)
    if args.estimate_only:
        (dest / "llmcgm_multi_estimate.json").write_text(json.dumps(est, indent=2))
        print("estimate-only: no model calls made.", flush=True)
        return 0

    prior: dict[str, dict[str, Any]] = {}
    result_path = dest / "llmcgm_multi.json"
    if args.resume and result_path.exists():
        for pat in json.loads(result_path.read_text()).get("patients", []):
            if _patient_complete(pat):
                prior[pat["id"]] = pat
        print(f"RESUME: reusing complete patients {sorted(prior)}", flush=True)

    model = get_model(SUBJECT)
    if not _preflight(model):
        print("Aborting before the cohort loop (no valid model access).", flush=True)
        return 1

    out: dict[str, Any] = {
        "benchmark": "LLM-CGM (Healey & Kohane, PSB 2025), github.com/lizhealey/LLM-CGM",
        "subject_model": f"{SUBJECT.provider}:{SUBJECT.model}",
        "temperature": SUBJECT.temperature,
        "arms": {
            "dexta": "orchestrator + tool belt + faithfulness/treatment rails",
            "plain": "same model, full CGM record as CSV in-context, no tools",
        },
        "n_patients": len(built), "n_days": N_DAYS,
        "cost_estimate": est,
        "generated_at": datetime.now().astimezone().isoformat(),
        "patients": [],
    }
    for p in built:
        pid = p["spec"]["id"]
        gt = ground_truth_np(p["glucose"])
        _crosscheck(p, gt)
        if pid in prior:
            print(f"\n=== {pid} (reused from prior complete run) ===", flush=True)
            out["patients"].append(prior[pid])
            continue
        print(f"\n=== {pid} {p['spec']['label']} "
              f"(seed={p['spec']['seed']}, {len(p['glucose'])} readings) ===", flush=True)
        res = run_patient(model, p, gt)
        out["patients"].append(res)
        (dest / "llmcgm_multi.json").write_text(json.dumps(out, indent=2, default=str))

    out["aggregate"] = aggregate(out["patients"])
    (dest / "llmcgm_multi.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n=== AGGREGATE ===", flush=True)
    for r in out["aggregate"]["per_patient"]:
        print(f"  {r['id']}: dexta MAE {_fmt(r['dexta_mae'])} (n={r['dexta_n']})  "
              f"plain MAE {_fmt(r['plain_mae'])} (n={r['plain_n']})", flush=True)
    print(f"  dexta dist: {out['aggregate']['dexta_mae_dist']}", flush=True)
    print(f"  plain dist: {out['aggregate']['plain_mae_dist']}", flush=True)
    print(f"\nWROTE {dest / 'llmcgm_multi.json'}", flush=True)
    return 0


def _crosscheck(p: dict[str, Any], gt: dict[str, Any]) -> None:
    """Assert the numpy ground truth matches the ported pure-Python formulas."""
    g = p["glucose"]
    ref = llmcgm.ground_truth([e.ts for e in g], [float(e.mg_dl) for e in g])
    for k, v in gt.items():
        rv = ref.get(k)
        if isinstance(v, float) and isinstance(rv, float):
            assert abs(v - rv) < 1e-6, f"{p['spec']['id']} {k}: np {v} != port {rv}"
        else:
            assert v == rv, f"{p['spec']['id']} {k}: np {v!r} != port {rv!r}"


if __name__ == "__main__":
    raise SystemExit(main())
