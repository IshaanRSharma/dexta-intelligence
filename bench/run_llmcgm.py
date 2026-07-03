"""dexta on the LLM-CGM benchmark (Healey & Kohane, PSB 2025).

LLM-CGM (github.com/lizhealey/LLM-CGM, MIT) is a published benchmark of 30
question-answering tasks over a CGM time series, each with a deterministic
ground-truth answer computed from the data. The repo ships the questions + the
ground-truth computation, not the data (it takes the user's dataframe), so we
feed it CGM from dexta's own synthetic generator and score the harness's answers
against the benchmark's OWN ground-truth formulas (ported faithfully below from
their `get_answers`, pure-Python so no pandas dependency).

This is an EXTERNAL benchmark: the questions and the scoring definitions are
theirs, not ours. We run the dexta orchestrator (the harness arm) on a curated
spread of the 30 tasks and score the deterministically-scorable ones, dumping
every answer for hand-verification. Honest scope: synthetic data (their real
data is restricted), one patient, single pass - a controlled external-validity
probe, not a clinical claim.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench import run_ladder as ladder  # reuse _ctx, harness_arm, store builders

from dexta_intelligence.llm.factory import ModelSpec, get_model
from dexta_intelligence.testing.synthetic import scenario_sensitivity_shift

# ── ground truth: ported from LLM-CGM get_answers (MIT, Healey & Kohane) ───────


def _runs(flags: list[bool]) -> list[int]:
    """Lengths of consecutive True runs (for hypo/hyper episodes)."""
    out: list[int] = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _part_of_day(h: int) -> str:
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 24:
        return "evening"
    return "night"


def ground_truth(ts: list[datetime], cgm: list[float]) -> dict[str, Any]:
    """The 30 LLM-CGM answers, computed exactly as the benchmark defines them."""
    n = len(cgm)
    mean = statistics.mean(cgm)
    sd = statistics.pstdev(cgm)  # np.std default ddof=0
    imax = max(range(n), key=lambda i: cgm[i])
    imin = min(range(n), key=lambda i: cgm[i])
    by_date: dict[date, int] = {}
    for t, v in zip(ts, cgm, strict=True):
        if v < 70 or v > 180:
            by_date[t.date()] = by_date.get(t.date(), 0) + 1
    hypo_flags = [v < 70 for v in cgm]
    hyper_runs = _runs([v > 180 for v in cgm])
    hypo_runs = _runs(hypo_flags)
    night = [v for t, v in zip(ts, cgm, strict=True) if 0 <= t.hour < 6]
    dinner = [v for t, v in zip(ts, cgm, strict=True) if 17 <= t.hour < 22]
    most_recent = max(t.date() for t in ts)
    yest = most_recent - timedelta(days=1)
    today_v = [v for t, v in zip(ts, cgm, strict=True) if t.date() == most_recent]
    yest_v = [v for t, v in zip(ts, cgm, strict=True) if t.date() == yest]
    last_low_ts = max((t for t, v in zip(ts, cgm, strict=True) if v < 70), default=None)
    return {
        "Q1_mean": mean,
        "Q2_max": max(cgm),
        "Q3_std": sd,
        "Q4_min": min(cgm),
        "Q5_TIR_pct": 100.0 * sum(1 for v in cgm if 70 < v < 180) / n,
        "Q6_TAR180_pct": 100.0 * sum(1 for v in cgm if v > 180) / n,
        "Q7_TBR70_pct": 100.0 * sum(1 for v in cgm if v < 70) / n,
        "Q8_CV": sd / mean,
        "Q9_TAR250_pct": 100.0 * sum(1 for v in cgm if v > 250) / n,
        "Q10_eA1c": (46.7 + mean) / 28.7,
        "Q11_TBR54_pct": 100.0 * sum(1 for v in cgm if v < 54) / n,
        "Q12_highest_time": ts[imax].strftime("%H:%M"),
        "Q14_lowest_time": ts[imin].strftime("%H:%M"),
        "Q17_longest_hyper_min": (max(hyper_runs) * 5) if hyper_runs else 0,
        "Q18_num_hypo": len(hypo_runs),
        "Q19_overnight_mean": statistics.mean(night) if night else None,
        "Q20_period_highest": _part_of_day(ts[imax].hour),
        "Q21_nocturnal_hypo": any(v < 70 for v in night),
        "Q22_dinner_max": max(dinner) if dinner else None,
        "Q15_last_low_ts": last_low_ts.isoformat() if last_low_ts else None,
        "Q27_today_better": (statistics.mean(today_v) < statistics.mean(yest_v))
        if today_v and yest_v else None,
        "Q29_max_lower_today": (max(today_v) < max(yest_v)) if today_v and yest_v else None,
    }


# ── the curated subset we run the model on (clean to score) ────────────────────

QUESTIONS: dict[str, tuple[str, str, str]] = {
    # key: (question text, ground-truth key, score type)
    "Q1": ("What was my mean glucose over this period (mg/dL)?", "Q1_mean", "num"),
    "Q2": ("What was my maximum glucose reading (mg/dL)?", "Q2_max", "num"),
    "Q4": ("What was my minimum glucose reading (mg/dL)?", "Q4_min", "num"),
    "Q5": ("What percent of time was I in range (70-180 mg/dL)?", "Q5_TIR_pct", "pct"),
    "Q7": ("What percent of time was I in hypoglycemia (below 70)?", "Q7_TBR70_pct", "pct"),
    "Q8": ("What was my glycemic variability (coefficient of variation)?", "Q8_CV", "cv"),
    "Q10": ("What is my estimated A1C?", "Q10_eA1c", "num"),
    "Q12": ("What time of day was my blood glucose highest?", "Q12_highest_time", "hour"),
    "Q17": ("What was my longest continuous time spent in hyperglycemia (minutes)?",
            "Q17_longest_hyper_min", "num"),
    "Q18": ("How many separate times did I experience hypoglycemia?", "Q18_num_hypo", "int"),
    "Q19": ("What was my mean overnight (midnight-6am) blood glucose?",
            "Q19_overnight_mean", "num"),
    "Q20": ("What period of the day did I have the highest blood glucose "
            "(morning/afternoon/evening/night)?", "Q20_period_highest", "cat"),
    "Q21": ("Did I have any nocturnal hypoglycemia (a low between midnight and 6am)?",
            "Q21_nocturnal_hypo", "bool"),
    "Q22": ("What was my highest glucose reading during dinner (5pm-10pm)?",
            "Q22_dinner_max", "num"),
    "Q27": ("Was my average glucose control today better than yesterday?",
            "Q27_today_better", "bool"),
}

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?")


def _nums(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(text)]


def score(stype: str, gt: Any, answer: str) -> bool | None:  # noqa: PLR0911
    """Deterministic scoring per question type. None = needs hand-review."""
    if gt is None:
        return None
    low = answer.lower()
    if stype in ("num", "int", "pct", "cv"):
        target = gt
        if stype == "pct":
            target = gt  # gt already a percent; answer expected in percent
        if stype == "cv":
            # accept either fraction (0.23) or percent (23) for CV
            cands = _nums(answer)
            return any(
                abs(c - gt) <= max(0.02, 0.08 * gt) or abs(c - gt * 100) <= max(2.0, 8.0)
                for c in cands
            )
        cands = _nums(answer)
        if stype == "int":
            return any(abs(c - target) < 0.5 for c in cands)
        tol = max(1.0, 0.05 * abs(target)) if stype == "num" else max(2.0, 0.08 * abs(target))
        return any(abs(c - target) <= tol for c in cands)
    if stype == "cat":
        return str(gt).lower() in low
    if stype == "bool":
        yes = any(w in low for w in ("yes", "you did", "there was", "present", "you had"))
        no = any(w in low for w in ("no ", "no.", "did not", "no nocturnal", "not better",
                                    "no episodes", "none", "was not"))
        if yes == no:
            return None  # ambiguous -> hand-review
        return yes == bool(gt)
    if stype == "hour":
        # accept an hour within +/-1 of the ground-truth hour
        gh = int(gt.split(":")[0])
        hours = [int(h) for h in re.findall(r"\b([01]?\d|2[0-3])\s*(?::00)?\s*(?:am|pm)?", low)]
        return any(abs(h - gh) <= 1 for h in hours) if hours else None
    return None


def main() -> int:
    model = get_model(ModelSpec("anthropic", "claude-sonnet-4-6", 0.0, 1800))
    # 21 days, a sensitivity shift with a real high+low spread (TIR ~80%, TAR ~15%,
    # TBR ~4%) so the hyper/hypo/temporal questions have meaningful ground truth.
    events, _ = scenario_sensitivity_shift(seed=5, n_days=21, effect_size=60.0)
    g = events["glucose"]
    ts = [e.ts for e in g]
    cgm = [float(e.mg_dl) for e in g]
    gt = ground_truth(ts, cgm)

    store = ladder.store_from_events(events)
    ctx = ladder._ctx(store)

    out: dict[str, Any] = {
        "benchmark": "LLM-CGM (Healey & Kohane, PSB 2025), github.com/lizhealey/LLM-CGM",
        "subject": "anthropic:claude-sonnet-4-6 via dexta harness",
        "data": f"dexta synthetic sensitivity_shift seed=5 e60, 21 days, {len(cgm)} readings",
        "ground_truth": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in gt.items()},
        "items": [],
    }
    n_scored = n_correct = 0
    for key, (qtext, gtkey, stype) in QUESTIONS.items():
        t0 = time.time()
        try:
            res = ladder.harness_arm(model, ctx, qtext)
            ans = res["text"]
            stop = res["stopped_reason"]
        except Exception as exc:
            ans, stop = f"ERROR: {exc}", "error"
        verdict = score(stype, gt.get(gtkey), ans)
        if verdict is not None:
            n_scored += 1
            n_correct += int(verdict)
        out["items"].append({
            "q": key, "question": qtext, "gt": gt.get(gtkey), "type": stype,
            "verdict": verdict, "stopped_reason": stop, "seconds": round(time.time() - t0, 1),
            "answer": ans,
        })
        gv = gt.get(gtkey)
        gvs = round(gv, 2) if isinstance(gv, float) else gv
        print(f"  {key:4s} {verdict!s:5s}  gt={gvs}  ({round(time.time()-t0,1)}s)", flush=True)
    out["auto_scored"] = f"{n_correct}/{n_scored}"
    print(f"\nAUTO-SCORED: {n_correct}/{n_scored} (rest need hand-review from the dump)",
          flush=True)

    dest = Path(__file__).resolve().parent / "results"
    dest.mkdir(exist_ok=True)
    (dest / "llmcgm_raw.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"WROTE {dest / 'llmcgm_raw.json'}", flush=True)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
