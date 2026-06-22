# UI enhancement roadmap

The intelligence, rigor, and safety are strong; the UI is the lagging surface.
Today it is server-rendered HTML with small inline-SVG sparklines and no real
glucose-trace chart. For a demo to a sophisticated audience, the gap is *show*,
not substance. This roadmap is ordered by demo ROI.

Design constraint: stay on the existing htmx + server-rendered stack (no SPA
rewrite). Charts should be server-rendered inline SVG (deterministic, no client
framework), consistent with the current sparkline approach.

## P0 - the hero visualization

The single biggest lift. One legible chart that tells the story at a glance.

1. **Glucose trace with attribution overlay.** The minute-level CGM trace around
   a spike, with the spike shaded, the target band drawn, and event markers
   (bolus, carb entry) placed on the timeline. The attribution ("late bolus,
   +22 min") annotated on the trace. Source data already exists
   (`zoom_event`, `get_boluses`, `get_carb_entries`).
2. **Reconciliation expected-vs-actual curves.** Overlay the loop's predicted
   curve (OpenAPS/Loop `devicestatus`) against realized glucose, with the
   divergence highlighted. The reconciliation engine already produces this data;
   it just has no chart.

Implementation: a small server-side SVG path builder (reuse the sparkline
polyline approach, scaled up) takes a series of (ts, value) plus markers and
emits an `<svg>`. No new dependency.

## P1 - make the agentic work legible

3. **Trace timeline component.** Render the plan to trace to evidence to skeptic
   as a vertical timeline (it exists as text today). Each tool call as a step
   with its scope and result; the skeptic's counter-evidence as a distinct
   block. This visualizes the differentiator.
4. **Faithfulness-guard surfacing.** When the guard rejects a number, show a
   small "claim rejected: not traceable" chip in the trace, so the safety rail
   is visible, not silent.

## P2 - polish

5. **AGP (ambulatory glucose profile)** percentile band chart on the dashboard
   (the consensus standard clinicians expect).
6. **Findings evidence cards** with a mini sparkline of the effect.
7. **Reports** rendered for print/PDF (clinician hand-off).

## Testing

Charts are pure functions (series in, SVG string out): unit-test the path
builder on fixed inputs (same pattern as the existing sparkline tests). No
visual-regression infra needed for v1.

## Sequencing

P0 first (the hero chart is the demo centerpiece), then P1 (the trace timeline
is what makes the agentic story land). P2 is post-demo polish.
