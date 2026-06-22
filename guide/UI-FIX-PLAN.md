# UI fix plan — phased execution

Ordered by demo ROI and dependency. Each phase is independently shippable; stop after each for a UI review before continuing.

**North star:** One job per page. One sync home. One investigate surface. Five nav tabs + More.

---

## Phase 1 — Unify investigate (P0)

**Problem:** Dashboard “Investigate” runs `CoordinatorAgent` (blocking deep sweep). Investigations + Chat run `OrchestratorAgent` (streaming tool trace). Same label, different engines.

**Deliverables:**
1. Remove dashboard investigate box and `POST /actions/investigate` redirect-to-dashboard flow (or redirect form to `/investigations` with query prefill).
2. Dashboard: single CTA **“Start an investigation →”** linking to `/investigations`.
3. Fix Investigations empty state (stop pointing at dashboard box).
4. Update dashboard investigate flash banners / dead routes if any.

**Acceptance:**
- [x] No investigate form on dashboard
- [x] All question-driven drill-down starts on Investigations (stream) or Chat (conversation)
- [x] Deep analysis remains on Investigations only (`Run deep analysis` button)
- [x] Tests updated; no broken POST handlers

**UI review checklist:**
- Dashboard above-the-fold: metrics → hero chart → Sync/Analyze → findings (no duplicate ask path)
- Investigations: stream form is the only “investigate” entry for structured Q&A

---

## Phase 2 — Consolidate sync (P0)

**Problem:** Six sync entry points (dashboard ×2, Connectors ×2, Settings per-source, sidebar copy).

**Deliverables:**
1. Dashboard: replace **Sync now** with link **“Sync in Connectors →”** (or one secondary ghost link); remove empty-state duplicate sync button.
2. Sidebar: copy points to Connectors, not dashboard Sync now.
3. Connectors page: clarify as **the** sync home (manual + autosync); optional one-line on dashboard when autosync is on.
4. Keep Settings per-source sync (credential test flow) but de-emphasize in copy (“after saving, sync from Connectors”).

**Acceptance:**
- [x] At most one sync action on dashboard (link, not POST duplicate)
- [x] Connectors owns manual + continuous sync
- [x] No conflicting “Sync now” in three places on same screen

**UI review checklist:**
- Fresh user: Settings → Connectors → Sync all is obvious path
- Autosync status visible on Connectors; dashboard doesn’t imply a second sync system

---

## Phase 3 — Dashboard diet (P0)

**Problem:** Seven action-bar CTAs; findings feed uncapped; card markup differs from Findings page.

**Deliverables:**
1. Action bar: **Sync link** + **Run analyze** (with lens) only; **More ▾** dropdown: Upload CSV, Rebuild wiki, Log context, Context prompts, Chat.
2. Cap dashboard active findings to top 5 + “View all on Findings →”.
3. Reuse `finding_card` macro on dashboard feed.
4. Move CSV upload hint under More or Connectors sidebar callout.

**Acceptance:**
- [x] ≤2 primary buttons visible without opening More
- [x] Dashboard loads ≤5 finding cards
- [x] Finding cards match Findings page vocabulary (scope, strength, lifecycle)

**UI review checklist:**
- First screenful: metrics, chart, 2 buttons, top findings — no action-bar wrap on 1280px

---

## Phase 4 — Nav tiering (P1)

**Problem:** Nine flat nav tabs; orphan pages (Log, Context, Wiki, Reconciliation, Evals).

**Deliverables:**
1. Primary nav: Dashboard · Chat · Findings · Connectors · Settings.
2. **More ▾** menu: Investigations, Goals, Reports, System, Wiki, Reconciliation, Log context, Evals (from System).
3. Active state works for More child routes.
4. Remove duplicate ghost links that replicate nav (dashboard “Ask a question” if Chat is in nav).

**Acceptance:**
- [x] Nav fits one line at 1280px without awkward wrap
- [x] Every major page reachable in ≤2 clicks
- [x] Reconciliation reachable without hunting Findings tabbar

**UI review checklist:**
- Click through every More item; active highlight correct
- Mobile: More collapses or scrolls gracefully (min: horizontal scroll with fade, ideal: hamburger)

---

## Phase 5 — Trace + cards consistency (P1)

**Problem:** Live streams have timeline; persisted investigation runs use plain `<ol>`. Faithfulness chips only on stream answer.

**Deliverables:**
1. Server partial `_trace_timeline.html` for persisted runs on Investigations page.
2. Optional: faithfulness row on stored investigation runs when answer was flagged.
3. Investigations empty state + run cards use same trace visual language as stream.js.

**Acceptance:**
- [x] Historical run trace looks like live stream trace
- [x] No regression on investigate.js streaming

**UI review checklist:**
- Run an investigation, refresh page — trace styling matches live run

---

## Phase 6 — Performance (P2)

**Problem:** `coverage()` on every page for status pill; reconciliation agent on GET; unbounded findings fetch.

**Deliverables:**
1. Cache status pill on `app.state` with TTL or update on sync/analyze only.
2. Reconciliation: read cached findings from store or run agent on POST/analyze only; GET renders last result or empty.
3. Dashboard/findings: cap store query (e.g. 50 active, display 5).

**Acceptance:**
- [x] Reconciliation page load does not invoke agent
- [x] Dashboard GET bounded query
- [x] Status pill still accurate after sync

**UI review checklist:**
- Reconciliation page loads instantly on repeat visit
- Dashboard still shows correct “Local only · N days” pill after sync

---

## Phase 7 — Goals + context polish (P2)

**Deliverables:**
1. Dashboard sidebar strip: active goals count + link to Goals; next check if due.
2. Merge Context into Log flow or single “Log / missing context” nav entry.
3. Goals page: show Tick even when empty with explanation (or dashboard link to tick).

**UI review checklist:**
- Goals discoverable without reading docs
- Log vs Context not confusing

---

## Execution protocol

1. Implement one phase per PR-sized diff.
2. Run targeted tests + `pytest tests/test_server.py` after each phase.
3. Restart `dexta serve`; hard-refresh browser; walk the phase **UI review checklist**.
4. Stop for human review before starting next phase.
