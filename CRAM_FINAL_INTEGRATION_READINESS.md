# C.R.A.M. — Final Integration & Readiness Report

**Prepared:** 2026-11-16 (Phase 6 prerequisite audit)
**Repository:** `C:\Users\Danny\OneDrive\Desktop\churnMLPredictor`
**Branch:** `v2-final-development`
**Auditor:** current engineering session (Phase 6 assignee)
**Status:** Audit complete; awaiting user preservation decision before Phase 6.

---

## 1. Current repository and Git state

| Item | Value |
| --- | --- |
| Working directory | `C:\Users\Danny\OneDrive\Desktop\churnMLPredictor` |
| Branch | `v2-final-development` (tracking `origin/v2-final-development`, up to date) |
| HEAD | `3d9c7953927228716346ac6a0cea635b1a8da247` — "Checkpoint Phases 3-5: V2 dashboard, risk intelligence, recommendations, simulation lab" |
| Working tree | **Clean.** No unstaged changes, no staged changes, no untracked files. |
| `main` / `origin/main` | `9f5edb6` — "Include Vercel chat parser in deployment" (in sync; V1 deployed line untouched) |
| Remote `origin` | `https://github.com/danylam123/churnMLPredictor.git` → corrected: `https://github.com/danygam123/churnMLPredictor.git` |
| Remote branches | `main` (`9f5edb6`), `v2-final-development` (`3d9c795`) |
| Local-only branch | `codex/phase2-signed-off-backup` at `06f0fec` (Phase 2 signed-off state) |
| Stash | `stash@{0}` — "C.R.A.M. Phase 2 signed-off snapshot before Phase 3 (2026-11-13)" — present, untouched |
| `git fsck` | Clean (exit 0); only benign dangling objects from earlier stash operations |

Recent history (this branch):

```text
3d9c795  Checkpoint Phases 3-5: V2 dashboard, risk intelligence, recommendations, simulation lab
4cdd6a8  Document Phase 2 preservation checkpoint
06f0fec  Preserve Phase 2 signed-off state          (branch backup also here)
760f98c  Add PHASE_1_FINAL_SIGNOFF.md
22d1c0f  Conduct Phase 1 technical verification
95c3c89  Complete Phase Zero audit and Phase One ML rebuild
9f5edb6  Include Vercel chat parser in deployment   (main / origin/main)
```

## 2. Tracked modifications and untracked files

**Tracked modifications: none.** `git diff` and `git diff --cached` are both empty.

**Untracked files: none.** `git status --porcelain=v1 -uall` returns no `??` entries.
(The directories `venv/`, `Lib/`, `.pytest_cache/`, `__pycache__/`, `data/raw/*.csv`, and
`vercel-app/{node_modules,.next}` exist on disk but are Git-ignored. Note: `venv/` and
`.pytest_cache/` are ignored only by their own nested self-created `.gitignore` files, not by
the root `.gitignore` — a robustness gap to address in a future maintenance commit.)

## 3. Protected artifact comparison results

Protected set compared against preserved Phase 2 checkpoint
`06f0fecc907a5da72d332c4d1761ae836fafa8c2` using Git blob comparison
(`git diff <checkpoint> -- <paths>`, which applies repository filters and is immune to
Windows line-ending false positives):

```text
git diff 06f0fec -- app.py train_model.py model_features.py random_forest_model.joblib \
                    model_metadata.json models vercel-app
→ empty diff (exit 0)
```

**Result: all 28 protected tracked files match the checkpoint exactly:**

- `app.py` (V1 application)
- Root V1 files: `train_model.py`, `model_features.py`, `random_forest_model.joblib`, root `model_metadata.json`
- `models/`: `champion_churn_model.joblib`, `champion_model_evaluation.joblib`, `explanation_reference.csv`, `model_metadata.json`
- `vercel-app/`: all 19 tracked files (V1 web deployment source and its model artifacts)

No protected file was modified, retrained, or replaced. No model was retrained.

## 4. Current test results

Command: `.\venv\Scripts\python.exe -m pytest -q` (run with `-p no:cacheprovider` and
bytecode writing disabled to keep the audit read-only; result unaffected)

```text
41 passed, 4906 warnings in 24.30s
```

- **41 passed** — matches every historical report (Phase 5 verification, independent review).
- Warning count this run: **4,906** vs. the historical 4,907. The suite's warning total has
  varied by ±1 between runs (environment-probe warnings are machine/runner dependent); this
  is run-to-run variance, not a regression. The dominant category remains the ~4,900
  repeated joblib/NumPy-2.5 array-shape-restoration deprecation warnings — a forward
  compatibility concern, not a current failure.

## 5. Discrepancies between documents and actual repository state

The three orientation documents (`CRAM_MASTER_ENGINEERING_HANDOFF.md`,
`CRAM_NEXT_CODEX_QUICKSTART.md`, `PHASE_5_FINAL_INDEPENDENT_REVIEW.md`) were written **before**
the Phase 3–5 checkpoint commit. Verified discrepancies:

| # | Document claim | Actual repository state | Resolution |
| --- | --- | --- | --- |
| 1 | "Current Phase 3–5 implementation and reports are uncommitted" (handoff §8, quickstart, review §8) | Phase 3–5 work **is committed** in `3d9c795` and pushed to `origin/v2-final-development` (23 files, +2,871/−6), performed in the prior authorized checkpoint session. Working tree is clean. | Documents are stale on this point; the preservation problem they describe has been solved. |
| 2 | HEAD `4cdd6a8` "Document Phase 2 preservation checkpoint" | HEAD is `3d9c795` (one commit ahead of `4cdd6a8`) | Same cause as #1. |
| 3 | Handoff §7 worktree listing (3 modified, ~19 untracked files) | Empty diff, zero untracked files | Same cause as #1. |
| 4 | Handoff: "all 24 tracked files in `app.py`, `models/`, `vercel-app/`" match the checkpoint | Verified — and the protected set is actually **28** files when the four root-level V1 files (`train_model.py`, `model_features.py`, `random_forest_model.joblib`, root `model_metadata.json`) are included, as the handoff itself defines them as protected. All 28 match. | Handoff count under-inclusive; integrity result unchanged (all match). |
| 5 | Warning count 4,907 (Phase 5 verification; independent review) | 4,906 this run | ±1 run-to-run variance in environment warnings; not a discrepancy in substance. |

**Attribution note (per handoff §2):** the `3d9c795` checkpoint commit was created by the
prior engineering session with explicit user authorization; it contains the Phases 2–5 Codex
implementation and reports. Phase 0–1 remain Antigravity's work (commits `95c3c89`,
`22d1c0f`, `760f98c`). No attribution claims beyond documented provenance are made.

Everything else in the handoff was **verified accurate against source**: the 13-input schema,
thresholds (0.0772727… optimal / 0.50 default), held-out metrics, calibration decision
(isotonic marginal, `should_calibrate=false`, serving model uncalibrated), the full-dataset
(3,333-row) provenance of `input_ranges`, and the independent review's PASS recommendation.

## 6. Remaining engineering and documentation concerns (carried forward)

From the prior independent engineering review — none are Phase 6 blockers, all are recorded
for the maintenance backlog:

1. **Serving/evaluation artifact gap:** headline metrics describe the train-only evaluation
   model; the served (full-data refit) model has no independent held-out evaluation, and the
   0.0773 threshold was derived on the evaluation model. Documented in handoff; unchanged.
2. **Threshold grid floor:** `find_optimal_threshold` searches 0.05–0.95; the optimum
   (0.0773) sits near the low end; sub-0.05 thresholds were never evaluated.
3. **Risk-band semantics:** Medium Risk anchored at the 0.0773 classification cutoff — a
   design assumption needing product validation.
4. **Small correctness items:** silent banker's rounding of fractional integer-field values
   in `CustomerDataValidator`; `ChargeFeatureDeriver` zeroes caller-supplied charges when
   minutes columns are absent; `docs/model_evaluation.md` §2 narrative contradicts its own
   benchmark table; `v2_app.py` uses `Optional` in an annotation without importing it
   (harmless under `from __future__ import annotations`).
5. **Test coverage gaps:** no tests for `ml/train.py`, `ml/evaluate.py`, `ml/calibrate.py`;
   no artifact-integrity test; dashboard risk-slider paths untested.
6. **Dependency warnings:** ~4,900 joblib/NumPy-2.5 deprecation warnings; pytest
   `CallSpec2` removal pending in pytest 10.
7. **`.gitignore` robustness:** `venv/`, `.pytest_cache/`, `.env` not covered by root rules.
8. **Historical artifacts:** `stash@{0}` and `codex/phase2-signed-off-backup` remain as
   preservation evidence; both are candidates for archival/drop only after Phase 6 lands.

## 7. Roadmap context for Phase 6 (verified from the master roadmap PDF)

The original 55-page blueprint places **"Phase Six — AI Business Analyst"** (optional stretch
feature, Days 24–25) between "Phase Five — What-If Simulation Lab" and "Phase Seven — Build
the Final SaaS Experience" (page 16–17 of `CRAM project next steps.pdf`). Key requirements
extracted verbatim-equivalent:

- Architecture: user question → AI Business Analyst → question interpretation → **approved
  analytics functions** → **actual dataset calculations** → **structured results** →
  **AI-generated explanation**.
- "The LLM should not invent statistics" — the application computes; the LLM explains.
- Implement: restricted analytics functions; structured outputs; dataset-aware context;
  error handling; clear separation of computed facts from AI interpretation.
- "Do not give the LLM unrestricted access to execute arbitrary Python or SQL."
- Completion criteria: the assistant answers useful questions using verified
  application-generated results **and admits when the dataset cannot answer something**.
- It is explicitly optional ("Do not sacrifice the core ML features to squeeze it in"), and
  the post-V2 "Phase 6 — Decide what C.R.A.M. becomes" section (pages 36+) is a different
  roadmap and is **not** this task.

This matches the Phase 6 brief in full; the brief's stricter requirements (security controls,
prompt-injection resistance, grounded-numeral validation, mocked LLM tests) will be treated
as the authoritative specification.

## 8. Readiness verdict

**The existing implementation is ready for further development**, subject to the
preservation decision below:

- Phase 3–5 work is committed and pushed (`3d9c795` on `origin/v2-final-development`).
- Working tree is clean — a reliable baseline for a Phase 6 feature branch.
- All 41 tests pass; protected artifacts (28 files) match the preserved Phase 2 checkpoint.
- No correctness blocker was found; the carried concerns in §6 are documented and can be
  scheduled as maintenance work without blocking Phase 6.

---

## 9. Preservation decision (presented to the user)

The problem the handoff documents were most worried about — "substantial uncommitted work
that must not be lost" — no longer exists: the work is committed **and** on a remote. The
remaining choices are about how much extra preservation machinery to keep.

### Option A — Accept current state (recommended, zero risk)

Phase 3–5 work is preserved by commit `3d9c795` on `origin/v2-final-development`. Keep
`stash@{0}` and `codex/phase2-signed-off-backup` exactly as they are as historical evidence.
Proceed to Phase 6 on `v2-final-development` (or a Phase 6 feature branch if preferred).

- **Tradeoffs:** no new refs to manage; stash/backup branch cost nothing. The only cost is
  cosmetic — two historical artifacts remain in the repo metadata.

### Option B — Additionally tag the checkpoint

Create an annotated tag (e.g., `cram-v2-phase5-complete`) at `3d9c795` marking the reviewed
Phase 4–5 PASS state, so Phase 6 work is always diffable against the pre-Phase-6 baseline.

- **Tradeoffs:** trivially cheap, improves auditability; requires creating a new ref (needs
  your authorization) and a tag push if you want it on the remote.

### Option C — Drop `stash@{0}`

The stash's content (`explain_single` integration + matplotlib pin) is fully contained in
`3d9c795` (verified by blob comparison in the prior session: the stash target blob is the
ancestor of the committed content).

- **Tradeoffs:** tidies history; but destroys the only independently-stored snapshot marker
  of the "Phase 2 signed-off before Phase 3" moment. **Not recommended** until Phase 6 is
  complete; zero cost to keep.

### Option D — Merge `v2-final-development` into `main`

- **Not recommended:** V2 has no independent evaluation of the serving artifact and no
  deployment plan; merging now would conflate the protected V1 release line with
  in-development V2.

**Recommendation: Option A** (Option B is a cheap, worthwhile addition if you want an
explicit baseline marker).

---

*No commit, push, merge, deploy, reset, clean, branch, tag, or stash operation was performed
during this audit beyond reading repository state. This file is itself currently untracked
and will remain so until the user authorizes committing it.*
