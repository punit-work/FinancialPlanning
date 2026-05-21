# DECISIONS — append-only log of non-obvious modelling and structural choices

Read before changing existing methodology. New entries at the top. Each entry: ISO date • short title • rationale • trade-off / when to revisit.

This file is seeded from the commit history that's actually in the repo today (2026-05-21). Going forward, append a new entry whenever a structural or modelling change goes in — the binary nature of `Glide Paths.xlsx` and the implicit nature of "why we chose this default" mean git log alone won't explain things later.

---

## 2026-05-21 — Glide paths stay in tranche-and-chain format, not target-allocation

When updating from `Glide Paths v2.xlsx`, we re-authored the new glide path values into the existing tranche-and-chain row format rather than rewriting `calculate_goal_cashflows()` to consume a target-allocation table.

**Why:** the target-allocation format ("at year -N, hold X% in Debt, Y% in Hybrid, rest in Equity") is silent on (a) how many tranches to split the goal into, (b) when each tranche enters each bucket, (c) the funding-source chain. The tranche-chain format encodes all three explicitly. Translating one to the other requires modelling assumptions that should be made by the human, not the simulator.

**Trade-off:** authoring a glide path in the chain format is more verbose and requires the editor to think in tranches.

**When to revisit:** if the team starts to author glide paths primarily in the new format and the chain format becomes a translation layer, rewrite the simulator to consume target weights and a rebalancing schedule.

---

## 2026-04-01 — All `Date` columns standardised to `datetime64[ns]`

Pandas `merge_asof` raises a dtype-mismatch error when the left and right `Date` columns have different time resolutions (e.g. `[ns]` vs `[us]`). The default resolution can shift between pandas versions and even between input paths (Excel read, Timestamp construction, date_range).

**Why:** rather than fix this at every merge site, we normalise at construction. `_NS_DTYPE = "datetime64[ns]"` and `_ensure_date_ns(df)` in `main_v2.py:11-17` are the convention. Streamlit Cloud independently hit the same issue, so `requirements.txt` pins `pandas>=3.0.0` to keep behaviour consistent across local + cloud.

**Trade-off:** any new code path that creates a DataFrame with a `Date` column must remember to call `_ensure_date_ns()` (or construct via `_ts()`). Forgetting it surfaces as `MergeError` deep inside the simulator.

**When to revisit:** if pandas ever fully unifies datetime arithmetic across resolutions, this convention can be relaxed.

---

## 2026-03-18 — Default `target_lifetime` lowered from 100 to 90

Changed in commit `d34bb03`. The previous default of 100 made the simulator size post-retirement pools (and Core Corpus runway) for a much longer tail than most users actually plan for, inflating the required retirement corpus and pushing the discovered retirement date later.

**Why:** 90 is a more representative planning horizon.

**Trade-off:** users planning for longevity-tail scenarios must explicitly raise the input. The UI surfaces this as a configurable field.

**When to revisit:** if users systematically ask for longer horizons, raise the default.

---

## 2026-02-23 — STCG / LTCG replaces flat per-bucket tax rate

Commit `b121263`. Each instrument bucket now carries `stcg_tax` and `ltcg_tax` rather than a single rate. Tax is determined per tax-lot at redemption based on holding period (≤ 365 days → STCG, > 365 → LTCG).

**Why:** the previous flat-rate model materially overstated tax on long-held core corpus lots and understated tax on short-term debt-pool churn. STCG/LTCG split mirrors Indian capital-gains rules and produces accurate per-redemption tax.

**Trade-off:** FIFO tax-lot accounting is more code (the `TaxLot` / `InvestmentPool` classes and the lot-walking logic in `add_withdrawls_to_trans()`). Worth it for accuracy.

**When to revisit:** if Indian tax rules change (e.g. revised LTCG rate, removal of indexation), update the per-bucket `stcg_tax` / `ltcg_tax` defaults in `find_retirement_date()` (`main_v2.py:1300-1307`) and log it here.

---

## 2026-02-22 — Removed 5-year-beyond-death post-retirement pool buffer

Commit `f6d83c0` (revert of `3ab82d9`). The simulator previously pre-funded post-retirement pools to 5 years beyond the death date as a conservative buffer; this was removed alongside the switch to showing total wealth (rather than core corpus only) on the chart.

**Why:** pools are now sized exactly to the death date. The "buffer" was hiding the genuine question of "does the corpus actually last?" by reserving extra capital.

**Trade-off:** the model treats the death date as a hard endpoint with no margin. Users who want a margin should raise `target_lifetime`.

---

## Open / pending decisions

(None tracked here yet. Add a stub entry the moment a decision is "we'll think about this later" so it doesn't get lost.)
