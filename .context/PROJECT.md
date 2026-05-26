# PROJECT — Financial Planning v3

## Purpose

Personal retirement-and-goals planner. Given a user's current corpus, monthly SIP (with step-up), recurring expenses, optional passive income streams, and a set of goals (each with a future-value target, maturity date, and goal type), find the **earliest month** at which the user can stop their SIP and still:

1. Fund every goal in full at its maturity date.
2. Cover every post-retirement expense until the user's death-date (`current_age + (target_lifetime - current_age)`).
3. Never run the Core Corpus or the post-retirement Debt pool below zero.

The answer is delivered through `find_retirement_date()` and rendered in the Streamlit UI as a year/month, a corpus chart, per-goal funding status, and a comprehensive month-by-month dataframe.

## Stack

| Layer | File | Notes |
|---|---|---|
| Simulator | `main_v2.py` | Pure logic, no UI. ~1,400 lines. Standalone entry via `python main_v2.py`. |
| UI | `streamlit_app.py` | Streamlit app. Collects inputs into a `config` dict and calls `main_v2.find_retirement_date()` / `main_v2.run_simulation()`. |
| Glide paths | `Glide Paths.xlsx` | Three sheets (`Non-Negotiable`, `Semi-Negotiable`, `Negotiable`). Format = tranche-chain (see SIMULATION_MODEL.md). |
| Audit trail | `GLIDE_PATHS_CHANGELOG.md` | Human-readable diff log for the glide paths file (binary, so git diffs are opaque). |
| Deps | `requirements.txt` | `streamlit`, `pandas`, `numpy`, `python-dateutil`, `openpyxl`, `xlsxwriter`. Pandas pinned `>=3.0.0` (see DECISIONS.md). |

## Current state (2026-05-21)

- Glide paths refreshed today: tranche restructure across all three goal types. See `GLIDE_PATHS_CHANGELOG.md` for the row-level diff.
- Tax model: STCG 20%, LTCG 12.5% for all equity-like buckets (core_corpus / equity / debt / hybrid / cash all share the same STCG/LTCG split — the differentiator is the return assumption).
- Default return assumptions (`find_retirement_date()` defaults, aligned with the Streamlit UI prefill on 2026-05-21):
  - core_corpus / equity: **12%** annual
  - hybrid: **10%** annual
  - debt: **6%** annual
  - cash: **4%** annual (UI does not expose; code only)
- Default `target_lifetime = 90` (changed from 100 in commit `444476f`-era).

## Key concepts (quick reference — full model in SIMULATION_MODEL.md)

- **Core Corpus** = the single pre-retirement equity-like pool. All SIP flows in, all goal pre-funding and post-retirement pool refills flow out.
- **Three goal types** = `Non-Negotiable`, `Semi-Negotiable`, `Negotiable`. Each has its own glide path sheet describing how money de-risks from Core Corpus → Hybrid → Debt → Goal as the maturity date approaches.
- **Post-retirement pools** = a **Debt pool** (funds the next 24 months of expenses) and a **Hybrid pool** (funds months 25–60). Refilled annually from Core Corpus.
- **Passive income** = optional streams that grow at one rate pre-retirement and another rate post-retirement; netted against monthly expenses inside the post-retirement loop.
- **Pseudo NAV** = synthetic NAV series produced by `generate_pseudo_nav()` — daily compounding from the annual return rate. The simulator doesn't use real market data; it uses these smooth compounding curves so the model is deterministic.

## Files in this folder

| File | Purpose |
|---|---|
| `main_v2.py` | Simulator (TaxLot, InvestmentPool, cashflow builders, post-retirement loop, retirement-date binary search). |
| `streamlit_app.py` | Streamlit UI. |
| `Glide Paths.xlsx` | The three glide-path sheets, loaded by `get_default_glide_paths()`. |
| `requirements.txt` | Pinned dependencies. |
| `GLIDE_PATHS_CHANGELOG.md` | Audit log for changes to the xlsx. |
| `.context/PROJECT.md` | This file. |
| `.context/SIMULATION_MODEL.md` | How the simulator actually works, end-to-end. |
| `.context/DECISIONS.md` | Append-only "why" log for non-obvious modelling choices. |

## Deployment

Streamlit Cloud (`streamlit run streamlit_app.py` locally; deployed equivalent on Streamlit Cloud). The pandas `>=3.0.0` pin in `requirements.txt` exists because Streamlit Cloud resolved an older pandas that broke `merge_asof` on Date columns.

## Out of scope

- Multi-user / persisted state. Session state lives only in the Streamlit `st.session_state` for the current browser session.
- Real market data / NAVs. Returns are deterministic via `generate_pseudo_nav()`.
- Tax regimes other than India STCG/LTCG.
- Stochastic / Monte Carlo simulation. The model is single-path, deterministic.
