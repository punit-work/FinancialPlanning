# Financial Planning v3 — Session Bootstrap

Personal retirement-and-goals planner. Streamlit UI (`streamlit_app.py`) on top of a single-module simulator (`main_v2.py`). User describes their corpus, SIP, expenses, passive income, and goals; the simulator finds the earliest retirement date for which the plan still funds every goal and every post-retirement expense without depleting the corpus.

## Auto-loaded context

Inlined at session start — treat both as authoritative for current state and how the model works:

@.context/PROJECT.md
@.context/SIMULATION_MODEL.md

## On-demand context (read when relevant)

- `.context/DECISIONS.md` — append-only log of why specific modelling/structural choices were made. Read before changing existing methodology so you understand the "why".
- `GLIDE_PATHS_CHANGELOG.md` — human-readable history of changes to `Glide Paths.xlsx` (the binary diffs in git are opaque, so this is the audit trail).
- `~/.claude/CLAUDE.md` — global Snowflake key-pair auth + scheme knowledge (already in context). This project doesn't currently touch Snowflake but the file is loaded anyway.

## Routing by task type

- **"Why does the retirement date come out at X?" / "Explain this result"** → start from SIMULATION_MODEL.md (mental model), then drill into `run_simulation()` in `main_v2.py`. The flow is: build goal cashflows → build SIP / expense / passive-income series → run post-retirement pool simulator → settle Core Corpus transactions → check success.
- **"Update the glide paths"** → follow the protocol in `GLIDE_PATHS_CHANGELOG.md` (smoke-test against the previous file, log the diff in the changelog, then commit). The format is a tranche-and-chain table, not a target-allocation table — see SIMULATION_MODEL.md § Glide paths if in doubt.
- **"Change a tax rate / return assumption"** → defaults live in `find_retirement_date()` (`main_v2.py:1300-1307`) and are overridable from `streamlit_app.py`. Log the change in DECISIONS.md if it shifts methodology.
- **"Add a feature in the UI"** → `streamlit_app.py` is the only entry point. The simulator is deliberately UI-agnostic; keep new logic in `main_v2.py` and call it from the UI.
- **"Why is this dtype error happening?"** → almost always merge_asof on `Date`. Every Date column must be `datetime64[ns]`. There's a helper `_ensure_date_ns()` and `_ts()` in `main_v2.py:13-21` — use them on any new DataFrame that has a Date column.

## Project conventions

- **Single source of truth for the model:** `main_v2.py`. The Streamlit app is a thin wrapper that collects inputs, calls into `main_v2`, and renders outputs. Don't fork modelling logic into the UI layer.
- **Glide paths are data, not code.** New paths are authored in `Glide Paths.xlsx` in the existing tranche-chain format. If you ever need to switch to a target-allocation format, that's a model rewrite — flag it explicitly, don't try to translate silently.
- **Date discipline:** any DataFrame with a `Date` column must be `datetime64[ns]`. Pandas merge_asof is unforgiving across versions if the two sides differ in resolution.
- **No `__pycache__` / `.claude/` in git** (already in `.gitignore`). `.context/` and `Glide Paths.xlsx` ARE tracked.
- **Audit trail:** changes to glide paths are logged in `GLIDE_PATHS_CHANGELOG.md`. Material modelling changes (tax rates, return assumptions, pool windowing, retirement-search behaviour) are logged in `.context/DECISIONS.md`.
- **Sample run:** `python main_v2.py` runs `main()` with a sample config and prints the discovered retirement date — useful smoke test after refactors.
