# SIMULATION_MODEL — How the simulator actually works

Read this end-to-end at least once. After that, the routing in `CLAUDE.md` will tell you which section is relevant for a given question.

## The big picture

`find_retirement_date()` does a **binary search over months** between the current date and the death date. For each candidate retirement month it calls `run_simulation()`. The earliest month for which `run_simulation()` returns `success=True` is the answer.

`run_simulation()` for a candidate retirement date executes this pipeline:

1. **Build per-goal cashflow schedules** (`calculate_goal_cashflows()`) — one DataFrame per goal, derived from the goal's glide-path sheet.
2. **Build SIP series** (`calculate_sip_cashflows()`) — monthly SIP amounts up to retirement (then zero), with optional step-up and SIP adjustments.
3. **Build expense series** (`calculate_expenses_cashflows()`) — monthly expenses out to the death date, with inflation, post-retirement adjustment, and per-stream adjustments.
4. **Build passive-income series** (`calculate_passive_income_cashflows()`) — monthly passive income from retirement onward, with separate pre/post-retirement growth rates.
5. **Run the post-retirement pool simulator** (`simulate_post_retirement()`) — annually refills Debt and Hybrid pools from Core Corpus to cover the next 24 / 60 months of net expenses. Monthly expenses are withdrawn from Debt. If Debt depletes, the whole simulation fails at that date.
6. **Build SIP transaction history into Core Corpus** (`create_sip_trans()`).
7. **Add all goal withdrawals + post-retirement pool replenishments + negative-SIP withdrawals to Core Corpus** (`add_withdrawls_to_trans()`), using FIFO tax-lot accounting. If Core Corpus depletes, the simulation fails.
8. **Generate the month-by-month comprehensive dataframe** (`generate_comprehensive_view()`) for the UI.

Returns: `(success, final_trans_df, failure_details, expense_movements_df, goal_dfs, comprehensive_df)`.

## Glide paths (critical — read carefully)

The format in `Glide Paths.xlsx` is **not** a target-allocation table. It is a **tranche-and-chain cashflow script**.

Each sheet (`Non-Negotiable` / `Semi-Negotiable` / `Negotiable`) has one row per cashflow event. Columns:

| Column | Meaning |
|---|---|
| `id` | Row identifier, unique within the sheet. |
| `place` | Where the money sits: `hybrid`, `debt`, or `goal` (terminal). |
| `years from inflow till end` | Years before goal-end that this row's money *arrives* in its place. |
| `years from outflow till end` | Years before goal-end that this row's money *leaves* its place (NaN for `goal` rows). |
| `inflow_from` | Either `core corpus` (source is the Core Corpus) or another row's `id` (chain link). |
| `outflow_to` | The `id` this row's money flows into (NaN for `goal` rows). |
| `% of goal value` | Fraction of the total goal target that this chain delivers. The `goal` rows' percentages must sum to 100. |

**Reading a chain**: each goal row (`place='goal'`) is the endpoint of a chain. Walk backwards via `inflow_from`. Example (one tranche from current Non-Negotiable):
- `id=3, place=goal, inflow_from=2, 25%` ← receives at year 0 (goal end)
- `id=2, place=debt, inflow_from=1, inflow=2y, outflow=0y, 25%` ← held in debt from year -2 to year 0
- `id=1, place=hybrid, inflow_from='core corpus', inflow=5y, outflow=2y, 25%` ← held in hybrid from year -5 to year -2, sourced from Core Corpus

`calculate_goal_cashflows()` walks each chain backwards and back-solves the principal that Core Corpus has to provide so the tranche reaches the goal target net of holding-period taxes (STCG ≤ 1y, LTCG > 1y). The per-link math is in `calculate_required_inflow()` (`main_v2.py:306-311`).

**Why this matters for editing**: changing a glide path means rewriting the chain rows, not just changing percentages. If you ever want a target-allocation format (e.g. "at year -5 hold 40% debt, 20% hybrid, 40% equity"), the simulator can't consume it directly — see DECISIONS.md and `GLIDE_PATHS_CHANGELOG.md` for the history.

## Post-retirement pool mechanics

Implemented in `simulate_post_retirement()` (`main_v2.py:1112-1290`).

Two `InvestmentPool` instances are maintained: `Debt` (next 24 months of expenses) and `Hybrid` (months 25–60).

At the start of each post-retirement year (`sim_date`):

1. Compute `target_debt_val` = PV at `sim_date` of net expenses (expense minus passive income) for the next 24 months at the Debt return + STCG/LTCG. Same for `target_hybrid_val` over months 25–60.
2. Compare to current market value of each pool **plus latent unrealised tax** (the pool needs to be "big enough" so that after taxes it covers the target).
3. If Hybrid has surplus and Debt has shortfall: transfer Hybrid → Debt first (this avoids unnecessary Core withdrawals).
4. Refill any remaining Debt shortfall from Core Corpus (`core_replenishments`).
5. Refill Hybrid shortfall from Core Corpus.
6. Loop the next 12 months: withdraw `(monthly_expense - passive_income)` from Debt. If Debt redemption returns `fully_funded=False`, the simulation fails at that month.

Any month where passive income exceeds expenses, the surplus is logged as `post_ret_inflows` and added back to Core Corpus by the caller.

## Tax-lot accounting

`InvestmentPool` and the `add_withdrawls_to_trans()` Core-Corpus path both use **FIFO tax lots**:
- Each investment creates a `TaxLot(date, units, purchase_price_per_unit)`.
- Redemptions consume lots in FIFO order. Per-lot tax = `gain × (STCG if holding_days ≤ 365 else LTCG)`.
- Two redemption modes: `redeem_net_amount(target_net)` (back-solve units to land on a target post-tax amount) and `redeem_gross_amount(target_gross)` (just sell `target_gross` worth, tax falls out as side-effect).

## Retirement-date binary search

`find_retirement_date()` searches months in `[current_date, death_date)`:

```
low = current month
high = death month
while low <= high:
    mid = (low + high) // 2
    if run_simulation(retirement_date=mid).success:
        record mid; try earlier (high = mid - 1)
    else:
        need more time (low = mid + 1)
```

Result is the earliest `(month, year)` that succeeds, or `None` if no month in range works.

## Date discipline

There is a single nanosecond dtype constant `_NS_DTYPE = "datetime64[ns]"` at `main_v2.py:11` and two helpers:
- `_ensure_date_ns(df)` — cast `df['Date']` to `datetime64[ns]` in place.
- `_ts(val)` — return a `pd.Timestamp` at `[ns]` resolution.

Every DataFrame with a `Date` column must use this resolution before any `merge_asof`. Newer pandas (>=3.0) is stricter about cross-resolution merges; the pin in `requirements.txt` exists because Streamlit Cloud previously resolved an older version that broke this.

## Things that look weird but are intentional

- **`core_corpus`, `equity`, and `hybrid` all default to 12% return.** They're conceptually different but the model uses the same number unless overridden — see DECISIONS.md if you're tempted to change this.
- **`generate_pseudo_nav()` produces a smooth compounding curve, not real NAVs.** This is deliberate: deterministic single-path simulation.
- **Negative net SIP (effects > SIP)** is converted into a Core Corpus withdrawal — see `streamlit_app.py` around the SIP-warning block and `main_v2.py:886-891`.
- **Goal `% of goal value` sums to 100 across `goal` rows, not across all rows.** Inflow/debt/hybrid rows carry the same percentage as their downstream goal row — that's how the back-solve walks the chain.
