# Glide Paths Changelog

Audit trail of changes to `Glide Paths.xlsx`. Each entry records what changed, when, and why. The git history of `Glide Paths.xlsx` is the source of truth for the binary file; this file gives a human-readable summary.

---

## 2026-05-21 — Tranche restructure across all three goal types

**Source:** `Glide Paths v2.xlsx` ("New Glide Paths" sheet), re-authored into the existing cashflow-chain format by the user.

**Validation:** Sample run on a representative 3-goal config produced the same retirement-date result (Jul 2027) as the previous version, confirming the chain shapes load and compute correctly. Each sheet's goal tranches sum to 100%.

### Non-Negotiable
- Tranche count: **5 → 4** (was 5×20%, now 4×25%).
- One tranche keeps the 3-link `hybrid → debt → goal` chain (hybrid 5y, debt 2y); the remaining three tranches are direct `debt → goal` at 4y, 3y, and 2y before goal.
- Net effect: shorter equity-like exposure, more debt across the tail.

### Semi-Negotiable
- Tranche count: **5 → 4** (was 5×20%, now 4×25%).
- One `hybrid → goal` tranche at 4y; three `debt → goal` tranches at 3y / 2y / 1y.
- The longer 5y hybrid+debt chain from the old version is dropped.

### Negotiable
- Tranche count: **5 → 5** (unchanged), but **percentages are no longer equal**.
- New mix: 30% hybrid (3y), 10% hybrid (2y), 10% hybrid (1y), 30% debt (2y), 20% debt (1y).
- Previously all five were 20%. Weighting now skews more toward the 3y hybrid and 2y debt buckets.

### How to recover the previous version
The previous `Glide Paths.xlsx` is preserved in git history. To inspect or restore:
```
git log -- "Glide Paths.xlsx"
git show <commit>:"Glide Paths.xlsx" > "Glide Paths_prev.xlsx"
```
