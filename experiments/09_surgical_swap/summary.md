# Surgical Token Swap — Hard Cases

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` (K=1, abs_vocab=30)
**Splits:** add_C4, add_C5, add_C6, sub_B3, sub_B4, sub_B5
**Wrong examples:** 126 | **Correct:** 474

## Protocol

For each wrong example on hard splits, try replacing each abs token with
every other token in the vocabulary. Record which swaps fix the answer.
Then check how many correct examples each swap breaks.

## Per-Position Fixability

How often can a wrong answer be fixed by changing the abs token at each position?

| Position | Wrong | Fixable | Fix Rate |
|----------|-------|---------|----------|
| d0 | 126 | 34 | 27.0% |
| d1 | 126 | 38 | 30.2% |
| d2 | 126 | 39 | 31.0% |
| d3 | 126 | 10 | 7.9% |
| d4 | 126 | 2 | 1.6% |

## Top Swaps (by fix count)

| Pos | Old→New | Subtask | Fixed | Broke | Ratio |
|-----|---------|---------|-------|-------|-------|
| d1 | t16→t25 | UC | 10 | 5 | 10:5 |
| d1 | t1→t26 | SA | 8 | 35 | 8:35 |
| d1 | t1→t23 | UC | 7 | 52 | 7:52 |
| d1 | t16→t21 | UC | 7 | 1 | 7:1 |
| d0 | t3→t25 | SA | 6 | 8 | 6:8 |
| d0 | t3→t23 | SA | 6 | 36 | 6:36 |
| d1 | t1→t28 | SA | 6 | 11 | 6:11 |
| d1 | t1→t24 | UC | 6 | 32 | 6:32 |
| d1 | t16→t20 | UC | 6 | 1 | 6:1 |
| d3 | t4→t25 | UC | 6 | 7 | 6:7 |
| d1 | t16→t24 | UC | 6 | 11 | 6:11 |
| d1 | t1→t29 | US | 6 | 171 | 6:171 |
| d1 | t1→t25 | SA | 5 | 6 | 5:6 |
| d0 | t10→t25 | UC | 5 | 1 | 5:1 |
| d1 | t1→t27 | SA | 5 | 15 | 5:15 |
| d1 | t16→t6 | UC | 5 | 2 | 5:2 |
| d1 | t16→t10 | UC | 5 | 2 | 5:2 |
| d0 | t8→t25 | UC | 5 | 6 | 5:6 |
| d2 | t17→t20 | UC | 5 | 1 | 5:1 |
| d2 | t17→t24 | UC | 5 | 5 | 5:5 |

## Example Fixes (best swap: d1 t16→t25)

- **add_C4:** `186151+063874` = `15172015151720`
  - Wrong: `15171915151720` → Fixed: `15172015151720`
- **add_C4:** `288271+071770` = `15182115151916`
  - Wrong: `15182015151916` → Fixed: `15182115151916`
- **add_C5:** `188748+161276` = `15182015151719`
  - Wrong: `15181915151719` → Fixed: `15182015151719`
- **add_C6:** `459694+790306` = `16172015151515`
  - Wrong: `16171915151515` → Fixed: `16172015151515`
- **add_C6:** `987583+052427` = `16151915151615`
  - Wrong: `16151815151615` → Fixed: `16151915151615`

## Verdict

**6 swaps with fix:break > 2:1 and >= 3 fixes.**

Token identity causally determines accuracy on hard cascade examples.
Specific tokens encode specific cascade computation — replacing them
with the right alternative surgically fixes wrong answers.
