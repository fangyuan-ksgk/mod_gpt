# Token Swap: Surgical Transplant

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d`
**Swap:** t16 → t21
**Splits:** C3-C6 (hard carry chains)

## Results

- **Fixed:** 6 examples (wrong → correct after swap)
- **Broke:** 0 examples (correct → wrong after swap)
- **Ratio:** 6:0

| Split | N | Normal | Swap | Fixed | Broke |
|-------|---|--------|------|-------|-------|
| add_C3 | 100 | 87% | 87% | 0 | 0 |
| add_C4 | 100 | 80% | 82% | 2 | 0 |
| add_C5 | 100 | 82% | 83% | 1 | 0 |
| add_C6 | 100 | 74% | 77% | 3 | 0 |

## Example Fixes

- **add_C4:** `045183+528818` = `0574001`
  - Normal: `0573001` (wrong)
  - Swapped: `0574001` (correct, 4 tokens swapped)
- **add_C4:** `288271+071770` = `0360041`
  - Normal: `0350041` (wrong)
  - Swapped: `0360041` (correct, 5 tokens swapped)
- **add_C5:** `188748+161276` = `0350024`
  - Normal: `0340024` (wrong)
  - Swapped: `0350024` (correct, 5 tokens swapped)
- **add_C6:** `459694+790306` = `1250000`
  - Normal: `1240000` (wrong)
  - Swapped: `1250000` (correct, 5 tokens swapped)
- **add_C6:** `987583+052427` = `1040010`
  - Normal: `1030010` (wrong)
  - Swapped: `1040010` (correct, 12 tokens swapped)
