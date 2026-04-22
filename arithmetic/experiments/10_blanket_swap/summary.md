# Blanket vs Surgical Token Swap

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` (K=1)
**Baseline accuracy:** 2002/2600 (77%)
**Tokens tested:** 22 (freq >= 100)
**Pairs tested:** 231

## The Test

For a pair (A, B) to prove tokens encode distinct information:
1. Blanket A→B must **lower** accuracy (B is wrong in A's natural contexts)
2. Blanket B→A must **lower** accuracy (A is wrong in B's natural contexts)
3. Surgical A→B on wrong examples must **fix** some (model assigned wrong token)
4. Surgical B→A on wrong examples must **fix** some (same, other direction)

If all four hold: the tokens are genuinely different, and the model sometimes
picks the wrong one. This rules out vocab redundancy AND vocab noise.

## Results: 68 ideal pairs found

| Token A | Token B | A→B Fixed | A→B Broke | B→A Fixed | B→A Broke |
|---------|---------|-----------|-----------|-----------|-----------|
| t1 (n=2600) | t12 (n=2495) | 69 | 90 | 75 | 86 |
| t2 (n=2148) | t3 (n=1821) | 25 | 53 | 19 | 32 |
| t3 (n=1821) | t6 (n=432) | 14 | 43 | 9 | 10 |
| t3 (n=1821) | t7 (n=910) | 18 | 44 | 7 | 8 |
| t1 (n=2600) | t7 (n=910) | 69 | 116 | 6 | 10 |
| t0 (n=2600) | t3 (n=1821) | 5 | 1980 | 19 | 1196 |
| t2 (n=2148) | t13 (n=559) | 16 | 65 | 5 | 13 |
| t6 (n=432) | t8 (n=2572) | 5 | 16 | 33 | 107 |
| t6 (n=432) | t13 (n=559) | 5 | 13 | 8 | 10 |
| t0 (n=2600) | t2 (n=2148) | 4 | 1987 | 15 | 1464 |
| t0 (n=2600) | t6 (n=432) | 4 | 1981 | 8 | 283 |
| t0 (n=2600) | t8 (n=2572) | 4 | 1983 | 23 | 1856 |
| t0 (n=2600) | t10 (n=1497) | 4 | 1984 | 43 | 859 |
| t0 (n=2600) | t12 (n=2495) | 4 | 1987 | 39 | 1725 |
| t0 (n=2600) | t13 (n=559) | 4 | 1985 | 15 | 357 |

**Best pair: t1 ↔ t12**

- Blanket t1→t12: fixes 69, breaks 90 (net -21)
- Blanket t12→t1: fixes 75, breaks 86 (net -11)

Both directions are net-negative globally, but both fix some wrong examples.
The tokens encode **distinct, non-redundant information** that the model
sometimes misassigns.

## Statistics

- Pairs where both directions hurt: **88**
- Pairs where both directions fix: **175**
- Ideal pairs (both hurt AND both fix): **68**
