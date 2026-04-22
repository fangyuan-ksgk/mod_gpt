# Token Vignettes

**Model:** `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` (K=1)
**Eval set:** canonical N=100 from HuggingFace

## Token t4

**Occurrences:** 1275
**Primary role:** US (41%), no_carry (63%)

**Subtask distribution:**
- US: 41% (530/1275)
- UC: 32% (409/1275)
- SA: 11% (143/1275)

**Position distribution:**
- d3: 98%

**Example problems:**

| Problem | Split | Labels | t4 position |
|---------|-------|--------|--------------|
| `536153+051522=0587675` | add_S0 | ['SA', 'SA', 'SA', 'SA', 'SA', 'SA', 'SA'] | d3 |
| `063242+415415=0478657` | add_S0 | ['SA', 'SA', 'SA', 'SA', 'SA', 'SA', 'SA'] | d3 |
| `201533+688444=0889977` | add_S0 | ['SA', 'SA', 'SA', 'SS', 'SS', 'SA', 'SA'] | d3 |
| `561100+225579=0786679` | add_S0 | ['SA', 'SA', 'SA', 'SA', 'SA', 'SA', 'SA'] | d3 |
| `006811+492164=0498975` | add_S0 | ['SA', 'SA', 'SS', 'SA', 'SS', 'SA', 'SA'] | d3 |

## Token t16

**Occurrences:** 540
**Primary role:** MD (29%), no_carry (59%)

**Subtask distribution:**
- MD: 29% (161/540)
- UD: 19% (107/540)
- UC: 18% (100/540)

**Position distribution:**
- d1: 42%
- d2: 21%
- d3: 30%
- d4: 5%

**Example problems:**

| Problem | Split | Labels | t16 position |
|---------|-------|--------|--------------|
| `285869+660963=0946832` | add_S1 | ['SA', 'UC', 'SC', 'UC', 'UC', 'UC', 'SC'] | d1 |
| `655737+485251=1140988` | add_S1 | ['UC', 'UC', 'UC', 'SC', 'SS', 'SA', 'SA'] | d1 |
| `199495+165991=0365486` | add_S1 | ['SA', 'UC', 'UC', 'UC', 'UC', 'SC', 'SA'] | d1 |
| `554701+897795=1452496` | add_S1 | ['UC', 'UC', 'UC', 'UC', 'SC', 'SS', 'SA'] | d1 |
| `748697+848798=1597495` | add_S1 | ['UC', 'SC', 'UC', 'UC', 'UC', 'UC', 'SC'] | d1 |

## Token t21

**Occurrences:** 122
**Primary role:** UB (60%), carry (50%)

**Subtask distribution:**
- UB: 60% (74/122)
- UD: 10% (13/122)
- US: 8% (10/122)

**Position distribution:**
- d1: 9%
- d2: 50%
- d3: 26%
- d4: 13%

**Example problems:**

| Problem | Split | Labels | t21 position |
|---------|-------|--------|--------------|
| `130609+069130=0199739` | add_S0 | ['SA', 'SA', 'SS', 'SS', 'SA', 'SA', 'SA'] | d2 |
| `751451+834792=1586243` | add_S1 | ['UC', 'SC', 'SA', 'UC', 'UC', 'SC', 'SA'] | d2 |
| `274962+783844=1058806` | add_S2 | ['UC', 'US', 'SC', 'UC', 'UC', 'SC', 'SA'] | d2 |
| `242169+435338=0677507` | add_S2 | ['SA', 'SA', 'SA', 'SA', 'UC', 'US', 'SC'] | d2 |
| `694541+313480=1008021` | add_S2 | ['UC', 'US', 'SC', 'UC', 'US', 'SC', 'SA'] | d2 |

