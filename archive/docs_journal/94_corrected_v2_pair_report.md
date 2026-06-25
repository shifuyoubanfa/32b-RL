# corrected-v2 Phase 1 / step94 strict DPO pair report

- status: **NO-GO**
- reason: data gates failed; inspect report before training
- scored queries: 120
- scored candidates with Kimi judge: 960
- all pairs: 35
- train pairs: 35
- heldout pairs: 0
- best-of-K headroom vs RFT base: NA
- hard negative rate: 65.7%
- mean length diff ratio: 0.137

## Pair Means

| metric | chosen | rejected | margin |
|---|---:|---:|---:|
| humanness | 0.786 | 0.393 | +0.393 |
| grounded | 0.959 | 0.849 | +0.110 |
| accuracy_score | 0.941 | 0.764 | +0.177 |

## Gates

| gate | value | pass |
|---|---:|---:|
| train pairs >= 80 | 35 | False |
| heldout pairs >= 20 | 0 | False |
| headroom >= 0.05 | NA | True |
| hard negatives >= 25% | 65.7% | True |

## Notes

- Old 60_dpo_pairs.jsonl is not reused for training here.
- This step uses Kimi-scored candidates from the existing final-RFT rollout pool.
- If status is NO-GO, stop before DPO and inspect pair quality rather than forcing training.
