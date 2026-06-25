# corrected-v2 mini DPO summary

| model | humanness | grounded | acc | correct% | correct+partial% | Δh vs RFT | Δacc vs RFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| RFT merged base | 0.697 | 0.858 | 0.818 | 65.6% | 91.1% | +0.000 | +0.000 |
| corrected-v2 mini DPO | 0.688 | 0.853 | 0.806 | 61.2% | 90.6% | -0.009 | -0.012 |

## Data Gates
- pair build: GO train=171 heldout=25 headroom=NA hard_negative=67.3%
- rejudge: NO-GO direction=59.2% sampled=196
- stable filter: GO train=53 heldout=10 stable=63/196 mean_h=+0.301

## Readout
- paired target direction proxy: Δh=-0.009, Δacc=-0.012.
- This is a mini validation run; do not promote to full unless humanness moves up while guardrails stay healthy.
