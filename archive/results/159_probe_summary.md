# derag_v5 headroom probe summary

- run_id: `main`
- **verdict: GO_RL**
- reason: RFT 自采样已稳定出现干净且不漂移的样本，DPO/GRPO 有正样本可学。

## 关键数字

| 指标 | all | eval(没见过) | train(见过,偏乐观) |
|---|---:|---:|---:|
| 病题数 | 489 | 95 | 394 |
| **X=RFT 自救率** | **0.851** | 0.8526 | 0.8503 |
| **Y=Kimi 改写成功率** | **0.532** | 0.4842 | 0.5431 |
| 平均每16遍合格数 | 8.986 | | |
| 平均每16遍think干净数 | 13.483 | | |
| 平均每16遍答案在范围数 | 10.524 | | |

## 判据
- X≥0.45 → GO_RL（整链值得跑）
- X<0.45 ∧ Y≥0.60 → GO_SFT_FIRST（先 SFT 搬中心再续 RL）
- X<0.45 ∧ Y<0.60 → NO_GO（天花板太低，交报告）

## 逐题明细
- RFT 自采样逐题：153_rft_headroom.jsonl（每题 16 遍各自 think_clean/answer_in_support/pass）
- Kimi 改写逐题：154_rewrite_headroom.jsonl（before/after 痕迹数 + 是否改干净 + 新 think）
- 病题清单+脏在哪：150_problems.jsonl
