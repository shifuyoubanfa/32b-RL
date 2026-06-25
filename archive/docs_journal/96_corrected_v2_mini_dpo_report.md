# 96_corrected_v2_mini_dpo 报告（验收集 224 条，Kimi 裁判）

- 推理 humanness 均值：**0.688**
- 推理 grounded(忠于参考) 均值：**0.853**
- 准确率(平均分,漂移)：**0.806**
- correct / partial / incorrect：137 / 66 / 21
- correct%：**61.2%**  ｜ correct+partial%：**90.6%**

## humanness 分布
| 区间 | 条数 | 占比 |
|---|---|---|
| 0.0-0.2 | 0 | 0.0% |
| 0.2-0.4 | 17 | 7.6% |
| 0.4-0.6 | 20 | 8.9% |
| 0.6-0.8 | 155 | 69.2% |
| 0.8-1.0 | 32 | 14.3% |

## RAG 痕迹计数（学生）
| 类型 | 次数 |
|---|---|
| explicit_ref | 27 |
| verbatim_copy | 63 |
| ref_enumeration | 15 |
| policy_source | 52 |

## accuracy × humanness 交叉（健康收敛应 correct ≥ incorrect）
- correct=0.745 ｜ partial=0.628 ｜ incorrect=0.507
