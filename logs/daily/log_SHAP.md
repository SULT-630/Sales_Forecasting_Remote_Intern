# Daily Log — 2025-xx-xx

## Objective
- Using SHAP method to evaluate the model
- learn architecture of XGB and LGBM

## Work Done
- Import SHAP explainer
- 解释哪个特征贡献的最多 （模型全局在依赖什么，比较特征重要性为什么不行，是否符合业务直觉，稳定驱动还是极端值驱动
- 解释某个特征是增加还是降低特征，正向/负向，单调与否
- 每个特征的影响的线性非线性与阈值效应，特征影响是否是线性的，是否存在拐点
- 是否存在交互逻辑
- 对于某些误差很大的样本，解释模型是如何预测出这个结果的，系统性问题还是随机噪声
- 对于高销量和低销量不同区域的样本的预测逻辑是否是一样的
- 是否存在系统性偏差，例如在某个日期的预测全部偏离一定值
- 可信度总结？

## Key Findings
- 不能使用特征重要性的原因：只按照某个特定模式观察（例如gain），无法观察到正负方向，更容易偏向连续变量，而且在交互作用存在时不太稳定

## Issues Encountered
- xxx

## How Issues Were Resolved
- xxx

## Problems
- xxx

## Next Steps
- xxx
- xxx