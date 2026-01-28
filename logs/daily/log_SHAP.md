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
- 两个模型在全局上最依赖的是ewm_24（包含历史信息，趋势），其次是sku_id, store_id，total price（不同sku的固有需求差异，不同门店的规模，定位等差距；更倾向于是历史模式的影响），再次是is_display, is_feature, is_discount等表示促销，活动的flag
- 基本都符合业务直觉，例如ewm为单调正向，total price为单调负向等；其中ewma基本是稳定驱动，total price正常值是稳定驱动，在极大值的地方会显著压抑销量;base price，total price change ratio，lag4并不单调
- 每个特征是否单调有阈值：
    - ewma近似线性上升，离群点很少，没有阈值或者反转，是模型最确定的信息来源
    - is_discount：经典二元flag的阶跃式，不过为1时候，影响也较弱从0.05到0.15+
    - is_display：同上，不过更强从0.2略低一些到0.4+
    - is_feature：同上，但稀疏而且很强，说明只要feature了就会有明显提升
    - lag 4：非线性，整体算是单调上升但是噪声极大，信息会被ewm吸收一部分，存在一定的冗余
    - sku_id, store_id：显然不同id会有不一样的分布，模型学习到了层级差异和每个sku，store的分布
    - total price：符合逻辑的整体单调下降，但是在80-140，150-250，300以上三个价格层级有不同表现，明显对于价格层级很敏感，高价位惩罚很严重，同时这个分布也符合最初根据分布得到的价格层级
    - total_price_change_ratio，负变化是正SHAP，符合业务逻辑，整体也是线性，但是噪声和离群点较多
- 交互作用：
    - total price & is discount：在低价区，有折扣的SHAP更容易为负，不过高价区则不然，不过样本较少不足以参考，说明在低价区，折扣可以抵消高定价带来的效应
    - total & display：陈列并不能抵消价格劣势，甚至在高价区被陈列的物品更不容易被购买
    - total & feature：虽然样本较少，但是确实体现能够会提升销量，但是缺乏高价区样本，所以不知道被推荐的商品是否对价格更不敏感
    - ewma & total price：分层都不明显，不存在明显的交互作用
    - ewma & is discount：分层也不明显
    - ewma & is display：同上
    - total price & total change ratio：高价区涨价的销量惩罚更严重

## Issues Encountered
- xxx

## How Issues Were Resolved
- xxx

## Problems
- xxx

## Next Steps
- xxx
- xxx