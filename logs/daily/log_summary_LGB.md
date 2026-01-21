# Daily Log — 2025-12-15

## Objective
- Alter code to adapt LGB portals

## Work Done

## Key Findings
- best on xgb
    - overall MAPE [test, train, valid]：9.9925, 10.4007, 10.6875
    - summary
    - 特征重要性:
0                   ewma_24_records  585446.942274
1                   is_featured_sku   82027.957052
2                            sku_id   54766.146930
3                    is_display_sku   49054.190077
4                          store_id   43071.378956
5          total_price_change_ratio   29201.581074
6                       total_price   26399.399599
7                   is_discount_sku   20028.468210
8                       lag_4_weeks   12505.228476
9           gap_since_lag_1_records   10805.831153
10                       base_price    8623.862540
11                 cos_week_of_year    5433.034877
12                 sin_week_of_year    4549.668711
13  target_changing_rate_per_week_1    3918.992774
14                        cos_month    3787.312249
15                        sin_month    2505.454004
16                          quarter    2143.576289
17                   is_quarter_end       0.000000
18                 is_quarter_start       0.000000
19                     is_month_end       0.000000
20                   is_month_start       0.000000

- best on xgb with sin cos week from start
    - overall MAPE [test, train, valid]：9.7410, 10.0478, 10.7429
    - 过拟合现象增加了，测试集的MAPE可能来源于random
    - 特征重要性:

- best on xgb with rolling mean std
    - overall MAPE [test, train, valid]：10.11，10.2667，10.7152
    - 显然更糟
    - 特征重要性:

- best on xgb with target changing rate
    - overall MAPE [test, train, valid]：10.0359，10.3224，10.7503
    - 其他相差不大，测试集准确率略低
    - 特征重要性:

- best on xgb with lag is_display， is_feature
    - overall MAPE [test, train, valid]：10.0415，10.3907，10.7648
    - 其他相差不大，测试集准确率略低
    - 特征重要性:

- best on xgb with base price change ratio
    - overall MAPE [test, train, valid]：9.9978，10.3325，10.7795
    - 其他相差不大，过拟合现象更多一些
    - 特征重要性:

- best on xgb with week total units sold
    - overall MAPE [test, train, valid]：10.2624，10.0280，10.9082
    - 显然更糟
    - 其他聚合特征也相同
    - 特征重要性:

- best on xgb with gap since lag 2 record
    - overall MAPE [test, train, valid]：10.3997, 10.3170, 10.6452
    - 更糟糕了
    - 其他调整次类结果类似，与XGB结果也类似
    - 特征重要性:

- best on xgb with lag feature 2,4
    - overall MAPE [test, train, valid]：
    - 更糟糕了
    - 其他调整次类结果类似，与XGB结果也类似
    - 特征重要性:

- best on xgb without year month quarter
    - overall MAPE [test, train, valid]：10.0289, 10.3860, 10.6486
    - 更糟糕了
    - 特征重要性:

- best on xgb without is quarter start end
    - overall MAPE [test, train, valid]：same
    - 一样的
    - 特征重要性:

- best on xgb without is month start end
    - overall MAPE [test, train, valid]：same
    - 一样的
    - 特征重要性:
0                   ewma_24_records  585446.942274
1                   is_featured_sku   82027.957052
2                            sku_id   54766.146930
3                    is_display_sku   49054.190077
4                          store_id   43071.378956
5          total_price_change_ratio   29201.581074
6                       total_price   26399.399599
7                   is_discount_sku   20028.468210
8                       lag_4_weeks   12505.228476
9           gap_since_lag_1_records   10805.831153
10                       base_price    8623.862540
11                 cos_week_of_year    5433.034877
12                 sin_week_of_year    4549.668711
13  target_changing_rate_per_week_1    3918.992774
14                        cos_month    3787.312249
15                        sin_month    2505.454004
16                          quarter    2143.576289

- best on xgb without sin cos month
    - overall MAPE [test, train, valid]：10.0795, 10.4164, 10.6534
    - worse
    - 特征重要性:

- 其他调整也类似，特征工程会与XGB类似

- 特征工程的best：
    - overall MAPE [test, train, valid]：10.1272, 9.2752, 10.6697(开始超参数)

- num_leaves = [31,63,127,255]
    - overall MAPE [test, train, valid]：10.0202, 9.8804, 10.6234
    - overall MAPE [test, train, valid]：10.1272, 9.2752, 10.6697
    - overall MAPE [test, train, valid]：10.0511, 9.1609, 10.7900
    - overall MAPE [test, train, valid]：10.0511, 9.1609, 10.7990
    选择 31

- min_child_samples = [20, 50, 100, 200]
    - overall MAPE [test, train, valid]：9.9677, 9.5359, 10.6879
    - overall MAPE [test, train, valid]：10.0202, 9.8804, 10.6234
    - overall MAPE [test, train, valid]：10.0641, 10.0077, 10.6502
    - overall MAPE [test, train, valid]：10.1714, 10.0796, 10.7464
    选50

- max_depth = [-1, 5, 6, 7, 8]
    - overall MAPE [test, train, valid]：10，0310，9.3328，10.5686
    - overall MAPE [test, train, valid]：10.1019，9.8790，10.5644
    - overall MAPE [test, train, valid]：10.0202, 9.8804, 10.6234
    - overall MAPE [test, train, valid]：9.9269，9.4470，10.7385
    - overall MAPE [test, train, valid]：9.9673，9.6805，10.6893
    选择6

- feature_fraction = [0.6, 0.7, 0.8, 0.9, 1.0]
    - overall MAPE [test, train, valid]：9.9251，9.4125，10.6394
    - overall MAPE [test, train, valid]：9.9838，9.9767，10.6350
    - overall MAPE [test, train, valid]：10.0202, 9.8804, 10.6234
    - overall MAPE [test, train, valid]：9.9175，9.6675，10.6657
    - overall MAPE [test, train, valid]：10.0008，9.6398，10.7067
    选择0.7

- min_split_gain=[0, 0.01, 0.05, 0.1]
    - overall MAPE [test, train, valid]：9.9838，9.9767，10.6350
    - overall MAPE [test, train, valid]：10.0212, 9.9216, 10.6840
    - overall MAPE [test, train, valid]：9.9554, 9.7796, 10.6775
    - overall MAPE [test, train, valid]：10.0571, 10.0716, 10.7222
    选0
    best iteration = 2043
    -> 2927

- learning_rate = [0.03, 0.04, 0.05, 0.07]
    - 0.03过拟合太大
    - 4058 -> 4553
    选择0.04 - best ietration较稳定

- 最终结果：
    - MAPE：overall MAPE [test, train, valid]：10.0248，9.7111，10.7274
    - best ieration = 3365
    - 所有超参数见entrance
    - 特征重要性：
0                   ewma_24_records  563247.394391
1                   is_featured_sku   77347.373259
2                            sku_id   62650.912503
3                          store_id   53590.779089
4                    is_display_sku   47566.874116
5          total_price_change_ratio   36924.140046
6                       total_price   32946.711686
7                       lag_4_weeks   32044.838168
8                   is_discount_sku   21080.613703
9                        base_price   13408.204087
10          gap_since_lag_1_records   12204.850052
11  target_changing_rate_per_week_1    8133.202197
12                 cos_week_of_year    7254.410172
13                 sin_week_of_year    6068.963828
14                        cos_month    4698.294396
15                        sin_month    3864.051571
16                          quarter    2470.150022

经由ensemble，得到best w = 0.54
最佳overall MAPE test = 9.9547

- LGBM
--- MAPE by (week, sku) top 10 Train: ---
           week  sku_id  mape_week_sku
26   2001-01-13  673209     177.139737
131  2002-04-13  327492     128.722425
2350 2020-06-11  673209     117.413644
22   2001-01-13  545621     106.254985
1478 2013-03-12  545621     102.532992
3134 2027-02-12  673209     101.364609
1006 2009-05-11  673209      91.826549
3050 2026-06-12  673209      91.093676
2322 2020-03-12  673209      78.810512
3381 2029-01-13  398721      67.962810

- XGB
--- MAPE by (week, sku) top 10 Train: ---
           week  sku_id  mape_week_sku
26   2001-01-13  673209     166.681931
131  2002-04-13  327492     127.704143
2350 2020-06-11  673209     123.140592
3134 2027-02-12  673209     104.001636
1478 2013-03-12  545621      98.475800
1006 2009-05-11  673209      92.661755
3050 2026-06-12  673209      92.533141
2322 2020-03-12  673209      77.624975
22   2001-01-13  545621      74.874089
3381 2029-01-13  398721      70.698351

可以看到最大MAPE的地方就在这些点，说明综合可能并不能带来更好的结果，这个异常可能是数据量太少带来的，由MAPE算法导致

那么现在如果用验证集加权平均找最佳MAPE是不是相当于以结果为导向的？

## Issues Encountered


## How Issues Were Resolved


## Problems


## Next Steps
