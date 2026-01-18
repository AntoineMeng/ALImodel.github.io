---
title: "他汀类药物急性肝损伤预测模型"
collection: portfolio
type: "Project"
permalink: /portfolio/statins-ali-prediction
date: 2026-01-18
excerpt: "基于迁移学习的深度神经网络预测模型,通过整合临床常规指标实现他汀类药物急性肝损伤的精准预测"
header:
  teaser: /images/portfolio/inclusion_flowchart.png
tags:
- 深度学习
- 迁移学习
- 医疗AI
- 预测模型
tech_stack:
- name: Python
- name: TensorFlow
- name: Scikit-learn
- name: SHAP
---

## 项目背景

随着生活方式的改变和人口老龄化的加剧，高脂血症已成为全球范围内常见的代谢性疾病之一。他汀类药物通过抑制肝脏中3-羟基-3-甲基戊二酰辅酶A(HMG-CoA)还原酶来减少体内胆固醇的合成，是临床上高脂血症和心血管疾病治疗和预防的基础药物。

尽管他汀类药物在大多数患者中表现出良好的安全性和耐受性，但仍有可能引起一系列不良反应，急性肝损伤(ALI)是其中较为严重的一种。ALI的发生具有显著的异质性，开发能够更精准和全面预测ALI的模型显得尤为重要。

## 数据来源

本研究数据来源于宁波市鄞州区域健康信息平台，平台内已累计涵盖当地超122万人口的多方面的、纵向的、完整的卫生信息，如基本人口学特征、疾病诊断随访、化验检查、处方、死因登记等。

由于急性肝损伤发生率较低，本研究采用同一系统更为常见的不良反应胃肠道症状作为源结局，基于预训练-微调的迁移学习策略开发预测模型。

## 数据预处理

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 删除重复值
data.drop_duplicates(inplace=True)

# 逻辑错误调整
data.loc[data['age'] > 120, 'age'] = np.nan
data.loc[data['bmi'] > 50, 'bmi'] = np.nan

# 连续变量异常值处理 (IQR方法)
def replace_outliers_with_nan(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), np.nan)
    return df

# 选取需要处理的连续变量列
continuous_cols = ['ldl_c', 'ast', 'alt', 'ggt']
for col in continuous_cols:
    if col in data.columns:
        replace_outliers_with_nan(data, col)

# 数据标准化
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
```

预处理流程包括：删除完全重复记录、识别并处理逻辑错误（不合理的年龄和BMI值）、利用箱线图规则自动识别异常值并转换为缺失值、采用多重插补法填补缺失数据，并对连续变量进行标准化。

## 统计分析

### 研究人群筛选流程

![纳排流程图](/images/portfolio/inclusion_flowchart.png)

图1展示了研究对象的筛选流程，从宁波鄞州区域健康信息平台122万人口数据中，最终纳入开发队列90,542例和时序验证队列35,898例。

### 描述性统计

开发队列和验证队列中，发生ALI与未发生ALI患者的基线特征比较显示：发生ALI的患者Charlson共病指数≥3的比例显著更高（开发队列66.9% vs 40.1%, P<0.001），存在轻度肝脏疾病和中重度肝脏疾病的比例也显著更高。

## 预测模型建立

本研究采用迁移学习策略，首先在胃肠道症状数据上预训练源模型，然后将学习到的特征迁移到目标ALI预测任务中。

### 源模型预训练

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

def create_model_transfer0_origin(hidden_layers=2, neurons=16, 
                                   l1_reg=0.01, l2_reg=0.01, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X1_train.shape[1],)))
    
    for _ in range(hidden_layers):
        model.add(Dense(neurons, activation='relu',
                       kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                 metrics=[tf.keras.metrics.AUC(name="auc")])
    return model
```

采用贝叶斯优化进行超参数调优，搜索空间包括：隐藏层数量(4-9层)、每层神经元数量(8-257)、L1/L2正则化系数(1e-4到1e-1)、学习率(1e-4到1e-1)。

### 目标模型微调

```python
# 冻结源模型前i层
for i in range(1, hidden_layers_transfer0_origin+1):
    transfer_model = Sequential()
    
    # 迁移预训练层
    for layer in best_model_transfer0_origin.layers[:i]:
        layer.trainable = False
        transfer_model.add(layer)
    
    # 添加新的可训练层
    for j in range(hidden_layers_transfer0_origin - i):
        transfer_model.add(Dense(neurons, activation='relu',
                                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    
    transfer_model.add(Dense(1, activation='sigmoid'))
    transfer_model.compile(loss='binary_crossentropy', 
                          optimizer=Adam(learning_rate=learning_rate),
                          metrics=[tf.keras.metrics.AUC(name="auc")])
```

## 模型评估结果

目标模型在测试集上的表现如下表所示：

| 评价维度 | 评价指标 | 数值 (95% CI) |
|---------|---------|--------------|
| 整体表现 | 准确率 | 0.901 (0.892, 0.905) |
| | 平衡准确率 | 0.764 (0.687, 0.841) |
| 区分度 | AUC | 0.835 (0.757, 0.914) |
| | AUPRC | 0.219 (0.079, 0.352) |
| | Fβ-score (β=3) | 0.289 (0.203, 0.374) |
| | G-mean | 0.751 (0.657, 0.845) |
| | AGm | 0.827 (0.780, 0.874) |

### 校准曲线

![校准曲线](/images/portfolio/calibration_curve.png)

校准曲线显示模型预测概率与实际发生概率的一致性较好，表明模型具有良好的校准度。

## 模型可解释性

使用SHAP(SHapley Additive exPlanations)对模型进行可解释性分析。

### 特征重要性排序

![特征重要性](/images/portfolio/feature_importance.png)

图3展示了各特征对模型预测的平均影响程度排序。排名靠前的特征对预测结果贡献最大。

### SHAP摘要图

![SHAP摘要图](/images/portfolio/shap_summary.png)

SHAP摘要图展示了每个特征对单个样本预测值的影响方向和大小。红色表示特征值较高，蓝色表示特征值较低，向右表示增加ALI风险，向左表示降低风险。

## 项目总结

本研究成功构建了基于迁移学习的深度神经网络预测模型，实现了对他汀类药物急性肝损伤的精准预测。模型AUC达到0.835，展现出良好的区分能力和校准度。

创新点包括：
1. 采用迁移学习策略，解决了ALI发生率低导致的数据不平衡问题
2. 整合了多个维度的临床常规指标，模型具有实际应用价值
3. 通过SHAP分析提供了模型的可解释性，增强了临床信任度

该模型可帮助临床医师识别发生ALI风险较高的他汀用药者，从而优化治疗方案，减少ALI的发生。
