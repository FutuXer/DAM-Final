# 项目技术计划书 — 基于多源时序特征融合的信用风险预测

**课程**: 数据挖掘与分析 (Data Mining and Analysis)
**选题类型**: B — 算法系统性对比研究
**数据集**: Home Credit Default Risk (Kaggle)
**团队规模**: 4 人
**仓库地址**: https://github.com/FutuXer/DAM-Final

---

## 1. 选题背景

### 1.1 应用场景与问题来源

信用风险评估是金融机构信贷决策的核心环节。传统风控模型以申请人的静态填表信息（年龄、收入、学历、住房状况）作为主要输入，通过逻辑回归或评分卡模型输出违约概率估计。该方法存在两个结构性局限：其一，静态截面数据无法反映申请人还款能力的时序变化趋势；其二，单一数据源的信息量有限，忽视了大量可获取的外部征信记录和内部历史交易流水。

Home Credit 是一家国际消费金融公司，服务人群以缺乏传统银行信贷记录的"薄档案"客户为主。对该人群的信用评估，依赖传统静态特征的风险区分度不足。2018 年，Home Credit 在 Kaggle 平台发布了匿名化贷款申请数据集，包含 7 张关联表，覆盖申请人的静态画像、外部征信局历史、内部 POS/信用卡/分期还款等交易流水，旨在推动面向替代性数据（alternative data）的信用风控建模研究。

### 1.2 选题依据与课程要求对应

本课程期末项目提供三个选题方向。选题 B 的要求为：针对同一任务，实现并对比至少 5 种不同类型的算法，在同一数据集和交叉验证策略下进行系统性实验分析。

选择选题 B 的依据：
1. Home Credit 数据集的多表结构天然适合作为对比实验的统一基准——所有算法在完全相同的特征矩阵上训练和评估，排除数据差异对结论的干扰。
2. 数据集中的类别不平衡（TARGET=1 约占 8%）、高维稀疏特征、时序依赖等特性，为不同算法类型提供了差异化的性能表现空间，使对比分析具有区分度。
3. 信用风控场景对模型可解释性有明确需求——线性模型提供系数解释，树模型提供特征重要性和 SHAP 值——不同算法在预测性能与可解释性之间的权衡正是选题 B 可以深入讨论的核心议题。

### 1.3 研究问题

在统一的多源特征工程和 5-Fold Stratified Cross-Validation 框架下，回答以下问题：

1. 线性模型（Logistic Regression）、Bagging 集成（Random Forest）、Boosting 集成（XGBoost、LightGBM）和神经网络（MLP）在该信用违约预测任务上的性能排序是怎样的？
2. 不同算法类型在不平衡二分类问题上的强项和弱项分别体现在哪些评估指标上？
3. 对最优模型而言，驱动违约预测的关键特征是什么？这些特征是否具有业务可解释性？

---

## 2. 数据来源

### 2.1 数据集基本信息

| 项目 | 内容 |
|------|------|
| 数据集名称 | Home Credit Default Risk |
| 来源平台 | Kaggle 竞赛 |
| 竞赛页面 | https://www.kaggle.com/competitions/home-credit-default-risk/data |
| 数据提供方 | Home Credit Group（捷信集团） |
| 公开时间 | 2018 年 |
| 数据性质 | 匿名化真实贷款申请记录（脱敏、哈希处理） |
| 授权范围 | 竞赛及学术研究用途 |

### 2.2 数据规模

| 序号 | 表名 | 行数 | 列数 | 文件大小 |
|------|------|------|------|----------|
| 1 | `application_train.csv` | 307,511 | 122 | 159 MB |
| 2 | `application_test.csv` | 48,744 | 121 | 26 MB |
| 3 | `bureau.csv` | 1,716,429 | 17 | 163 MB |
| 4 | `bureau_balance.csv` | 27,299,925 | 3 | 359 MB |
| 5 | `previous_application.csv` | 1,670,214 | 37 | 387 MB |
| 6 | `POS_CASH_balance.csv` | 10,001,358 | 8 | 375 MB |
| 7 | `credit_card_balance.csv` | 3,840,312 | 23 | 405 MB |
| 8 | `installments_payments.csv` | 13,605,401 | 8 | 690 MB |
| **合计** | — | **58,549,894** | — | **2.5 GB** |

### 2.3 数据获取方式

数据通过 Kaggle API 下载，需先在 Kaggle 账户设置中生成 API Token（kaggle.json），然后执行：

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
```

项目仓库的 `data/raw/` 目录已添加至 `.gitignore`，数据文件不上传至 Git 仓库。团队成员需各自下载数据至本地同名目录。

---

## 3. 数据集描述

### 3.1 数据表关联结构

7 张表的关联关系以 `SK_ID_CURR`（贷款申请 ID）为核心：

```
application_train (SK_ID_CURR)
    │
    ├── 1:N ── bureau (SK_ID_CURR) ── 1:N ── bureau_balance (SK_BUREAU_ID)
    │
    └── 1:N ── previous_application (SK_ID_CURR)
                    │
                    ├── 1:N ── POS_CASH_balance (SK_ID_PREV)
                    ├── 1:N ── credit_card_balance (SK_ID_PREV)
                    └── 1:N ── installments_payments (SK_ID_PREV)
```

### 3.2 各表内容概述

**application_train / application_test（主表）**
每条记录为一个贷款申请，SK_ID_CURR 为主键。包含 TARGET（目标变量：1 = 有还款困难，0 = 正常）、贷款金额/年金/商品价格、申请人性别/年龄/子女数/家庭成员数/收入/学历/职业/住房类型、三个外部数据源归一化评分 (EXT_SOURCE_1/2/3)、社交圈违约观测数、20 个文件提供标志、6 个不同时间窗口的征信局查询次数、数十个建筑相关归一化特征（_AVG/_MODE/_MEDI 后缀）、地区评分等。

**bureau（外部征信局记录）**
SK_BUREAU_ID 为主键，通过 SK_ID_CURR 关联主表（1:N）。每条记录为申请人在外部征信局的一笔历史信用。包含信用激活状态 (CREDIT_ACTIVE)、信用类型 (CREDIT_TYPE，如车贷/现金贷)、信用金额/债务/逾期金额/最大逾期金额、逾期天数 (CREDIT_DAY_OVERDUE)、延期次数 (CNT_CREDIT_PROLONG)、信用起止时间等。

**bureau_balance（征信局月度状态）**
无独立主键，以 SK_BUREAU_ID + MONTHS_BALANCE 为联合键。记录每笔征信局信用每月的状态 (STATUS)：0 = 无逾期、1 = 逾期 1-30 天、2 = 逾期 31-60 天、3 = 逾期 61-90 天、4 = 逾期 91-120 天、5 = 逾期 120+ 天或核销、C = 关闭、X = 未知。

**previous_application（历史贷款申请）**
SK_ID_PREV 为主键，通过 SK_ID_CURR 关联主表（1:N）。包含申请金额/批准金额/首付金额、合同状态（批准/取消/拒绝）、拒绝原因 (CODE_REJECT_REASON)、利率、产品类型、渠道类型、决策时间等。

**POS_CASH_balance / credit_card_balance（月度余额快照）**
以 SK_ID_PREV + MONTHS_BALANCE 为粒度。记录 POS 现金贷和信用卡的月度余额、信用额度、逾期天数 (SK_DPD)、合同状态等。

**installments_payments（分期还款明细）**
以 SK_ID_PREV + NUM_INSTALMENT_NUMBER 为粒度。记录每笔历史贷款每期分期的应还金额 (AMT_INSTALMENT)、实还金额 (AMT_PAYMENT)、应还日期 (DAYS_INSTALMENT)、实际还款日期 (DAYS_ENTRY_PAYMENT)。

### 3.3 数据特殊约定

- **时间字段**：`DAYS_` 前缀列均为相对于申请日的天数偏移，负值表示申请日之前。例如 DAYS_BIRTH = -10950 表示申请时年龄为 10950 / 365 ≈ 30 岁。
- **就业异常标记**：DAYS_EMPLOYED = 365243 为无业人员的编码标记，并非真实天数。
- **归一化列**：EXT_SOURCE_* 和建筑特征系列已经过归一化，值域通常为 [0, 1]。
- **哈希脱敏**：标注 `hashed` 的列已做匿名化哈希处理，原始值不可恢复。

---

## 4. 技术方案

### 4.1 特征工程总体策略

将 7 张表整合为单张训练宽表。辅助表不直接与主表 Join，而是以 SK_ID_CURR 为分组键执行统计聚合，将每组的多条记录压缩为固定维度特征向量，再通过 Left Join 合并到主表。

聚合算子：{count, nunique, mean, std, min, max, sum, median}。对时序数据，额外增加最近 N 个月（N ∈ {6, 12, 24}）的滑动窗口聚合。

### 4.2 主表特征工程 (队员 A)

**A1. 缺失值处理**
- 数值列：中位数填充。
- 类别列：众数填充。
- 缺失率 > 60% 的列：填充后额外生成二值标记列 `{col}_MISSING`（共 41 列建筑特征系列符合此条件）。
- 涉及范围：EXT_SOURCE_2（缺失率 ~30%）、EXT_SOURCE_3（缺失率 ~20%）、OWN_CAR_AGE（缺失率 ~66%）、建筑特征 _AVG/_MODE/_MEDI 系列（缺失率 50-70%）。

**A2. 异常值处理**
- DAYS_EMPLOYED = 365243：识别为无业标记，生成 FLAG_UNEMPLOYED，原始值替换为中位数。
- 金额列 (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE)：上 99% 分位数 Winsorize 截尾。
- CNT_CHILDREN > 10 → 截断为 10；CNT_FAM_MEMBERS > 15 → 截断为 15。
- 逻辑修正：CNT_FAM_MEMBERS < CNT_CHILDREN + 1 的记录，将 CNT_FAM_MEMBERS 调整为 CNT_CHILDREN + 1。

**A3. 类别特征编码**
- 低基数（≤ 5 类别）：One-Hot Encoding。
- 高基数（> 5 类别）：5-Fold Stratified Target Encoding，用 SMOTE 外的原始标签计算每折的目标均值。
- 涉及列：NAME_CONTRACT_TYPE, CODE_GENDER, NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, WEEKDAY_APPR_PROCESS_START, ORGANIZATION_TYPE, OCCUPATION_TYPE。

**A4. 时间特征转换**
DAYS_* 列取绝对值后除以 365.25，转换为以年为单位的正值：
- DAYS_BIRTH → AGE_YEARS
- DAYS_EMPLOYED → YEARS_EMPLOYED
- DAYS_REGISTRATION → YEARS_REGISTRATION
- DAYS_ID_PUBLISH → YEARS_ID_PUBLISH
- DAYS_LAST_PHONE_CHANGE → YEARS_LAST_PHONE_CHANGE

**A5. 特征衍生（14 类，净增约 70 列）**
1. 财务比率：INCOME_CREDIT_RATIO, ANNUITY_INCOME_RATIO, CREDIT_GOODS_RATIO, ANNUITY_CREDIT_RATIO
2. 外部评分组合：EXT_SOURCE_WEIGHTED (均值), EXT_SOURCE_MIN, EXT_SOURCE_MAX, EXT_SOURCE_STD, EXT_SOURCE_1x2
3. 社交圈违约率：SOCIAL_DEF_30_RATIO, SOCIAL_DEF_60_RATIO, SOCIAL_HAS_DEFAULT
4. 征信局查询总量：BUREAU_ENQUIRY_TOTAL（6 个 AMT_REQ_CREDIT_BUREAU_* 求和）
5. 年龄分箱：AGE_BIN（6 段）、AGE_YOUNG (<30)、AGE_OLD (>60)
6. 收入分箱：INCOME_BIN（6 段）
7. 工作特征：EMPLOYED_AGE_RATIO, FLAG_NEW_EMPLOYEE (<2 年)
8. 家庭结构：CHILDREN_RATIO, ADULTS_COUNT
9. 人均指标：INCOME_PER_PERSON, CREDIT_PER_PERSON
10. 建筑特征交叉均值：每类建筑特征取 _AVG/_MODE/_MEDI 的均值
11. 文件提供数：DOCUMENTS_PROVIDED（20 个 FLAG_DOCUMENT 列求和）、DOCUMENTS_LOW
12. 地区评分组合：REGION_RATING_COMBO
13. 车辆年龄分箱：CAR_AGE_BIN
14. 注册/身份证时间差：REGISTRATION_ID_GAP, REGISTRATION_RECENT

处理后主表维度：191 列（已完成，产出文件：[data/processed/processed_application_train.csv](data/processed/processed_application_train.csv)）。

### 4.3 外部征信局特征工程 (队员 B)

数据源：bureau.csv (1.72M 行) + bureau_balance.csv (27.3M 行)

**阶段一：bureau 表聚合**
- 基础计数：按 SK_ID_CURR 统计历史信用总数 (BUREAU_COUNT)、活跃/关闭信用数、各 CREDIT_TYPE 的 one-hot 占比。
- 金额聚合：对 AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_OVERDUE, AMT_CREDIT_MAX_OVERDUE, AMT_CREDIT_SUM_LIMIT, AMT_ANNUITY 执行 {mean, max, min, sum}。
- 时间聚合：对 DAYS_CREDIT 取 mean/min/max（min = 最近信用距今，max = 最早信用距今 = 信用历史长度）。对 DAYS_CREDIT_ENDDATE 取 mean/min。
- 逾期聚合：对 CREDIT_DAY_OVERDUE 执行 max/mean/sum；对 CNT_CREDIT_PROLONG 执行 sum/mean。

**阶段二：bureau_balance 聚合（需先 join bureau 获取 SK_ID_CURR）**
- 状态分布：各 STATUS 值 (0/C/X/1/2/3/4/5) 的占比。
- 严重度指标：STATUS 映射为数值 (C→0, X→NaN, 0→0, 1→1, ..., 5→5) 后的均值 (STATUS_MEAN) 和最大值 (STATUS_MAX)。
- 时序趋势：最近 6/12/24 个月的 STATUS_MEAN 滑动窗口。
- 连续性指标：最长连续逾期月数（STATUS ≥ 1 的最大游程）。

**阶段三：衍生特征**
- 逾期月数占比：STATUS_GE1_RATIO = STATUS ≥ 1 月数 / 总月数。
- 逾期恶化趋势：STATUS_MEAN(最近 6 月) - STATUS_MEAN(全部历史)。
- 近期征信活跃度：最近 12 个月内有更新的信用占比。

预计产出 50-100 维，输出为 [data/processed/features_bureau.csv](data/processed/features_bureau.csv)。

### 4.4 内部历史行为特征工程 (队员 C)

数据源：previous_application.csv (1.67M), POS_CASH_balance.csv (10.0M), credit_card_balance.csv (3.84M), installments_payments.csv (13.6M)

**C1. previous_application 聚合**
- 计数与比例：PREV_APP_COUNT；批准/拒绝/取消占比（基于 NAME_CONTRACT_STATUS）；各 NAME_CONTRACT_TYPE, NAME_PORTFOLIO, CHANNEL_TYPE 占比。
- 金额统计：AMT_APPLICATION, AMT_CREDIT, AMT_DOWN_PAYMENT 的 mean/max/sum。
- 时间特征：DAYS_DECISION 的 mean/min（审批速度与最近一次申请距今）。
- 拒绝原因：CODE_REJECT_REASON 频率编码。
- 利率特征：RATE_INTEREST_PRIMARY, RATE_DOWN_PAYMENT 的 mean。

**C2. POS_CASH_balance 聚合**
- POS_COUNT（合同数量）；活跃合同占比（基于 NAME_CONTRACT_STATUS）。
- SK_DPD 的 max/mean（逾期天数）。
- CNT_INSTALMENT_FUTURE 的 mean/sum（剩余还款负担）。
- 最近 6 个月 SK_DPD 均值趋势。

**C3. credit_card_balance 聚合（全局 + 最近 6 个月）**
- 透支率：BALANCE_LIMIT_RATIO = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL，取 mean/max。
- 提现行为：AMT_DRAWINGS_ATM_CURRENT 和 CNT_DRAWINGS_ATM_CURRENT 的 mean/sum。
- 消费行为：AMT_DRAWINGS_POS_CURRENT 和 CNT_DRAWINGS_POS_CURRENT 的 mean/sum。
- 还款率：AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY 的 mean。
- SK_DPD 的 max/mean。
- 最近 6 个月透支率 vs 全局透支率的差值（恶化/改善信号）。

**C4. installments_payments 聚合（通过 SK_ID_PREV 关联至 previous_application 以获取 SK_ID_CURR）**
- 还款差额特征：DIFF_PAYMENT = AMT_INSTALMENT - AMT_PAYMENT，取 mean/sum/std（正值表示欠款）。
- 还款时间偏差：DIFF_DAYS = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT，取 mean/std（正值表示提前还款）。
- 逾期计数：DIFF_DAYS < 0 的期数占比。
- 还款完整度：SUM(AMT_PAYMENT) / SUM(AMT_INSTALMENT)。
- 最近 6 个月上述指标的趋势。

**C5. 跨表衍生**
- PREV_APPROVED_SUM / AMT_CREDIT：历史批准总额与当前贷款金额比。
- 高频提现标记：ATM 提现次数 > 75% 分位数。
- 多平台借贷标记：同时存在 POS + 信用卡 + 分期记录。

预计产出 80-150 维，输出为 [data/processed/features_internal.csv](data/processed/features_internal.csv)。

### 4.5 特征合并 (队员 A — A6)

以 SK_ID_CURR 为键，Left Join 合并三项特征产出：

```
processed_application_train.csv (队员 A)
  ← features_bureau.csv (队员 B)
  ← features_internal.csv (队员 C)
```

合并后在辅助表中无对应记录的主表行，缺失值以中位数填充。最终训练宽表预计 300-450 维。

### 4.6 特征选择 (队员 D)

在模型训练前，按以下顺序执行特征降维：
1. 缺失率过滤：移除合并后缺失率 > 60% 的列。
2. 方差过滤：移除方差 < 0.01 的近似常量列。
3. Pearson 相关性去冗余：对 |r| > 0.95 的特征对，移除与 TARGET 相关性的绝对值较小者。
4. 树模型特征重要性筛选：使用 LightGBM 以默认参数训练后，取 importance top-k（k 取 100, 150, 200，通过 CV 确定最优值）。

### 4.7 建模方案

#### 类别不平衡处理

对比两种策略并选择较优方案：
- 样本层面：SMOTE 过采样（k_neighbors=5），仅对训练折中的训练集部分应用。
- 算法层面：XGBoost/LightGBM 设置 scale_pos_weight = (负样本数 / 正样本数)。

#### 算法选择

| 算法 | 类型归属 | 选择依据 |
|------|----------|----------|
| Logistic Regression | 线性模型 | 可解释性基线：回归系数直接表示各特征对对数几率的影响方向和大小 |
| Random Forest | Bagging 集成 | Bootstrap 采样 + 随机子空间，低方差、对异常值鲁棒 |
| XGBoost | Boosting 集成 | 二阶泰勒展开近似目标函数，引入叶节点权重 L1/L2 正则；Level-wise 生长 + 预排序 |
| LightGBM | Boosting 集成 | 直方图算法 + Leaf-wise 生长 + GOSS 采样；训练速度显著快于 XGBoost，适合高维数据 |
| MLP | 神经网络 | 引入非线性深度变换，检验传统 ML 在该结构化数据上是否已达性能边界 |

五种算法覆盖线性→非线性、Bagging→Boosting、浅层→深层的设计空间。XGBoost 和 LightGBM 虽均属 Boosting，但优化策略、树生长方式和采样机制有实质差异，构成可对比的两个独立变体。

#### 超参数调优

使用 Optuna (TPE Sampler)，每个模型 100 trials。
- Logistic Regression: C [1e-3, 1e3] (log-uniform), penalty ∈ {l1, l2}。
- Random Forest: n_estimators [100, 1000], max_depth [5, 30], min_samples_split [2, 50]。
- XGBoost: max_depth [3, 12], learning_rate [0.01, 0.3], subsample [0.6, 1.0], colsample_bytree [0.6, 1.0], reg_alpha [0, 10], reg_lambda [0, 10]。
- LightGBM: num_leaves [15, 255], learning_rate [0.01, 0.3], feature_fraction [0.6, 1.0], min_child_samples [10, 100], reg_alpha [0, 10]。
- MLP: hidden_layer_sizes (1-3 层，每层 32-256 神经元), alpha [1e-5, 1e-1], learning_rate_init [1e-4, 1e-2]。

### 4.8 评估框架

#### 评估指标

不使用 Accuracy（全预测为 0 即可获得 ~92% 准确率，不具判别价值）。

| 指标 | 含义 | 适用场景 |
|------|------|----------|
| AUC-ROC | 模型将正样本排在负样本之前的概率 | 衡量排序能力，不受分类阈值影响 |
| Recall | TP / (TP + FN) | 衡量对违约客户的捕获率，漏判代价高 |
| Precision | TP / (TP + FP) | 衡量预测为违约的精确度 |
| F1-Score | 2PR / (P+R) | Precision 与 Recall 的调和平均 |

主要对比指标为 AUC-ROC 和 Recall。

#### 验证策略

5-Fold Stratified Cross-Validation：
- 每折维持与全集相同的 TARGET 分布。
- 所有模型使用相同 Fold 划分（random_state=42）。
- 报告格式：mean ± std over 5 folds。

附加验证：以最优模型在 application_test.csv 上预测并提交 Kaggle，以 Public Leaderboard Score 作为外部参考（不纳入正式评分）。

### 4.9 模型可解释性分析

- 树模型（选择 CV 最优者）：SHAP Summary Plot（全局 Top-20 特征及影响方向）、SHAP Waterfall Plot（单样本决策分解）、SHAP Dependence Plot（关键特征与预测值的非线性关系）。
- Logistic Regression：标准化回归系数（Odds Ratio），输出系数最大的正/负向特征各 10 个。

---

## 5. 团队分工计划

### 5.1 分工原则

分工按数据源边界切分，而非按功能模块切分。理由如下：
1. 各数据表的聚合特征工程互不依赖，可并行推进。
2. 每位成员负责特定数据表的全流程（从原始数据清洗到特征产出），对该数据表形成领域专长，符合期末项目 Tips 中"在自己领域成为专家"的要求。
3. 模型训练（队员 D）需等待特征合并完成，模型训练层工作与前三人解耦。

### 5.2 角色与职责

| 角色 | 姓名 | 负责数据 | 核心职责 | 依赖关系 |
|------|------|----------|----------|----------|
| 队员 A | — | application_train.csv / application_test.csv (307K 行, 122 列) | 主表清洗 (A1-A2)、类别编码 (A3)、时间转换 (A4)、特征衍生 (A5)、最终宽表合并 (A6) | 依赖 B、C 交付特征表后方可执行 A6 |
| 队员 B | — | bureau.csv (1.72M 行) + bureau_balance.csv (27.3M 行) | 外部征信局统计聚合 (B1-B5)、衍生特征 (B6)、特征表交付 (B7) | 无前置依赖 |
| 队员 C | — | previous_application.csv (1.67M) + POS_CASH_balance.csv (10.0M) + credit_card_balance.csv (3.84M) + installments_payments.csv (13.6M) | 内部历史行为特征聚合 (C1-C4)、跨表衍生 (C5)、特征表交付 (C6) | 无前置依赖 |
| 队员 D | — | 最终宽表 (接收自 A) | 特征选择 (D1-D3)、5 种算法训练与 Optuna 调参 (D4-D8)、统一评估 (D9)、SHAP 分析 (D10)、报告统稿与 PPT (D11-D12) | 依赖 A 产出 final_feature_matrix.csv |

### 5.3 协作机制

- 代码仓库：统一使用 GitHub 仓库（https://github.com/FutuXer/DAM-Final），每人建分支开发，A6 和 D 阶段合并至 main。
- 数据交付：特征产出统一为 CSV 格式，以 SK_ID_CURR 为列，放置于 `data/processed/` 目录。
- 同步频率：每 2-3 天通过群消息同步进度；遇到内存/数据问题即时沟通。
- Commit 规范：`[角色] 简要描述`，如 `[B] bureau 逾期聚合特征完成`。

---

## 6. 项目 Pipeline

```
data/raw/*.csv (8 files, ~2.5 GB)
    │
    ├── 队员A: src/data/preprocess_application.py
    │   ├── A1 缺失值处理 (中位数/众数填充 + 高缺失标记)
    │   ├── A2 异常值处理 (DAYS_EMPLOYED=365243 → 无业标记; 99% Winsorize)
    │   ├── A4 时间转换 (DAYS_* → 年)
    │   ├── A5 特征衍生 (14 类, 净增 ~70 列)
    │   └── A3 类别编码 (≤5: One-Hot; >5: Target Encoding)
    │   → data/processed/processed_application_train.csv (307K × 191)
    │   → data/processed/processed_application_test.csv  (49K × 191)
    │
    ├── 队员B: src/data/preprocess_bureau.py
    │   ├── B1-B4 bureau 聚合 (计数/金额/时间/逾期)
    │   ├── B5 bureau_balance 状态聚合 (STATUS 分布/时序趋势)
    │   └── B6 衍生 (逾期占比/恶化趋势)
    │   → data/processed/features_bureau.csv (307K × 50-100)
    │
    ├── 队员C: src/data/preprocess_internal.py
    │   ├── C1 previous_application 聚合
    │   ├── C2 POS_CASH 聚合
    │   ├── C3 credit_card 聚合 (+ 最近 6 个月滑动窗)
    │   ├── C4 installments 聚合 (还款差额/时间偏差)
    │   └── C5 跨表衍生
    │   → data/processed/features_internal.csv (307K × 80-150)
    │
    └── 队员A: A6 — merge_with_features()
        → data/processed/final_feature_matrix.csv (307K × 300-450)
            │
            └── 队员D: src/models/model_training.py
                ├── D1 预处理 (StandardScaler, train/val/test 划分)
                ├── D2 不平衡处理 (SMOTE vs scale_pos_weight)
                ├── D3 特征选择 (缺失率→方差→Pearson→Tree importance)
                ├── D4-D8 5 种算法训练 (LR, RF, XGB, LGB, MLP) + Optuna
                ├── D9 统一评估 (AUC-ROC/Recall/F1 + 5-Fold CV)
                └── D10 SHAP 可解释性分析
```

---

## 7. 时间表

### 7.1 阶段划分与关键节点

| 阶段 | 时间 | 关键节点 | 队员 A | 队员 B | 队员 C | 队员 D |
|------|------|----------|--------|--------|--------|--------|
| **第 1 周**：数据预处理 | 第 1-3 天 | 数据下载就位、环境配置完成 | A1-A2（缺失值与异常值） | 熟悉 bureau 表结构、设计聚合方案 | 熟悉 4 张内部表结构、设计聚合方案 | 安装依赖包、阅读竞赛论文/Notebook |
| | 第 4-7 天 | 主表特征初步产出 | A3-A5（编码、时间、衍生） | B1-B4（bureau 聚合代码编写） | C1-C2（previous_application + POS 聚合） | 学习 SHAP 文档、搭建训练框架 |
| **第 2 周**：特征工程收尾 + 建模启动 | 第 8-10 天 | B/C 特征交付 | A6（合并 B、C 特征为宽表） | B5-B6（balance 聚合 + 衍生，收尾交付） | C3-C4（credit_card + installments 聚合） | D1-D3（特征选择，确认最终训练维度） |
| | 第 11-14 天 | 模型基线跑通 | 辅助 A6 调试（处理合并后的列冲突/缺失） | 配合调试（特征与预期不符时调整） | C5-C6（跨表衍生，收尾交付） | D4-D6（LR/RF/XGBoost 基线训练 + Optuna） |
| **第 3 周**：模型调优与评估 | 第 15-18 天 | 模型调参完成 | 撰写报告"主表特征工程"章节 | 撰写报告"外部征信特征"章节 | 撰写报告"内部行为特征"章节 | D7-D9（LightGBM/MLP + 5-Fold CV 对比评估） |
| | 第 19-21 天 | 评估结果确定 | — | — | — | D10（SHAP 分析 + 图表产出） |
| **第 4 周**：报告与答辩 | 第 22-25 天 | 报告初稿 | 提交负责章节 | 提交负责章节 | 提交负责章节 | 统稿、排版、查漏补缺 |
| | 第 26-28 天 | 终稿 + PPT | 辅助 PPT 素材 | 辅助 PPT 素材 | 辅助 PPT 素材 | PPT 制作、全员审核 |
| | 第 29-30 天 | 答辩准备 | 演练 | 演练 | 演练 | 演练 + 最终定稿 |

### 7.2 时间表说明

- 第 1 周：A/B/C 三名成员的特征工程工作完全并行，互不阻塞。A 的 A1-A5（主表特征）不依赖 B/C 产出。
- 第 2 周：B 和 C 需在第 2 周中期交付特征表，A 在收到后 2 天内完成 A6 合并。D 在第 2 周下半周开始建模，确保调参有充足时间。
- 第 3 周：D 完成全部 5 种模型的 Optuna 调参和 CV 评估，A/B/C 同步撰写报告的特征工程章节。
- 第 4 周：D 负责统稿和 PPT 定稿；第 4 周末全员演练。

### 7.3 高优先级里程碑

| 里程碑 | 截止时间 | 判定标准 |
|--------|----------|----------|
| M1: 数据就位 | 第 1 周第 2 天 | 所有 .csv 文件存在于 data/raw/ 目录 |
| M2: 主表处理完成 | 第 1 周第 7 天 | processed_application_train.csv 产出，0 缺失值，0 object 列 |
| M3: B/C 特征交付 | 第 2 周第 10 天 | features_bureau.csv + features_internal.csv 交付至 data/processed/ |
| M4: 最终宽表产出 | 第 2 周第 12 天 | final_feature_matrix.csv 产出，列数 300-450，无缺失 |
| M5: 5 种模型基线 | 第 2 周第 14 天 | 5 种算法均在 5-Fold CV 上跑通，AUC-ROC > 0.65 |
| M6: 调参完成 | 第 3 周第 19 天 | Optuna 100 trials × 5 模型完成，最优参数确定 |
| M7: 评估报告 | 第 3 周第 21 天 | CV 对比表 + ROC 曲线图 + SHAP 图全部产出 |
| M8: 报告定稿 | 第 4 周第 28 天 | 项目报告 PDF（10-20 页）定稿 |
| M9: PPT 定稿 | 第 4 周第 29 天 | 答辩 PPT 定稿 |
| M10: 答辩 | 第 4 周第 30 天 | 全员预演 ≥ 1 次，正式答辩 |

---

## 8. 交付物清单

| 序号 | 交付物 | 格式 | 负责 | 对应评分项及占比 |
|------|--------|------|------|------------------|
| 1 | 数据预处理与特征工程代码 | `.py` + `.ipynb` | A, B, C | 代码实现 (25%) |
| 2 | 5 种算法训练与评估代码 | `.py` + `.ipynb` | D | 代码实现 (25%) |
| 3 | 项目技术报告 | PDF (10-20 页) | D 统稿, A/B/C 供素材 | 报告 (30%) |
| 4 | 答辩 PPT | `.pptx` | D 主制, 全员审核 | 演示答辩 (25%) |
| 5 | Git 仓库 + commit 记录 | GitHub | 全员 | 仓库与团队协作 (10%) |
| 6 | 团队分工说明 | `分工.md` | A | 分工 (5%) |
| 7 | 海报 (加分项) | 图片/PDF | 全员 | 加分 |

---

## 9. 技术环境

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.10+ |
| 核心库 | pandas, numpy, scikit-learn 1.x |
| 集成树 | xgboost, lightgbm |
| 神经网络 | scikit-learn MLPClassifier |
| 不平衡处理 | imbalanced-learn (SMOTE) |
| 超参调优 | Optuna (TPE Sampler) |
| 模型解释 | SHAP |
| 可视化 | matplotlib, seaborn |
| 环境复现 | pip + requirements.txt |
| 版本控制 | Git + GitHub |

---

## 10. 风险识别与对策

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| bureau_balance 27.3M 行聚合时内存溢出（>16GB） | 中 | B 工作阻塞 | 使用 `pd.read_csv(chunksize=500000)` 分块聚合；降采样至近 24 个月；仅保留 STATUS ≥ 1 的记录做精细聚合 |
| installments_payments 13.6M 行 join previous_application 时内存溢出 | 中 | C 工作阻塞 | 预先在 previous_application 上建立 SK_ID_PREV → SK_ID_CURR 映射字典（1.67M 行），用 map 替代 merge |
| 最终特征维度 > 500 导致训练缓慢 | 中 | D 训练周期过长 | 严格按 D3 流程执行特征选择；Optuna 中纳入特征采样；LightGBM 天然支持特征分桶 |
| 类别不平衡处理失当导致 Recall 极低 | 低 | 指标对比失真 | 同时跑 SMOTE 和 scale_pos_weight 两组实验，取 CV 最优方案 |
| 队员间交付延迟导致 D 等待 | 中 | 第 2 周建模无法启动 | A 的 A1-A5 独立完成；D 可先基于 A 的主表特征（不含 B/C 特征）跑 Baseline 模型，验证 Pipeline 可用性 |
