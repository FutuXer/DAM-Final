# 项目规划书：日本旅游多源数据挖掘与情绪地标聚类

> **对应选题**：选项 A —— 多源数据融合挖掘
>
> **团队规模**：4 人 | **项目周期**：5 周 | **难度级别**：中高

---

## 一、项目概述

### 1.1 选题背景

传统旅游推荐系统主要依赖地理距离和单一评分维度，忽略游客的真实体验语义——两个物理上相邻的景点可能在游客感受中完全不同，而两个相隔很远的居酒屋可能共享高度一致的"深夜治愈感"。本项目通过融合物理空间的结构化 POI 数据与游客游记/评论的非结构化文本数据，在"地理距离 + 语义相似度"的联合特征空间中重新发现城市的功能区划，挖掘传统地图上不可见的**情绪游玩圈**（如：二次元圣地巡礼圈、深夜美食圈、古迹静谧圈、购物狂欢圈等）。

### 1.2 核心创新点

| 创新维度 | 具体内容 |
|---------|---------|
| **多模态融合** | 将 4 个来源的异构数据——地理坐标、评论文本、交通路网、区域统计——进行特征级拼接，构建 390+ 维联合特征向量 |
| **度量重构** | 设计联合距离公式 $D_{joint} = \alpha \cdot D_{geo} + (1-\alpha) \cdot (1 - Sim_{text})$，替代传统欧式距离 |
| **隐性模式挖掘** | 在重构特征空间中用无监督聚类发现跨行政区的"情绪商圈" |
| **系统性对比** | 建立多维度对比矩阵：NLP 方法 × 聚类算法 × 距离权重，提供量化证据链 |

---

## 二、数据源方案（四源零爬虫）

### 2.1 数据源总览

本项目对标选项 A 最高难度示例（城市犯罪预测：4 数据源），整合 **4 个不同模态/来源** 的数据集。**全部数据源均为免费、即时下载、无需爬虫**。

| # | 数据源 | 模态 | 具体来源 | 获取难度 |
|:--:|------|:--:|------|:--:|
| ① | **POI 地理数据** | 结构化（点状空间） | HOTOSM / Geofabrik OSM | ★☆☆☆☆ (零门槛) |
| ② | **旅游评论文本** | 非结构化（自然语言） | HuggingFace `itinerai/attractions` + `ACOSRes` | ★☆☆☆☆ (零门槛) |
| ③ | **铁路交通数据** | 结构化（网络空间） | ekidata.jp / GitHub `open-data-jp-railway-stations` | ★☆☆☆☆ (零门槛) |
| ④ | **区域统计数据** | 结构化（表格） | e-Stat SSDSE（日本政府统计门户） | ★★☆☆☆ (极低) |

> **核心原则**：全部数据源均为公开、免费、即时下载，无爬虫依赖，确保项目可复现、可验证。这与作业要求中"模块化、可一键运行"高度契合。

---

#### 数据源 ①：结构化 POI 地理数据

| 项目 | 内容 |
|------|------|
| **首选来源** | [HOTOSM Japan Points of Interest](https://data.humdata.org/dataset/hotosm_jpn_points_of_interest)（Humanitarian Data Exchange） |
| **备用来源** | [Geofabrik Japan OSM Extract](https://download.geofabrik.de/asia/japan.html)（每日更新，含 `gis_osm_pois_free_1` 图层） |
| **获取方式** | 直接下载 GeoJSON（37.9 MB 点位 / 74.8 MB 面状），`pandas.read_json()` 一键转 CSV |
| **覆盖范围** | 日本全境，月度更新 |
| **核心字段** | POI ID、名称（日/英）、经纬度、`fclass` 类别标签（150+ 类）、几何类型 |
| **预估量级** | 全日本 50,000+ POI → 筛选后 8,000–15,000 条 |
| **POI 类别筛选** | 聚焦四类旅游强相关：餐饮（restaurant/cafe/bar）、景点（tourism/attraction）、购物（shop/mall）、住宿（hotel） |
| **补充手段** | Overpass API 按需补充特定城市/类别的营业时间、评分等属性字段 |

#### 数据源 ②：非结构化旅游评论文本

| 项目 | 内容 |
|------|------|
| **首选来源** | [HuggingFace `itinerai/attractions`](https://huggingface.co/datasets/itinerai/attractions)（TripAdvisor 日本景点数据） |
| **补充来源** | [HuggingFace `ACOSRes`](https://huggingface.co/datasets/LujainAbdulrahman/ACOSRes)（日本餐厅四元组评论文本） |
| **获取方式** | HuggingFace `datasets` 库一键加载：`load_dataset("itinerai/attractions")` |
| **覆盖范围** | 以东京为主的日本景点 + 餐厅 |
| **核心字段** | 评论文本（DESCRIPTION）、评分（RATING）、标签（REVIEW_TAGS）、类别（CATEGORIES）、**经纬度**（LAT/LON） |
| **预估量级** | 数千条带经纬度的评论文本数据 |
| **关键优势** | 已有经纬度坐标，可直接通过 L0 空间匹配与 POI 数据关联，免去繁重的文本实体对齐工作 |
| **备用增强** | 如时间允许，可向 [NII IDR 申请 Rakuten Travel Reviews](https://www.nii.ac.jp/dsc/idr/rakuten/)（729 万条日文酒店评论），申请期间不阻塞项目进度 |

> **重要决策记录**：B站弹幕方案已排除（短文本信息密度低 + wbi 签名验证复杂）。携程/马蜂窝爬虫方案降级为**可选的补充中文评论文本来源**——项目不依赖爬虫产出，爬虫成功是锦上添花，失败不影响项目交付。

#### 数据源 ③：铁路交通基础设施数据

| 项目 | 内容 |
|------|------|
| **来源** | [ekidata.jp](https://www.ekidata.jp/)（日本铁路站点开放数据，CC BY 4.0） |
| **镜像** | [GitHub `piuccio/open-data-jp-railway-stations`](https://github.com/piuccio/open-data-jp-railway-stations)（JSON 格式，含坐标） |
| **获取方式** | 直接下载 CSV / JSON，含全日本所有铁路站点 |
| **核心字段** | 站名（汉字/假名/罗马字）、**经纬度**、线路 ID、所属公司、开业日期 |
| **衍生特征** | 计算每个 POI 到最近车站的 Haversine 距离 → **可达性得分**；统计 POI 1km 半径内车站数量 → **交通便利度** |
| **业务价值** | 交通可达性是旅游体验的核心维度——"离车站步行 5 分钟的居酒屋"和"需要打车才能到的山顶餐厅"属于完全不同的情绪圈 |

#### 数据源 ④：区域人口与经济社会统计

| 项目 | 内容 |
|------|------|
| **来源** | [e-Stat SSDSE](https://www.nstac.go.jp/SSDSE/)（日本政府统计门户，教育用标准数据集） |
| **获取方式** | 直接下载 CSV，覆盖日本全部 1,741 市区町村 |
| **核心字段** | 总人口、人口密度、老龄化率、外国人比例、服务业从业人数、可住面积 |
| **衍生特征** | 按行政区划将人口/经济指标关联到 POI → 区域特征向量（城市化程度、旅游化程度） |
| **业务价值** | 区分"本地居民常去的社区居酒屋"与"国际化旅游区的打卡餐厅"——这两类地点的情绪标签完全不同 |

### 2.2 数据获取策略：零爬虫依赖

```
第 1 周工作流（全部并行执行，无需等待）：
  Day 1-2: 队员 A 下载 HOTOSM GeoJSON → 转 CSV → 筛选 POI 类别
  Day 1-2: 队员 B 从 HuggingFace 加载 attractions/ACOSRes → 检查字段完整性
  Day 1-2: 队员 D 从 ekidata.jp 下载车站 CSV → 验证经纬度覆盖
  Day 2-3: 队员 C 从 e-Stat 下载 SSDSE → 按市区町村整理
  Day 3-5: 全员 → 数据清洗 + 特征拼接实验
  Day 5:   决策点 → 是否需要启动爬虫补充中文评论？（如不需要，不启动）

爬虫定位：可选增强项，非必需依赖。项目核心 Pipeline 不依赖任何爬虫产出。
```

---

## 三、核心技术路线

### 3.1 整体 Pipeline

```
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ ① OSM POI 数据 │  │ ② 评论文本数据  │  │ ③ 铁路站点数据  │  │ ④ 区域统计数据  │
│ (结构化/点状)  │  │ (非结构化/文本) │  │ (结构化/网络)   │  │ (结构化/表格)   │
│ HOTOSM 全日本  │  │ HuggingFace    │  │ ekidata.jp     │  │ e-Stat SSDSE   │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │                   │
        ▼                   ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 坐标归一化    │  │ 文本预处理     │  │ 最近站距离     │  │ 按市区町村     │
│ 类别独热编码  │  │ 分词/去停用词  │  │ 1km站点密度   │  │ 关联 POI      │
│ fclass→特征  │  │ 实体对齐→POI  │  │ 可达性得分     │  │ 人口/经济特征  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                   │                   │                   │
        │                   ▼                   │                   │
        │          ┌──────────────┐             │                   │
        │          │ NLP 向量化    │             │                   │
        │          │ 3 种方案对比  │             │                   │
        │          └──────┬───────┘             │                   │
        │                   │                   │                   │
        └───────────────────┼───────────────────┴───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ 四源特征拼接     │
                  │ 联合特征矩阵     │
                  │ (Geo 2d +       │
                  │  Text 384d +    │
                  │  Transit 2d +   │
                  │  Demo 5d)       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 联合距离度量     │
                  │ D_joint(α)      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 无监督聚类       │
                  │ 3 种算法 ×      │
                  │ 2 种特征组合     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 评估 & 可视化    │
                  └─────────────────┘
```

### 3.2 实体对齐专项方案

**关键前提**：数据源 ②（HuggingFace `itinerai/attractions`）已自带经纬度坐标，可直接通过空间邻近度与 POI 数据关联，大幅降低实体对齐难度。

**三级对齐策略**（按优先级级联）：

| 层级 | 方法 | 条件 | 目标 | 预期覆盖率 |
|------|------|------|------|:----------:|
| **L0 空间匹配** | 评论文本自带 LAT/LON + Haversine 距离最近 POI < 50m | 数据源 ② 有坐标 | 直接匹配，无需文本比对 | ~70% |
| **L1 精确匹配** | 文本中直接出现 POI 标准名称（中/日/英） | 数据源 ② 无坐标 | 清水寺 ↔ Kiyomizudera | ~15% |
| **L2 模糊匹配** | 编辑距离 (Levenshtein) + 同义词词典 | L1 未命中 | "清水舞台" ↔ 清水寺 | ~10% |
| 无法对齐 | 标记为未匹配，仅参与纯文本聚类分析 | | | ~5% |

> 相比原方案，**L0 空间匹配层的新增使得实体对齐从文本匹配难题转变为空间索引任务**，可靠性和效率均大幅提升。

**对齐质量评估**：随机抽样 100 条对齐结果，人工标注正确性，计算 Precision/Recall。

### 3.3 NLP 语义向量化方案对比（3 种）

| 方法 | 维度 | 算力需求 | 优势 | 劣势 |
|------|:----:|:--------:|------|------|
| **TF-IDF** | 可调（Top-K 关键词） | 极低 | 速度快、可解释性强、关键词可直接用于业务解读 | 丢失上下文语义、无法捕捉同义词 |
| **Word2Vec** (轻量日文预训练) | 200–300 | 低 | 捕捉词级语义、对 OOV 有一定鲁棒性 | 需要日文分词、聚合方式影响效果 |
| **Multilingual-MiniLM** | 384 | 中等 | 捕捉上下文语义、多语言支持好、HuggingFace 直接可用 | 推理时间较长、可解释性差 |

> **策略**：三种方法全部实现，作为对比实验的一维。如果算力紧张，MiniLM 可以只在最终阶段使用。

### 3.4 联合距离度量公式

$$D_{joint}(p_i, p_j) = \alpha \cdot D_{geo}^{norm}(p_i, p_j) + (1-\alpha) \cdot (1 - Sim_{text}(v_i, v_j))$$

其中：
- $D_{geo}^{norm}$：归一化后的球面距离（Haversine），缩放到 [0, 1]
- $Sim_{text}$：文本向量的余弦相似度，取值 [-1, 1]，映射到 [0, 1]
- $\alpha \in [0, 1]$：地理距离的权重，通过网格搜索确定最优值

**物理含义**：
- $\alpha \to 1$：退化为纯地理聚类（Baseline）
- $\alpha \to 0$：纯语义聚类，完全忽略物理位置
- 最优 $\alpha$ 代表"地理约束"与"体验共鸣"的最佳平衡点

---

## 四、聚类算法对比矩阵设计

建立 **3（NLP 方案）× 3（聚类算法）× 3（特征组合）× N（α 取值）** 的多维对比空间：

### 4.1 聚类算法（3 种）

| 算法 | 距离度量 | 参数 | 适用场景 |
|------|---------|------|---------|
| **K-Means** | 支持自定义距离 | K（簇数） | 球形簇、各向同性，作为经典 Baseline |
| **DBSCAN** | 支持自定义距离 | eps, min_samples | 任意形状簇、可发现噪声点，适合城市中的"稀疏景点" |
| **层次聚类 (Agglomerative)** | 支持自定义距离 | K、linkage 方式 | 可产生层次结构（商圈→子商圈），解释性强 |

### 4.2 对比实验拓扑

```
对比维度 1: NLP 向量化方法
  ├── TF-IDF (Top-500 keywords)
  ├── Word2Vec (200d)
  └── MiniLM (384d)

对比维度 2: 聚类算法
  ├── K-Means
  ├── DBSCAN
  └── Agglomerative Clustering

对比维度 3: 特征组合（消融实验，验证多源增益）
  ├── 2-Src: Geo(2d) + Text(384d)          ← 最小两源
  ├── 3-Src: Geo(2d) + Text(384d) + Transit(2d)  ← 验证交通增量
  └── 4-Src: Geo + Text + Transit + Demo(5d)      ← 完整四源

对比维度 4: 距离权重 α
  └── α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

报告呈现: 选取代表性配置 ~30 组，完整结果见附录或代码仓库
```

### 4.3 Baseline 定义

| Baseline 编号 | 特征组合 | 算法 | 目的 |
|:---:|------|------|------|
| **B0** | 仅 Geo (2d) | K-Means | 纯地理聚类——传统旅游推荐系统的水平 |
| **B1** | 仅 Text (384d) | K-Means | 纯语义聚类——验证语义信号的独立价值 |
| **B2** | 2-Src (Geo + Text) | K-Means（α=0.5） | 两源融合 Baseline——参数寻优起点 |
| **B3** | 4-Src Full | K-Means（α=0.5） | 四源融合 Baseline——验证多源信息增益 |

---

## 五、评估指标体系

### 5.1 内部评估（无监督，不依赖标注）

| 指标 | 公式/说明 | 用途 |
|------|---------|------|
| **轮廓系数 (Silhouette Score)** | $s = \frac{b-a}{\max(a,b)}$ | 衡量簇内紧密度与簇间分离度，主评估指标 |
| **戴维斯-布尔丁指数 (DBI)** | $\frac{1}{K}\sum_{i=1}^{K}\max_{j \neq i}\frac{\sigma_i+\sigma_j}{d(c_i,c_j)}$ | 越小越好，衡量簇间相似度 |
| **Calinski-Harabasz 指数** | $\frac{tr(B_K)}{tr(W_K)} \times \frac{N-K}{K-1}$ | 方差比率准则，辅助验证 |

### 5.2 外部验证（定性）

- **聚类可视化验证**：用 Folium/Kepler.gl 将聚类结果渲染在地图上，人工检查聚类标签的地理连续性与语义一致性
- **案例归因分析**：选取 3–5 个典型聚类，还原到原始评论中验证其"情绪一致性"——这是报告"讨论"章节的核心素材

### 5.3 统计显著性检验

- 对最佳配置 vs Baseline 的轮廓系数差异做 **Wilcoxon 符号秩检验**（多次随机初始化的结果）

---

## 六、预期成果与交付物

严格对齐课程要求的 7 项交付物：

| # | 课程要求 | 本项目对应产出 |
|:--:|---------|---------------|
| 1 | **项目计划书** (5%) | 即本文档，第 13 周前提交 |
| 2 | **代码仓库** (10%) | GitHub 仓库，4 人完整的 commit 历史，含 README |
| 3 | **完整代码** (25%) | 模块化 Python 代码，`python main.py` 一键运行全流程 |
| 4 | **项目报告** (30%) | 10–20 页学术论文格式：摘要、引言、方法、实验、讨论、结论 |
| 5 | **团队分工声明** (5%) | 逐条列出每人贡献的代码文件、报告章节、工作量百分比 |
| 6 | **答辩 PPT + 演示** (25%) | 10 分钟答辩（8 分钟讲解 + 2 分钟交互地图演示） |
| 7 | **项目海报** (加分项) | A1 尺寸学术海报，突出 Pipeline 与核心聚类效果图 |

### 交互式成果展示（精简方案）

> **已从原方案调整**：不再使用 Vue3 + 大模型对话 + 旅游路线规划（该方案属于选项 C，工作量对 5 周项目不切实际）。

调整为 **Folium + Streamlit 轻量方案**：

1. **Folium 交互式地图**（核心）：将聚类结果按不同颜色渲染在日本地图上，支持缩放、点击查看 POI 详情与聚类标签的语义关键词
2. **Streamlit 对比面板**（可选增强）：提供下拉框切换不同的 NLP 方案 / 聚类算法 / α 值，实时查看聚类结果变化和评估指标对比
3. 如果时间和能力允许，Streamlit 面板中增加一个简单的"条件筛选"功能：用户选择偏好关键词（如"安静"、"美食"），系统高亮匹配的聚类区域

---

## 七、团队角色与分工

### 7.1 角色定义

| 角色 | 负责人 | 核心职责 | 可考核产出 |
|------|:------:|---------|-----------|
| **数据工程负责人** | 队员 A | 数据源 ①④ 下载与清洗（HOTOSM POI + e-Stat）、实体对齐（L0 空间匹配）、可达性特征计算、四源特征拼接 | 原始数据 CSV、四源清洗后特征宽表、实体对齐覆盖率报告 |
| **算法建模负责人** | 队员 B | 数据源 ② 加载、NLP 向量化（3 种方案）、联合距离度量实现、聚类算法（3 种）× 特征组合（3 种）实现与调参 | 模型训练代码、网格搜索结果表、最优参数配置 |
| **评估与分析负责人** | 队员 C | 数据源 ③ 下载与衍生特征、消融实验设计、评估指标代码、对比图表输出、统计检验、聚类案例归因分析 | 评估代码、消融对比图表（含 Silhouette/DBI 曲线）、报告"实验"与"讨论"章节初稿 |
| **工程与交付负责人** | 队员 D | 代码模块整合、主流程 Pipeline 编排、Folium 交互地图、GitHub 仓库管理、报告统稿、答辩 PPT 制作 | `main.py` 一键运行脚本、README、可交付的 HTML 地图、最终报告 PDF、答辩 PPT |

### 7.2 协作机制

- **主责分明**：每人对自己负责的模块代码质量负全责
- **交叉审查**：队员 A 与 B 互相 review 数据接口，队员 C 与 D 互相 review 可视化产出
- **第 1 周全员参与**：数据获取阶段所有人同步工作（HOTOSM + HuggingFace + ekidata + e-Stat），降低单点瓶颈风险
- **Git 规范**：每人独立分支开发，合并前至少一人 review

---

## 八、五周时间表

| 周次 | 时间 | 里程碑 | 队员 A (数据) | 队员 B (算法) | 队员 C (评估) | 队员 D (工程) |
|:----:|------|--------|:---:|:---:|:---:|:---:|
| **W1** | 第 13 周 | **计划书提交 + 四源数据到手** | HOTOSM POI 下载、筛选 | HuggingFace 数据加载 | e-Stat 数据下载整理 | ekidata.jp 车站数据下载 | 仓库初始化、环境配置、Pipeline 骨架 |
| **W2** | 第 14 周 | **特征矩阵产出** | 实体对齐（L0 空间匹配）+ 可达性特征计算 | NLP 向量化（TF-IDF + Word2Vec 先行） | 区域统计特征关联 + 数据质量评估 | 四源特征拼接脚本 + MiniLM 模型准备 |
| **W3** | 第 15 周 | **算法实验完成** | 协助消融实验（切换特征组合） | 联合距离度量 + 3 种算法 × 3 种特征组合 | 网格搜索脚本 + 评估指标代码 | 集成 B 和 C 的代码到主 Pipeline |
| **W4** | 第 16 周 | **全部分析产出** | 协助案例归因分析 | 参数寻优完成、最优结果确定 | 所有对比图表输出 + 报告"实验"章节 | Folium 地图开发、Streamlit 对比面板 |
| **W5** | 第 17 周 | **答辩准备** | 报告"数据与预处理"章节撰写 | 报告"方法"章节撰写 | 报告"讨论"章节撰写 | 报告统稿、PPT 制作、答辩排练 |

### 关键时间节点

- **W1 周三**：四源数据全部到齐，完成初步质量检查
- **W2 周一**：实体对齐（L0 空间匹配）V1 完成 + 可达性/人口特征计算完成，产出第一版特征宽表
- **W3 周五**：核心实验组（2-Src + 3-Src + 4-Src × 最优 α）跑完，产出原始结果
- **W4 周五**：所有对比图表、消融分析、Folium 地图、报告初稿完成
- **W5 周三**：最终报告 PDF 定稿、答辩 PPT 定稿
- **W5 周五**：答辩

---

## 九、风险预估与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|:---:|:---:|---------|
| HuggingFace 数据量不足或字段缺失 | 低 | 高 | 启动爬虫补充中文评论（携程 Top 50 景点）；或申请 Rakuten Travel Reviews 作为大规模后备 |
| 实体对齐 L0 空间匹配精度不足 | 低 | 中 | HOTOSM POI 密集度高，50m 半径内通常有对应 POI；若失败则降级为 L1 文本匹配 |
| 算力不足，MiniLM 推理过慢 | 中 | 中 | 降级为 CPU 批量推理；或主力用 TF-IDF + Word2Vec，MiniLM 只做一组对比 |
| 四源拼接后特征维度高、信息冗余 | 中 | 中 | 先用 PCA 降维后可视化检查；如某源贡献微小，在消融实验中量化并诚实报告 |
| 联合聚类效果不优于 Baseline | 低 | 中 | 负结果也是结果——在报告中分析原因（文本信号太弱？多源噪声过多？α 选择不当？），学术上完全可接受 |
| 队员工作量不均衡 | 中 | 中 | 第 3 周进行工作量评估，动态调整 |
| Overpass API 速率限制（如使用） | 低 | 低 | 分城市、分时段请求；主要依赖预下载的 HOTOSM GeoJSON，API 只做补充 |

---

## 十、技术栈与工具

| 层级 | 技术选型 |
|------|---------|
| **数据获取** | HOTOSM (GeoJSON)、HuggingFace `datasets`、e-Stat API、ekidata.jp、Overpass API (补充) |
| **数据处理** | pandas、numpy、geopandas、shapely、scikit-learn (MinMaxScaler, OneHotEncoder, PCA) |
| **NLP** | gensim (Word2Vec)、sentence-transformers (MiniLM)、scikit-learn (TfidfVectorizer)、fugashi (日文分词) |
| **聚类算法** | scikit-learn (K-Means, Agglomerative, DBSCAN) |
| **评估与统计** | scikit-learn (Silhouette, DBI, CH-Index)、scipy (Wilcoxon) |
| **可视化** | matplotlib、seaborn、Folium、keplergl (备选) |
| **应用前端** | Streamlit (可选增强) |
| **版本管理** | Git + GitHub |

---

## 十一、首次会议待讨论清单

- [x] **数据覆盖范围确定**：本州岛，重点数据集可以放在关西+东京范围
- [x] **队员角色确认**：4 人各自认领 A/B/C/D 角色，每人均参与 W1 数据下载
- [ ] **HuggingFace 数据验证**：下载 `itinerai/attractions` 和 `ACOSRes`，检查字段完整性和实际数据量
- [ ] **四源数据质量检查**：POI 覆盖率、文本量级、车站经纬度精度、人口统计关联率
- [ ] **NLP 方案优先级**：算力有限时优先保 TF-IDF + Word2Vec 还是直接冲 MiniLM？
- [ ] **爬虫定位确认**：中文评论爬虫仅作可选项——是否启动？如启动，由谁负责？
- [x] **GitHub 仓库名**：统一命名，建立协作规范

---

## 十二、代码目录结构与交付清单

### 12.1 仓库目录结构

```
DAM-Final/
├── README.md                         # 项目说明、环境配置、一键运行指南
├── requirements.txt                  # Python 依赖列表
├── main.py                           # 一键运行入口：python main.py
├── config.py                         # 全局参数（α 范围、K 值、数据路径等）
├── .gitignore
│
├── data/
│   ├── raw/                          # 原始下载数据（不提交 Git，.gitignore 排除）
│   │   ├── pois_raw.geojson          # ① HOTOSM 日本全境 POI
│   │   ├── reviews_raw/              # ② HuggingFace 缓存
│   │   ├── stations_raw.csv          # ③ ekidata.jp 铁路站点
│   │   └── census_raw/               # ④ e-Stat SSDSE 区域统计
│   └── processed/                    # 清洗后数据（提交 Git）
│       ├── pois_clean.csv            # POI 筛选 + 归一化
│       ├── reviews_aligned.csv       # 实体对齐后的评论-POI 映射
│       ├── stations_clean.csv        # 车站 + 衍生可达性特征
│       ├── census_clean.csv          # 市区町村统计 + 特征工程
│       └── feature_matrix.csv        # ★ 最终四源特征宽表
│
├── src/
│   ├── __init__.py
│   ├── data/                         # ── 队员 A 主责 ──
│   │   ├── __init__.py
│   │   ├── download_poi.py           # ① HOTOSM 下载 + 类别筛选
│   │   ├── download_census.py        # ④ e-Stat 下载 + 字段整理
│   │   ├── preprocess.py             # 坐标归一化、类别独热编码、缺失值处理
│   │   ├── entity_alignment.py       # L0 空间匹配 + L1/L2 文本匹配
│   │   └── feature_merge.py          # 四源特征拼接 → feature_matrix.csv
│   │
│   ├── models/                       # ── 队员 B 主责 ──
│   │   ├── __init__.py
│   │   ├── download_reviews.py       # ② HuggingFace 数据加载
│   │   ├── nlp_vectorizers.py        # TF-IDF / Word2Vec / MiniLM 三方案
│   │   ├── distance_metrics.py       # D_joint(α) 联合距离实现
│   │   ├── clustering.py             # K-Means / DBSCAN / Agglomerative
│   │   └── grid_search.py            # α + K + eps 网格搜索
│   │
│   ├── evaluation/                   # ── 队员 C 主责 ──
│   │   ├── __init__.py
│   │   ├── download_transit.py       # ③ ekidata.jp 下载 + 可达性计算
│   │   ├── metrics.py                # Silhouette / DBI / CH-Index 计算
│   │   ├── ablation.py               # 消融实验：2-Src vs 3-Src vs 4-Src
│   │   ├── statistical_tests.py      # Wilcoxon 符号秩检验
│   │   └── case_analysis.py          # 聚类归因 + 业务洞察输出
│   │
│   └── visualization/                # ── 队员 D 主责 ──
│       ├── __init__.py
│       ├── cluster_map.py            # Folium 交互式聚类地图 → outputs/maps/
│       ├── comparison_charts.py      # Silhouette/DBI 对比曲线 → outputs/figures/
│       └── streamlit_app.py          # Streamlit 对比面板（可选增强）
│
├── outputs/                          # 生成物（提交 Git）
│   ├── figures/                      # PNG/SVG 对比图表
│   ├── maps/                         # Folium HTML 交互地图
│   └── results/                      # CSV 实验结果表
│
├── notebooks/                        # 探索性分析（可选）
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_vectorization.ipynb
│   └── 03_clustering_experiments.ipynb
│
├── docs/                             # 文档
│   ├── team_contribution.md          # 团队分工声明（作业第 5 项）
│   └── data_dictionary.md            # 数据字段说明文档
│
├── report/
│   ├── report.pdf                    # 期末报告（10-20 页，作业第 4 项）
│   ├── poster.pdf                    # 项目海报 A1（加分项）
│   └── presentation.pptx             # 答辩 PPT（作业第 6 项）
│
└── ProjDsgn.md                       # 本文件 —— 项目计划书（作业第 1 项）
```

### 12.2 交付文件与作业评分对照

| 作业要求 | 对应文件 | 负责人 | 备注 |
|---------|---------|:------:|------|
| 1. 项目计划书 (5%) | `ProjDsgn.md` | D（提交） | 即本文档 |
| 2. 代码仓库 (10%) | 整个 GitHub 仓库 | D（管理） | 每人独立分支 + PR 合并 |
| 3. 完整代码 (25%) | `main.py` + `src/**/*.py` | A/B/C/D | `python main.py` 一键运行 |
| 4. 项目报告 (30%) | `report/report.pdf` | D（统稿） | A→数据章节，B→方法章节，C→实验+讨论章节，D→引言+结论章节 |
| 5. 团队分工声明 (5%) | `docs/team_contribution.md` | D | 逐条列出每人代码文件 + 报告章节 |
| 6. 答辩 PPT + 演示 (25%) | `report/presentation.pptx` + `outputs/maps/*.html` | D（制作）+ 全员（讲解） | 8 分钟讲解 + 2 分钟地图演示 |
| 7. 项目海报 (加分) | `report/poster.pdf` | D（排版）+ 全员（内容） | A1 尺寸，突出 Pipeline + 聚类效果图 |
| — | `README.md` | D | 项目说明、环境配置、运行方法 |
| — | `requirements.txt` | D | 一键安装：`pip install -r requirements.txt` |

### 12.3 每人交付清单

#### 队员 A —— 数据工程负责人

| 文件 | 位置 | 说明 |
|------|------|------|
| `download_poi.py` | `src/data/` | ① HOTOSM GeoJSON 下载 + POI 类别筛选 |
| `download_census.py` | `src/data/` | ④ e-Stat SSDSE 下载 + 字段整理 |
| `preprocess.py` | `src/data/` | 坐标 Min-Max 归一化、POI 类别独热编码、缺失值处理 |
| `entity_alignment.py` | `src/data/` | L0 空间匹配（<50m）+ L1 名称精确 + L2 模糊匹配 |
| `feature_merge.py` | `src/data/` | 四源特征拼接，输出 `data/processed/feature_matrix.csv` |
| `data_dictionary.md` | `docs/` | 所有字段的中文说明、取值范围、来源 |
| `pois_clean.csv` | `data/processed/` | 清洗后的 POI 表 |
| `feature_matrix.csv` | `data/processed/` | ★ 最终交付物——四源特征宽表 |
| 报告章节 | `report/report.pdf` | "数据与预处理" 章节（数据来源说明、清洗流程、对齐评估、宽表统计） |

#### 队员 B —— 算法建模负责人

| 文件 | 位置 | 说明 |
|------|------|------|
| `download_reviews.py` | `src/models/` | ② HuggingFace `load_dataset` 加载 attractions + ACOSRes |
| `nlp_vectorizers.py` | `src/models/` | TF-IDF / Word2Vec / MiniLM 三种向量化实现 |
| `distance_metrics.py` | `src/models/` | `D_joint(α)` 联合距离函数实现 |
| `clustering.py` | `src/models/` | K-Means / DBSCAN / Agglomerative，支持自定义 metric |
| `grid_search.py` | `src/models/` | α 网格搜索 + K 值/eps 参数寻优 |
| 对比结果表 | `outputs/results/` | 最优参数配置 + 各组 Silhouette/DBI 汇总 |
| 报告章节 | `report/report.pdf` | "方法" 章节（NLP 方案、距离公式推导、聚类算法选型、参数寻优策略） |

#### 队员 C —— 评估与分析负责人

| 文件 | 位置 | 说明 |
|------|------|------|
| `download_transit.py` | `src/evaluation/` | ③ ekidata.jp 下载 + Haversine 最近站距离 + 1km 站点密度 |
| `metrics.py` | `src/evaluation/` | Silhouette / DBI / CH-Index 计算函数 |
| `ablation.py` | `src/evaluation/` | 2-Src vs 3-Src vs 4-Src 特征组合消融实验 |
| `statistical_tests.py` | `src/evaluation/` | Wilcoxon 符号秩检验 |
| `case_analysis.py` | `src/evaluation/` | 选取 Top-5 聚类，还原评论文本做归因解释 |
| 对比图表 | `outputs/figures/` | Silhouette 曲线、DBI 柱状图、消融对比图 |
| 报告章节 | `report/report.pdf` | "实验" 章节（实验设置、结果表格、图表分析）+ "讨论" 章节（案例归因、业务洞察、局限性） |

#### 队员 D —— 工程与交付负责人

| 文件 | 位置 | 说明 |
|------|------|------|
| `config.py` | 项目根目录 | 全局参数集中管理 |
| `main.py` | 项目根目录 | 一键运行入口，串联全部 Pipeline |
| `requirements.txt` | 项目根目录 | 依赖列表，含版本号 |
| `README.md` | 项目根目录 | 项目说明、环境配置、运行步骤、结果截图 |
| `.gitignore` | 项目根目录 | 排除 `data/raw/`、`__pycache__`、`.ipynb_checkpoints` |
| `cluster_map.py` | `src/visualization/` | Folium 交互式地图，聚类标签分色渲染 |
| `comparison_charts.py` | `src/visualization/` | C 的指标数据 → matplotlib/seaborn 出图 |
| `streamlit_app.py` | `src/visualization/` | Streamlit 下拉切换面板（可选） |
| `team_contribution.md` | `docs/` | 团队分工声明 |
| `report.pdf` | `report/` | 统稿 + "引言" "结论" 章节撰写 |
| `presentation.pptx` | `report/` | 答辩 PPT |
| `poster.pdf` | `report/` | A1 海报 |
| HTML 地图文件 | `outputs/maps/` | 可交付的交互式聚类地图 |

### 12.4 第一周各人立即行动清单

```
队员 A（今天）：
  → 下载 HOTOSM Japan POI GeoJSON
  → pandas 转 CSV，筛选 tourism/restaurant/shop/hotel 四类
  → 检查数据量和城市覆盖 → 在群里发截图

队员 B（今天）：
  → pip install datasets sentence-transformers
  → from datasets import load_dataset
  → load_dataset("itinerai/attractions") — 检查字段是否含 lat/lon/reviews
  → load_dataset("LujainAbdulrahman/ACOSRes") — 检查结构和量级
  → 在群里发数据快照

队员 C（今天）：
  → 访问 ekidata.jp → 下载全日本站点 CSV
  → 验证经纬度格式 + 统计站点总数
  → 访问 e-Stat SSDSE → 下载 1-2 个典型市区町村的数据试关联

队员 D（今天）：
  → 创建 GitHub 仓库，设分支保护
  → 按 12.1 目录结构建好空文件夹 + .gitignore
  → 写 requirements.txt（pandas numpy scikit-learn gensim
    sentence-transformers folium streamlit matplotlib seaborn scipy geopandas shapely）
  → 写 config.py 模板（数据路径、α 默认值等）
  → 提交 Initial commit
```

---

> **版本记录**：v3.1 —— 新增第十二章"代码目录结构与交付清单"，明确每人负责的文件、存放路径、交付标准。
