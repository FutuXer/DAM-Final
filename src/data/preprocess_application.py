"""
队员 A — 主表特征工程 Pipeline
处理 application_train.csv / application_test.csv
步骤: A1 缺失值 → A2 异常值 → A3 类别编码 → A4 时间转换 → A5 特征衍生
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# 主表中有实际含义的类别列（非数值编码）
CAT_COLS_NOMINAL = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE", "OCCUPATION_TYPE",
]

# 二值标志列（已编码为 0/1，不需再处理）
BINARY_FLAG_COLS = [
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL", "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
]

# FLAG_DOCUMENT_* 列
DOCUMENT_COLS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]

# DAYS 时间列（需要取绝对值转成年）
DAYS_COLS = [
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
]

# 外部评分列
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# 社交圈违约列
SOCIAL_CIRCLE_COLS = [
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
]

# Credit Bureau 查询次数列
BUREAU_ENQUIRY_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# 建筑相关特征的 suffix 组
BUILDING_SUFFIXES = ["AVG", "MODE", "MEDI"]
BUILDING_BASES = [
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA", "FONDKAPREMONT", "HOUSETYPE", "TOTALAREA",
    "WALLSMATERIAL", "EMERGENCYSTATE",
]


# ============================================================
# A1: 缺失值处理
# ============================================================
def analyze_missing(df):
    """输出每列的缺失率，返回高缺失列列表."""
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"[A1] 共 {len(missing)} 列存在缺失值")
    print(f"[A1] 缺失率 > 50% 的列: {list(missing[missing > 0.5].index)}")
    return missing


def fill_missing(df):
    """按列类型填充缺失值."""
    df = df.copy()
    for col in df.columns:
        if col == "TARGET":
            continue
        missing_rate = df[col].isnull().mean()
        if missing_rate == 0:
            continue
        if missing_rate > 0.60:
            # 高缺失列：保留但标记
            df[f"{col}_MISSING"] = df[col].isnull().astype(np.uint8)

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


# ============================================================
# A2: 异常值处理
# ============================================================
def treat_outliers(df):
    """处理已知异常值和离群点盖帽."""
    df = df.copy()

    # --- DAYS_EMPLOYED: 365243 = 无业 (异常标记值) ---
    mask_unemployed = df["DAYS_EMPLOYED"] == 365243
    df["FLAG_UNEMPLOYED"] = mask_unemployed.astype(np.uint8)
    df.loc[mask_unemployed, "DAYS_EMPLOYED"] = 0

    # --- OWN_CAR_AGE: 异常值截断 (> 60 年截断为 60) ---
    df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].clip(upper=60)

    # --- AMT_INCOME_TOTAL: 上 99% 盖帽 ---
    cap_income = df["AMT_INCOME_TOTAL"].quantile(0.99)
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=cap_income)

    # --- AMT_CREDIT: 上 99% 盖帽 ---
    cap_credit = df["AMT_CREDIT"].quantile(0.99)
    df["AMT_CREDIT"] = df["AMT_CREDIT"].clip(upper=cap_credit)

    # --- AMT_ANNUITY: 上 99% 盖帽 ---
    cap_annuity = df["AMT_ANNUITY"].quantile(0.99)
    df["AMT_ANNUITY"] = df["AMT_ANNUITY"].clip(upper=cap_annuity)

    # --- AMT_GOODS_PRICE: 上 99% 盖帽 ---
    if "AMT_GOODS_PRICE" in df.columns:
        cap_goods = df["AMT_GOODS_PRICE"].quantile(0.99)
        df["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"].clip(upper=cap_goods)

    # --- CNT_CHILDREN: 极端值处理 ---
    df.loc[df["CNT_CHILDREN"] > 10, "CNT_CHILDREN"] = 10

    # --- CNT_FAM_MEMBERS: 极端值处理 ---
    df.loc[df["CNT_FAM_MEMBERS"] > 15, "CNT_FAM_MEMBERS"] = 15

    # --- 人数逻辑修正: CNT_FAM_MEMBERS >= CNT_CHILDREN + 1 ---
    mask = df["CNT_FAM_MEMBERS"] < df["CNT_CHILDREN"] + 1
    df.loc[mask, "CNT_FAM_MEMBERS"] = df.loc[mask, "CNT_CHILDREN"] + 1

    print("[A2] 异常值处理完成: 失业标记、收入/信用盖帽、人数修正")
    return df


# ============================================================
# A3: 类别特征编码
# ============================================================
def target_encode(train, test, cat_cols, target_col="TARGET", n_folds=5):
    """基于 5-Fold 的 Target Encoding，避免数据泄露."""
    train = train.copy()
    test = test.copy()
    for col in cat_cols:
        train[f"{col}_TE"] = np.nan
        global_mean = train[target_col].mean()
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train, train[target_col]):
            fold_mean = train.iloc[tr_idx].groupby(col)[target_col].mean()
            train.loc[train.index[val_idx], f"{col}_TE"] = (
                train.loc[train.index[val_idx], col].map(fold_mean)
            )
        train[f"{col}_TE"] = train[f"{col}_TE"].fillna(global_mean)

        # 对 test 使用全部 train 数据的均值
        full_mean = train.groupby(col)[target_col].mean()
        test[f"{col}_TE"] = test[col].map(full_mean).fillna(global_mean)
    return train, test


def encode_categorical(train, test=None):
    """类别特征编码: 低基数 → One-Hot, 高基数 → Target Encoding.
    自动检测所有 object/category 列进行编码 (不仅是 CAT_COLS_NOMINAL)."""
    train = train.copy()
    if test is not None:
        test = test.copy()

    # 自动收集所有待编码的类别列 (包括衍生阶段产生的 AGE_BIN 等)
    all_cat_cols = list(train.select_dtypes(include=["object", "category"]).columns)
    # 合并预定义列表 + 自动检测
    cols_to_encode = list(dict.fromkeys(CAT_COLS_NOMINAL + all_cat_cols))
    cols_to_encode = [c for c in cols_to_encode if c in train.columns]

    # --- 低基数类别列: One-Hot (≤5 个类别) ---
    low_card_cols = []
    high_card_cols = []
    for col in cols_to_encode:
        n_unique = train[col].nunique()
        if n_unique <= 5:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    train = pd.get_dummies(train, columns=low_card_cols, drop_first=False)
    if test is not None:
        test = pd.get_dummies(test, columns=low_card_cols, drop_first=False)

    # --- 高基数类别列: Target Encoding ---
    if high_card_cols and test is not None and "TARGET" in train.columns:
        train, test = target_encode(train, test, high_card_cols)
    elif high_card_cols:
        for col in high_card_cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            if test is not None:
                known = set(le.classes_)
                test[col] = test[col].apply(lambda x: x if x in known else "Unknown")
                le_classes = list(le.classes_) + ["Unknown"]
                le.classes_ = np.array(le_classes)
                test[col] = le.transform(test[col].astype(str))

    # --- 二值标志列: 确保为 0/1 ---
    for col in BINARY_FLAG_COLS + DOCUMENT_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(0).astype(np.uint8)
            if test is not None and col in test.columns:
                test[col] = test[col].fillna(0).astype(np.uint8)

    print(f"[A3] 编码完成: One-Hot {len(low_card_cols)} 列, Target Encoded {len(high_card_cols)} 列")
    if test is not None:
        return train, test
    return train


# ============================================================
# A4: 时间特征转换
# ============================================================
def convert_days_features(df):
    """将 DAYS_ 列取绝对值除以 365 转为年."""
    df = df.copy()
    for col in DAYS_COLS:
        if col not in df.columns:
            continue
        new_name = col.replace("DAYS_", "").replace("BIRTH", "AGE_YEARS") \
                       .replace("EMPLOYED", "YEARS_EMPLOYED") \
                       .replace("REGISTRATION", "YEARS_REGISTRATION") \
                       .replace("ID_PUBLISH", "YEARS_ID_PUBLISH") \
                       .replace("LAST_PHONE_CHANGE", "YEARS_LAST_PHONE_CHANGE")
        df[new_name] = df[col].abs() / 365.25
        # 保留原始列供后续衍生使用
    print(f"[A4] 时间列转换完成: {len(DAYS_COLS)} 列 → 年")
    return df


# ============================================================
# A5: 特征衍生
# ============================================================
def engineer_features(df):
    """构造领域驱动衍生特征."""
    df = df.copy()

    # --- 5.1 财务比率 ---
    # 收入贷款比 (DTI-like)
    df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)
    # 年金收入比 (负担率)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    # 贷款商品比 (LTV-like)
    if "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
        df["CREDIT_GOODS_RATIO"] = df["CREDIT_GOODS_RATIO"].clip(upper=10)
    # 年金贷款比 (偿还率)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)

    # --- 5.2 外部评分加权组合 ---
    available_ext = [c for c in EXT_SOURCE_COLS if c in df.columns]
    if len(available_ext) >= 2:
        # 按与 TARGET 的相关性加权
        df["EXT_SOURCE_WEIGHTED"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[available_ext].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[available_ext].max(axis=1)
        df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1)
    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
        df["EXT_SOURCE_1x2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]

    # --- 5.3 社交圈违约特征 ---
    df["SOCIAL_DEF_30_RATIO"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    df["SOCIAL_DEF_60_RATIO"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / (df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    df["SOCIAL_HAS_DEFAULT"] = ((df["DEF_30_CNT_SOCIAL_CIRCLE"] > 0) |
                                 (df["DEF_60_CNT_SOCIAL_CIRCLE"] > 0)).astype(np.uint8)

    # --- 5.4 Credit Bureau 查询总次数 ---
    available_bur = [c for c in BUREAU_ENQUIRY_COLS if c in df.columns]
    if available_bur:
        df["BUREAU_ENQUIRY_TOTAL"] = df[available_bur].sum(axis=1)

    # --- 5.5 年龄分箱 ---
    if "AGE_YEARS" in df.columns:
        df["AGE_BIN"] = pd.cut(df["AGE_YEARS"],
                                bins=[0, 25, 35, 45, 55, 65, 120],
                                labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"])
        df["AGE_BIN"] = df["AGE_BIN"].astype(str)  # 后续编码
        df["AGE_YOUNG"] = (df["AGE_YEARS"] < 30).astype(np.uint8)
        df["AGE_OLD"] = (df["AGE_YEARS"] > 60).astype(np.uint8)

    # --- 5.6 收入分箱 ---
    df["INCOME_BIN"] = pd.cut(df["AMT_INCOME_TOTAL"],
                               bins=[0, 90000, 135000, 180000, 225000, 500000, 1e9],
                               labels=["very_low", "low", "medium", "high", "very_high", "extreme"])
    df["INCOME_BIN"] = df["INCOME_BIN"].astype(str)

    # --- 5.7 工作时间相关 ---
    if "YEARS_EMPLOYED" in df.columns and "AGE_YEARS" in df.columns:
        # 工作经验占人生比例
        df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / (df["AGE_YEARS"] + 1)
        df["EMPLOYED_AGE_RATIO"] = df["EMPLOYED_AGE_RATIO"].clip(upper=1.5)
        # 新入职标记 (< 2 年)
        df["FLAG_NEW_EMPLOYEE"] = (df["YEARS_EMPLOYED"] < 2).astype(np.uint8)

    # --- 5.8 家庭特征 ---
    if "CNT_CHILDREN" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)
        df["ADULTS_COUNT"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
        df["ADULTS_COUNT"] = df["ADULTS_COUNT"].clip(lower=1)

    # --- 5.9 人均指标 ---
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / (df["CNT_FAM_MEMBERS"] + 1)

    # --- 5.10 建筑特征均值 ---
    for base in BUILDING_BASES:
        cols = [f"{base}_{s}" for s in BUILDING_SUFFIXES if f"{base}_{s}" in df.columns]
        if len(cols) >= 2:
            df[f"{base}_MEAN_CROSS"] = df[cols].mean(axis=1)

    # --- 5.11 提供文件总数 ---
    available_docs = [c for c in DOCUMENT_COLS if c in df.columns]
    if available_docs:
        df["DOCUMENTS_PROVIDED"] = df[available_docs].sum(axis=1)
        df["DOCUMENTS_LOW"] = (df["DOCUMENTS_PROVIDED"] <= 1).astype(np.uint8)

    # --- 5.12 地区评分组合 ---
    if "REGION_RATING_CLIENT" in df.columns and "REGION_RATING_CLIENT_W_CITY" in df.columns:
        df["REGION_RATING_COMBO"] = (
            df["REGION_RATING_CLIENT"].fillna(2) * 0.6 +
            df["REGION_RATING_CLIENT_W_CITY"].fillna(2) * 0.4
        )

    # --- 5.13 车辆相关 ---
    if "OWN_CAR_AGE" in df.columns:
        df["CAR_AGE_BIN"] = pd.cut(df["OWN_CAR_AGE"].fillna(-1),
                                    bins=[-2, 0, 5, 10, 15, 20, 100],
                                    labels=["no_car", "<5", "5-10", "10-15", "15-20", "20+"])
        df["CAR_AGE_BIN"] = df["CAR_AGE_BIN"].astype(str)

    # --- 5.14 注册/身份证时间差 ---
    if "YEARS_REGISTRATION" in df.columns and "YEARS_ID_PUBLISH" in df.columns:
        df["REGISTRATION_ID_GAP"] = (
            df["YEARS_REGISTRATION"] - df["YEARS_ID_PUBLISH"]
        ).abs()
        df["REGISTRATION_RECENT"] = (df["YEARS_REGISTRATION"] < 2).astype(np.uint8)

    print(f"[A5] 特征衍生完成，当前维度: {df.shape[1]}")
    return df


# ============================================================
# A7: 数值标准化
# ============================================================
def standardize_numerical(train, test=None):
    """对连续数值列做 Z-score 标准化，排除二值/ID 列."""
    train = train.copy()
    if test is not None:
        test = test.copy()

    exclude_cols = {"SK_ID_CURR", "TARGET"}
    # 收集所有二值列（0/1 标志列和 One-Hot 列）
    binary_cols = set()
    for col in train.columns:
        if col in exclude_cols:
            continue
        if train[col].nunique() <= 2:
            # 确认确实是 0/1 二值
            vals = train[col].dropna().unique()
            if set(vals).issubset({0, 1, 0.0, 1.0}):
                binary_cols.add(col)

    scale_cols = [c for c in train.columns
                  if c not in exclude_cols and c not in binary_cols
                  and train[c].dtype in ["int64", "float64", "int32", "float32"]]

    scaler = StandardScaler()
    train[scale_cols] = scaler.fit_transform(train[scale_cols])
    if test is not None:
        test[scale_cols] = scaler.transform(test[scale_cols])

    print(f"[A7] 标准化完成: {len(scale_cols)} 个数值列已 Z-score 标准化")
    if test is not None:
        return train, test
    return train


# ============================================================
# 主 Pipeline
# ============================================================
def process_application_table(train_path, test_path=None, output_dir=PROCESSED_DIR):
    """
    完整处理 application 主表.
    参数:
        train_path: application_train.csv 路径
        test_path: application_test.csv 路径 (可选)
        output_dir: 输出目录
    返回:
        train_processed, test_processed (或仅 train)
    """
    print("=" * 60)
    print("队员 A — 主表特征工程 Pipeline")
    print("=" * 60)

    # 加载
    print("\n[加载] 读取数据...")
    train = pd.read_csv(train_path)
    print(f"  Train: {train.shape}")

    test = None
    if test_path:
        test = pd.read_csv(test_path)
        print(f"  Test:  {test.shape}")

    # 保存 SK_ID_CURR 用于最终 join
    train_id = train["SK_ID_CURR"].copy()
    test_id = test["SK_ID_CURR"].copy() if test is not None else None

    # 分离 TARGET
    if "TARGET" in train.columns:
        y_train = train["TARGET"].copy()
        train = train.drop(columns=["TARGET"])
    else:
        y_train = None

    # --- A1: 缺失值 ---
    print("\n" + "-" * 40)
    print("[A1] 缺失值处理")
    analyze_missing(train)
    train = fill_missing(train)
    if test is not None:
        test = fill_missing(test)

    # --- A2: 异常值 ---
    print("\n" + "-" * 40)
    train = treat_outliers(train)
    if test is not None:
        for col in train.columns:
            if col not in test.columns:
                continue

    # 对 test 也做基本的异常值处理
    if test is not None:
        test["FLAG_UNEMPLOYED"] = (test["DAYS_EMPLOYED"] == 365243).astype(np.uint8)
        test.loc[test["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = 0

    # --- A4: 时间特征 (在编码之前做，因为 AGE_BIN 等衍生依赖它) ---
    print("\n" + "-" * 40)
    train = convert_days_features(train)
    if test is not None:
        test = convert_days_features(test)

    # --- A5: 特征衍生 ---
    print("\n" + "-" * 40)
    train = engineer_features(train)
    if test is not None:
        test = engineer_features(test)

    # --- A3: 类别编码 (放在衍生之后，因为 AGE_BIN/INCOME_BIN 等需要编码) ---
    print("\n" + "-" * 40)
    # 将新增的类别衍生列也加入编码列表
    extra_cat = []
    for col in ["AGE_BIN", "INCOME_BIN", "CAR_AGE_BIN"]:
        if col in train.columns:
            extra_cat.append(col)
    cat_cols_all = CAT_COLS_NOMINAL + extra_cat
    # 用 DataFrame 临时存储类别列配置
    CAT_COLS_NOMINAL_EXT = [c for c in cat_cols_all if c in train.columns]

    # 调用编码
    if test is not None:
        train, test = encode_categorical_with_test(train, test, CAT_COLS_NOMINAL_EXT, y_train)
    else:
        # 分组编码
        low_card, high_card = [], []
        for col in CAT_COLS_NOMINAL_EXT:
            if train[col].nunique() <= 5:
                low_card.append(col)
            else:
                high_card.append(col)
        train = pd.get_dummies(train, columns=low_card, drop_first=False)
        for col in high_card:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))

    # 清理: 移除可能产生问题的列
    cols_to_drop = ["SK_ID_CURR"]  # 保留到最后
    # 移除原始类别列 (已被编码)
    for col in CAT_COLS_NOMINAL_EXT:
        if col in train.columns:
            cols_to_drop.append(col)

    # 确保没有非数值列
    obj_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        if col not in cols_to_drop and col != "SK_ID_CURR":
            cols_to_drop.append(col)

    train_clean = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
    if "SK_ID_CURR" not in train_clean.columns:
        train_clean["SK_ID_CURR"] = train_id.values

    if test is not None:
        test_clean = test.drop(columns=[c for c in cols_to_drop if c in test.columns])
        if "SK_ID_CURR" not in test_clean.columns:
            test_clean["SK_ID_CURR"] = test_id.values

    # 确保 train 和 test 列一致
    if test is not None:
        common_cols = list(set(train_clean.columns) & set(test_clean.columns))
        # SK_ID_CURR must be in common
        if "SK_ID_CURR" in train_clean.columns and "SK_ID_CURR" in test_clean.columns:
            common_cols = list(set(common_cols + ["SK_ID_CURR"]))
        train_clean = train_clean[[c for c in common_cols if c in train_clean.columns]]
        test_clean = test_clean[[c for c in common_cols if c in test_clean.columns]]

    # --- A7: 数值标准化 ---
    print("\n" + "-" * 40)
    if test is not None:
        train_clean, test_clean = standardize_numerical(train_clean, test_clean)
    else:
        train_clean = standardize_numerical(train_clean)

    # 保存
    train_path_out = f"{output_dir}/processed_application_train.csv"
    train_clean.to_csv(train_path_out, index=False)
    print(f"\n[保存] Train → {train_path_out} ({train_clean.shape})")

    if test is not None:
        test_path_out = f"{output_dir}/processed_application_test.csv"
        test_clean.to_csv(test_path_out, index=False)
        print(f"[保存] Test  → {test_path_out} ({test_clean.shape})")

    print("\n" + "=" * 60)
    print("队员 A Pipeline 完成!")
    print("=" * 60)

    if test is not None:
        return train_clean, test_clean
    return train_clean


def encode_categorical_with_test(train, test, cat_cols, y_train):
    """对 train/test 同时做类别编码."""
    train = train.copy()
    test = test.copy()

    low_card, high_card = [], []
    for col in cat_cols:
        if col not in train.columns:
            continue
        if train[col].nunique() <= 5:
            low_card.append(col)
        else:
            high_card.append(col)

    # One-Hot
    train = pd.get_dummies(train, columns=low_card, drop_first=False)
    test = pd.get_dummies(test, columns=low_card, drop_first=False)

    # Target Encoding for high cardinality
    if high_card and y_train is not None:
        for col in high_card:
            train[f"{col}_TE"] = np.nan
            global_mean = y_train.mean()
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, val_idx in kf.split(train, y_train):
                fold_mean = y_train.iloc[tr_idx].groupby(train.iloc[tr_idx][col]).mean()
                train.loc[train.index[val_idx], f"{col}_TE"] = (
                    train.loc[train.index[val_idx], col].map(fold_mean)
                )
            train[f"{col}_TE"] = train[f"{col}_TE"].fillna(global_mean)
            full_mean = y_train.groupby(train[col]).mean()
            test[f"{col}_TE"] = test[col].map(full_mean).fillna(global_mean)
    elif high_card:
        for col in high_card:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            test[col] = test[col].apply(lambda x: x if x in set(le.classes_) else "Unknown")
            le_classes = list(le.classes_) + ["Unknown"]
            le.classes_ = np.array(le_classes)
            test[col] = le.transform(test[col].astype(str))

    return train, test


def merge_with_features(train_processed, feature_files, output_dir=PROCESSED_DIR):
    """
    A6: 合并队员 B、C 产出的特征表.
    feature_files: list of (file_path, member_label)
    """
    df = pd.read_csv(train_processed) if isinstance(train_processed, str) else train_processed.copy()

    for fpath, label in feature_files:
        feats = pd.read_csv(fpath)
        print(f"[A6] 合并 {label}: {fpath} → {feats.shape[1]-1} 个特征")
        df = df.merge(feats, on="SK_ID_CURR", how="left")

    # 填充合并产生的新缺失值
    new_null_cols = df.columns[df.isnull().any()]
    for col in new_null_cols:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    out_path = f"{output_dir}/final_feature_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"[A6] 最终宽表已保存: {out_path} ({df.shape})")
    return df


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    import sys
    train_file = sys.argv[1] if len(sys.argv) > 1 else f"{RAW_DIR}/application_train.csv"
    test_file = sys.argv[2] if len(sys.argv) > 2 else f"{RAW_DIR}/application_test.csv"

    result = process_application_table(train_file, test_file)
    if isinstance(result, tuple):
        train_proc, test_proc = result
    else:
        train_proc = result
