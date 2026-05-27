"""
队员 B — bureau 表特征工程 Pipeline
处理 bureau.csv + bureau_balance.csv
输出: 按 SK_ID_CURR 聚合的特征矩阵
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 配置
# ============================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

CAT_COLS = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]

DAYS_COLS = [
    "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
]

AMT_COLS = [
    "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
]


# ============================================================
# B1: 缺失值处理
# ============================================================
def fill_missing_bureau(df):
    df = df.copy()
    for col in df.columns:
        missing_rate = df[col].isnull().mean()
        if missing_rate == 0:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


# ============================================================
# B2: 时间列转换
# ============================================================
def convert_days_bureau(df):
    df = df.copy()
    for col in DAYS_COLS:
        if col not in df.columns:
            continue
        new_name = "YEARS_" + col[5:]
        df[new_name] = df[col].abs() / 365.25
    return df


# ============================================================
# B3: 类别编码
# ============================================================
def encode_bureau(df, test=None):
    df = df.copy()
    cat_present = [c for c in CAT_COLS if c in df.columns]
    for col in cat_present:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        if test is not None and col in test.columns:
            test = test.copy()
            known = set(le.classes_)
            test[col] = test[col].apply(lambda x: x if x in known else "Unknown")
            le_classes = list(le.classes_) + ["Unknown"]
            le.classes_ = np.array(le_classes)
            test[col] = le.transform(test[col].astype(str))
    if test is not None:
        return df, test
    return df


# ============================================================
# B4: 特征衍生 (行级)
# ============================================================
def engineer_bureau_features(df):
    df = df.copy()

    # --- 负债率: debt / credit_sum ---
    df["BUREAU_DEBT_CREDIT_RATIO"] = df["AMT_CREDIT_SUM_DEBT"] / (df["AMT_CREDIT_SUM"] + 1)
    df["BUREAU_DEBT_CREDIT_RATIO"] = df["BUREAU_DEBT_CREDIT_RATIO"].clip(upper=5)

    # --- 逾期金额占比 ---
    df["BUREAU_OVERDUE_DEBT_RATIO"] = df["AMT_CREDIT_SUM_OVERDUE"] / (df["AMT_CREDIT_SUM_DEBT"] + 1)
    df["BUREAU_OVERDUE_DEBT_RATIO"] = df["BUREAU_OVERDUE_DEBT_RATIO"].clip(upper=5)

    # --- 信用卡使用率: debt / limit ---
    df["BUREAU_CREDIT_CARD_USAGE"] = df["AMT_CREDIT_SUM_DEBT"] / (df["AMT_CREDIT_SUM_LIMIT"] + 1)
    df["BUREAU_CREDIT_CARD_USAGE"] = df["BUREAU_CREDIT_CARD_USAGE"].clip(upper=5)

    # --- 年金还款占比 ---
    df["BUREAU_ANNUITY_DEBT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT_SUM_DEBT"] + 1)
    df["BUREAU_ANNUITY_DEBT_RATIO"] = df["BUREAU_ANNUITY_DEBT_RATIO"].clip(upper=5)

    # --- 是否逾期标记 ---
    df["BUREAU_HAS_OVERDUE"] = (df["CREDIT_DAY_OVERDUE"] > 0).astype(np.uint8)
    df["BUREAU_HAS_MAX_OVERDUE"] = (df["AMT_CREDIT_MAX_OVERDUE"] > 0).astype(np.uint8)

    # --- 贷款期限 (年) ---
    if "YEARS_CREDIT_ENDDATE" in df.columns and "YEARS_CREDIT" in df.columns:
        df["BUREAU_CREDIT_DURATION"] = (
            df["YEARS_CREDIT_ENDDATE"] - df["YEARS_CREDIT"]
        ).clip(lower=0)

    # --- 是否超期关闭 ---
    if "YEARS_ENDDATE_FACT" in df.columns and "YEARS_CREDIT_ENDDATE" in df.columns:
        df["BUREAU_ENDDATE_DIFF"] = (
            df["YEARS_ENDDATE_FACT"] - df["YEARS_CREDIT_ENDDATE"]
        )

    print(f"[B4] bureau 行级衍生完成，维度: {df.shape[1]}")
    return df


# ============================================================
# B5: bureau_balance 聚合特征
# ============================================================
def process_balance(df_balance):
    """从 bureau_balance 长表聚合出特征."""
    print(f"[B5] 处理 bureau_balance: {df_balance.shape}")

    STATUS_ORDER = {str(i): i for i in range(6)}
    STATUS_ORDER["C"] = 0
    STATUS_ORDER["X"] = np.nan

    df = df_balance.copy()
    df["STATUS_NUM"] = df["STATUS"].map(STATUS_ORDER).fillna(0)

    agg = df.groupby("SK_ID_BUREAU").agg(
        MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        STATUS_MEAN=("STATUS_NUM", "mean"),
        STATUS_MAX=("STATUS_NUM", "max"),
        STATUS_SUM=("STATUS_NUM", "sum"),
        MONTHS_RANGE=("MONTHS_BALANCE", lambda x: x.max() - x.min()),
        MONTHS_RECENT=("MONTHS_BALANCE", "max"),
    ).reset_index()

    agg["BALANCE_HAS_HIGH_DPD"] = (agg["STATUS_MAX"] >= 3).astype(np.uint8)
    agg["BALANCE_HAS_ANY_DPD"] = (agg["STATUS_MAX"] >= 1).astype(np.uint8)
    agg["BALANCE_AVG_STATUS"] = agg["STATUS_MEAN"]

    # DPD 等级计数
    for i in range(6):
        col = f"STATUS_{i}_CNT"
        cnt = df[df["STATUS"] == str(i)].groupby("SK_ID_BUREAU").size().reset_index(name=col)
        agg = agg.merge(cnt, on="SK_ID_BUREAU", how="left")
        agg[col] = agg[col].fillna(0).astype(int)

    agg.drop(columns=["STATUS_MEAN", "STATUS_MAX", "STATUS_SUM"], inplace=True)
    print(f"[B5] bureau_balance 聚合完成: {agg.shape}")
    return agg


# ============================================================
# B6: SK_ID_CURR 级聚合
# ============================================================
def aggregate_to_client(df):
    """将 bureau 行级数据聚合到 SK_ID_CURR 级别."""
    print(f"[B6] 聚合到客户级，输入: {df.shape}")

    agg = df.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("BUREAU_FLAG_ACTIVE", "sum"),
        BUREAU_OVERDUE_COUNT=("BUREAU_HAS_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE_COUNT=("BUREAU_HAS_MAX_OVERDUE", "sum"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN=("AMT_CREDIT_MAX_OVERDUE", "mean"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_MIN=("AMT_CREDIT_SUM", "min"),
        BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT_MAX=("AMT_CREDIT_SUM_DEBT", "max"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_CNT_CREDIT_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_DEBT_CREDIT_RATIO_MEAN=("BUREAU_DEBT_CREDIT_RATIO", "mean"),
        BUREAU_DEBT_CREDIT_RATIO_MAX=("BUREAU_DEBT_CREDIT_RATIO", "max"),
        BUREAU_OVERDUE_DEBT_RATIO_MEAN=("BUREAU_OVERDUE_DEBT_RATIO", "mean"),
        BUREAU_CREDIT_CARD_USAGE_MEAN=("BUREAU_CREDIT_CARD_USAGE", "mean"),
        BUREAU_ANNUITY_DEBT_RATIO_MEAN=("BUREAU_ANNUITY_DEBT_RATIO", "mean"),
        BUREAU_CREDIT_DURATION_MEAN=("BUREAU_CREDIT_DURATION", "mean"),
        BUREAU_CREDIT_DURATION_MAX=("BUREAU_CREDIT_DURATION", "max"),
        BUREAU_YEARS_CREDIT_MIN=("YEARS_CREDIT", "min"),
        BUREAU_YEARS_CREDIT_MEAN=("YEARS_CREDIT", "mean"),
        BUREAU_YEARS_CREDIT_UPDATE_MEAN=("YEARS_CREDIT_UPDATE", "mean"),
        # bureau_balance 派生特征聚合
        BUREAU_BALANCE_MONTHS_MEAN=("MONTHS_COUNT", "mean"),
        BUREAU_BALANCE_MONTHS_SUM=("MONTHS_COUNT", "sum"),
        BUREAU_BALANCE_MONTHS_RANGE_MEAN=("MONTHS_RANGE", "mean"),
        BUREAU_BALANCE_MONTHS_RECENT_MAX=("MONTHS_RECENT", "max"),
        BUREAU_BALANCE_HAS_HIGH_DPD_SUM=("BALANCE_HAS_HIGH_DPD", "sum"),
        BUREAU_BALANCE_HAS_ANY_DPD_SUM=("BALANCE_HAS_ANY_DPD", "sum"),
        BUREAU_BALANCE_AVG_STATUS_MEAN=("BALANCE_AVG_STATUS", "mean"),
        BUREAU_BALANCE_STATUS_0_SUM=("STATUS_0_CNT", "sum"),
        BUREAU_BALANCE_STATUS_1_SUM=("STATUS_1_CNT", "sum"),
        BUREAU_BALANCE_STATUS_2_SUM=("STATUS_2_CNT", "sum"),
        BUREAU_BALANCE_STATUS_3_SUM=("STATUS_3_CNT", "sum"),
        BUREAU_BALANCE_STATUS_4_SUM=("STATUS_4_CNT", "sum"),
        BUREAU_BALANCE_STATUS_5_SUM=("STATUS_5_CNT", "sum"),
    ).reset_index()

    # 逾期比例
    agg["BUREAU_OVERDUE_RATIO"] = agg["BUREAU_OVERDUE_COUNT"] / (agg["BUREAU_COUNT"] + 1)
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / (agg["BUREAU_COUNT"] + 1)
    agg["BUREAU_MAX_OVERDUE_RATIO"] = agg["BUREAU_MAX_OVERDUE_COUNT"] / (agg["BUREAU_COUNT"] + 1)

    # total debt / total credit
    agg["BUREAU_TOTAL_DEBT_CREDIT_RATIO"] = (
        agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / (agg["BUREAU_AMT_CREDIT_SUM_SUM"] + 1)
    )

    # 填充聚合产生的缺失值
    for col in agg.columns:
        if col == "SK_ID_CURR":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[B6] 聚合完成: {agg.shape}")
    return agg



# ============================================================
# 主 Pipeline
# ============================================================
def process_bureau_table(
    bureau_path=f"{RAW_DIR}/bureau.csv",
    balance_path=f"{RAW_DIR}/bureau_balance.csv",
    output_dir=PROCESSED_DIR,
    use_balance=True,
):
    print("=" * 60)
    print("队员 B — bureau 表特征工程 Pipeline")
    print("=" * 60)

    # 加载
    print("\n[加载] 读取数据...")
    bureau = pd.read_csv(bureau_path)
    print(f"  bureau: {bureau.shape}")

    # --- B1: 缺失值 ---
    print("\n" + "-" * 40)
    print("[B1] 缺失值处理")
    bureau = fill_missing_bureau(bureau)

    # --- B2: 时间列转换 ---
    print("\n" + "-" * 40)
    bureau = convert_days_bureau(bureau)

    # --- B3: 类别编码 (编码前标记活跃信用，保留原始字符串语义) ---
    bureau["BUREAU_FLAG_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.uint8)
    print("\n" + "-" * 40)
    bureau = encode_bureau(bureau)

    # --- B4: 特征衍生 ---
    print("\n" + "-" * 40)
    bureau = engineer_bureau_features(bureau)

    # --- B5: bureau_balance ---
    if use_balance:
        print("\n" + "-" * 40)
        balance = pd.read_csv(balance_path)
        print(f"  bureau_balance: {balance.shape}")
        balance_agg = process_balance(balance)
        bureau = bureau.merge(balance_agg, on="SK_ID_BUREAU", how="left")
        cols_to_fill = [c for c in balance_agg.columns if c not in ("SK_ID_BUREAU", "SK_ID_CURR")]
        for c in cols_to_fill:
            if c in bureau.columns:
                bureau[c] = bureau[c].fillna(0)
        print(f"  bureau after merge: {bureau.shape}")

    # --- B6: 聚合到 SK_ID_CURR ---
    print("\n" + "-" * 40)
    client_features = aggregate_to_client(bureau)

    # 保存 (不在此时标准化，统一交给队员 D 处理)
    out_path = f"{output_dir}/features_bureau.csv"
    client_features.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({client_features.shape})")
    print("\n" + "=" * 60)
    print("bureau Pipeline 完成!")
    print("=" * 60)

    return client_features


if __name__ == "__main__":
    process_bureau_table()
