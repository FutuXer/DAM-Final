"""
队员 C — credit_card_balance 表特征工程 Pipeline
处理 credit_card_balance.csv (月度信用卡余额快照)
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

CAT_COLS = ["NAME_CONTRACT_STATUS"]

AMT_COLS = [
    "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE",
]

CNT_COLS = [
    "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
]

DPD_COLS = ["SK_DPD", "SK_DPD_DEF"]

RECENT_MONTHS_THRESHOLD = -6


# ============================================================
# C1: 数据类型修正 + 缺失值
# ============================================================
def clean_and_fill_cc(df):
    df = df.copy()

    for col in CNT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in df.columns:
        if col in ["SK_ID_PREV", "SK_ID_CURR"]:
            continue
        missing = df[col].isnull().mean()
        if missing == 0:
            continue
        if df[col].dtype == "object":
            modes = df[col].mode()
            df[col] = df[col].fillna(modes.iloc[0] if len(modes) > 0 else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    print(f"[C1] 清洗完成: {df.shape}")
    return df


# ============================================================
# C2: 类别编码
# ============================================================
def encode_cc(df):
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    print(f"[C2] 类别编码完成: {CAT_COLS}")
    return df


# ============================================================
# C3: 行级特征衍生
# ============================================================
def engineer_cc_features(df):
    df = df.copy()

    # --- 信用卡额度使用率 ---
    df["CC_LIMIT_USAGE"] = df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    df["CC_LIMIT_USAGE"] = df["CC_LIMIT_USAGE"].clip(upper=5)

    # --- 还款覆盖余额比 ---
    df["CC_PAYMENT_BALANCE_RATIO"] = (
        df["AMT_PAYMENT_TOTAL_CURRENT"] / (df["AMT_BALANCE"] + 1)
    ).clip(upper=10)

    # --- 应收本金占比 ---
    df["CC_PRINCIPAL_RECEIVABLE_RATIO"] = (
        df["AMT_RECEIVABLE_PRINCIPAL"] / (df["AMT_TOTAL_RECEIVABLE"] + 1)
    ).clip(upper=5)

    # --- 最小分期还款占比 ---
    df["CC_INSTALLMENT_BALANCE_RATIO"] = (
        df["AMT_INST_MIN_REGULARITY"] / (df["AMT_BALANCE"] + 1)
    ).clip(upper=5)

    # --- 总提现次数 ---
    df["CC_DRAWINGS_TOTAL_CNT"] = (
        df["CNT_DRAWINGS_ATM_CURRENT"] + df["CNT_DRAWINGS_CURRENT"] +
        df["CNT_DRAWINGS_OTHER_CURRENT"] + df["CNT_DRAWINGS_POS_CURRENT"]
    )

    # --- 总提现金额 ---
    df["CC_DRAWINGS_TOTAL_AMT"] = (
        df["AMT_DRAWINGS_ATM_CURRENT"] + df["AMT_DRAWINGS_CURRENT"] +
        df["AMT_DRAWINGS_OTHER_CURRENT"] + df["AMT_DRAWINGS_POS_CURRENT"]
    )

    # --- 平均单笔提现金额 ---
    df["CC_AVG_DRAWING_AMT"] = (
        df["CC_DRAWINGS_TOTAL_AMT"] / (df["CC_DRAWINGS_TOTAL_CNT"] + 1)
    ).clip(upper=1e6)

    # --- POS 消费占比 ---
    df["CC_POS_RATIO"] = (
        df["AMT_DRAWINGS_POS_CURRENT"] / (df["CC_DRAWINGS_TOTAL_AMT"] + 1)
    )

    # --- ATM 取现占比 ---
    df["CC_ATM_RATIO"] = (
        df["AMT_DRAWINGS_ATM_CURRENT"] / (df["CC_DRAWINGS_TOTAL_AMT"] + 1)
    )

    # --- 逾期标记 ---
    df["CC_HAS_DPD"] = (df["SK_DPD"] > 0).astype(np.uint8)
    df["CC_HAS_DEF_DPD"] = (df["SK_DPD_DEF"] > 0).astype(np.uint8)
    df["CC_DPD_OVER_30"] = (df["SK_DPD"] >= 30).astype(np.uint8)
    df["CC_DPD_OVER_60"] = (df["SK_DPD"] >= 60).astype(np.uint8)

    print(f"[C3] 行级衍生完成，维度: {df.shape[1]}")
    return df


# ============================================================
# C4: 聚合到 SK_ID_CURR
# ============================================================
def aggregate_cc_to_client(df):
    print(f"[C4] 聚合到客户级，输入: {df.shape}")

    # 全量聚合
    agg_all = df.groupby("SK_ID_CURR").agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_AMT_BALANCE_MIN=("AMT_BALANCE", "min"),
        CC_AMT_BALANCE_SUM=("AMT_BALANCE", "sum"),
        CC_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_CREDIT_LIMIT_MAX=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        CC_CREDIT_LIMIT_MIN=("AMT_CREDIT_LIMIT_ACTUAL", "min"),
        CC_LIMIT_USAGE_MEAN=("CC_LIMIT_USAGE", "mean"),
        CC_LIMIT_USAGE_MAX=("CC_LIMIT_USAGE", "max"),
        CC_LIMIT_USAGE_MIN=("CC_LIMIT_USAGE", "min"),
        CC_PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
        CC_PAYMENT_TOTAL_SUM=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
        CC_PAYMENT_TOTAL_MAX=("AMT_PAYMENT_TOTAL_CURRENT", "max"),
        CC_PAYMENT_BALANCE_RATIO_MEAN=("CC_PAYMENT_BALANCE_RATIO", "mean"),
        CC_RECEIVABLE_TOTAL_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
        CC_RECEIVABLE_TOTAL_MAX=("AMT_TOTAL_RECEIVABLE", "max"),
        CC_PRINCIPAL_RATIO_MEAN=("CC_PRINCIPAL_RECEIVABLE_RATIO", "mean"),
        CC_DRAWINGS_TOTAL_CNT_SUM=("CC_DRAWINGS_TOTAL_CNT", "sum"),
        CC_DRAWINGS_TOTAL_CNT_MEAN=("CC_DRAWINGS_TOTAL_CNT", "mean"),
        CC_DRAWINGS_TOTAL_CNT_MAX=("CC_DRAWINGS_TOTAL_CNT", "max"),
        CC_DRAWINGS_TOTAL_AMT_SUM=("CC_DRAWINGS_TOTAL_AMT", "sum"),
        CC_DRAWINGS_TOTAL_AMT_MEAN=("CC_DRAWINGS_TOTAL_AMT", "mean"),
        CC_DRAWINGS_TOTAL_AMT_MAX=("CC_DRAWINGS_TOTAL_AMT", "max"),
        CC_AVG_DRAWING_AMT_MEAN=("CC_AVG_DRAWING_AMT", "mean"),
        CC_POS_RATIO_MEAN=("CC_POS_RATIO", "mean"),
        CC_ATM_RATIO_MEAN=("CC_ATM_RATIO", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_SUM=("SK_DPD", "sum"),
        CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        CC_HAS_DPD_RATIO=("CC_HAS_DPD", "mean"),
        CC_HAS_DEF_DPD_RATIO=("CC_HAS_DEF_DPD", "mean"),
        CC_DPD_OVER_30_RATIO=("CC_DPD_OVER_30", "mean"),
        CC_DPD_OVER_60_RATIO=("CC_DPD_OVER_60", "mean"),
        CC_INST_MIN_REGULARITY_MEAN=("AMT_INST_MIN_REGULARITY", "mean"),
        CC_INSTALMENT_MATURE_CUM_MAX=("CNT_INSTALMENT_MATURE_CUM", "max"),
        CC_INSTALMENT_MATURE_CUM_MEAN=("CNT_INSTALMENT_MATURE_CUM", "mean"),
        CC_INSTALLMENT_BALANCE_RATIO_MEAN=("CC_INSTALLMENT_BALANCE_RATIO", "mean"),
    ).reset_index()

    # 近期行为聚合 (最近 N 个月)
    recent_mask = df["MONTHS_BALANCE"] >= RECENT_MONTHS_THRESHOLD
    df_recent = df[recent_mask]

    if len(df_recent) > 0:
        agg_recent = df_recent.groupby("SK_ID_CURR").agg(
            CC_RECENT_LIMIT_USAGE_MEAN=("CC_LIMIT_USAGE", "mean"),
            CC_RECENT_LIMIT_USAGE_MAX=("CC_LIMIT_USAGE", "max"),
            CC_RECENT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
            CC_RECENT_PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            CC_RECENT_DRAWINGS_CNT_MEAN=("CC_DRAWINGS_TOTAL_CNT", "mean"),
            CC_RECENT_DPD_MEAN=("SK_DPD", "mean"),
            CC_RECENT_DPD_MAX=("SK_DPD", "max"),
            CC_RECENT_MONTHS=("MONTHS_BALANCE", "count"),
        ).reset_index()

        agg_all = agg_all.merge(agg_recent, on="SK_ID_CURR", how="left")

    # 每月行为趋势 (用 merge 避免 .values 对齐风险)
    first_idx = df.groupby("SK_ID_CURR")["MONTHS_BALANCE"].idxmax()
    last_idx = df.groupby("SK_ID_CURR")["MONTHS_BALANCE"].idxmin()

    first_month = df.loc[first_idx, ["SK_ID_CURR", "AMT_BALANCE", "CC_LIMIT_USAGE", "SK_DPD"]].copy()
    first_month.columns = ["SK_ID_CURR", "CC_FIRST_BALANCE", "CC_FIRST_LIMIT_USAGE", "CC_FIRST_DPD"]

    last_month = df.loc[last_idx, ["SK_ID_CURR", "AMT_BALANCE", "CC_LIMIT_USAGE", "SK_DPD"]].copy()
    last_month.columns = ["SK_ID_CURR", "CC_LAST_BALANCE", "CC_LAST_LIMIT_USAGE", "CC_LAST_DPD"]

    trend = first_month.merge(last_month, on="SK_ID_CURR", how="left")
    trend["CC_TREND_BALANCE"] = (
        trend["CC_LAST_BALANCE"] - trend["CC_FIRST_BALANCE"]
    ) / (trend["CC_FIRST_BALANCE"].abs() + 1)
    trend["CC_TREND_LIMIT_USAGE"] = trend["CC_LAST_LIMIT_USAGE"] - trend["CC_FIRST_LIMIT_USAGE"]
    trend["CC_TREND_DPD"] = trend["CC_LAST_DPD"] - trend["CC_FIRST_DPD"]
    trend = trend[["SK_ID_CURR", "CC_TREND_BALANCE", "CC_TREND_LIMIT_USAGE", "CC_TREND_DPD"]]

    agg_all = agg_all.merge(trend, on="SK_ID_CURR", how="left")

    # 衍生比率
    agg_all["CC_LIMIT_USAGE_SPREAD"] = (
        agg_all["CC_LIMIT_USAGE_MAX"] - agg_all["CC_LIMIT_USAGE_MIN"]
    )
    agg_all["CC_ACTIVE_CREDIT_CARDS"] = (
        agg_all["CC_CREDIT_LIMIT_MEAN"] > 0
    ).astype(int)

    for col in agg_all.columns:
        if col == "SK_ID_CURR":
            continue
        if agg_all[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg_all[col] = agg_all[col].fillna(0)

    print(f"[C4] 聚合完成: {agg_all.shape}")
    return agg_all


# ============================================================
# 主 Pipeline
# ============================================================
def process_credit_card_table(
    cc_path=f"{RAW_DIR}/credit_card_balance.csv",
    output_dir=PROCESSED_DIR,
):
    print("=" * 60)
    print("队员 C — credit_card_balance 表特征工程 Pipeline")
    print("=" * 60)

    print(f"\n[加载] 读取 {cc_path}...")
    cc = pd.read_csv(cc_path)
    print(f"  credit_card_balance: {cc.shape}")

    # --- C1: 清洗 + 缺失值 ---
    print("\n" + "-" * 40)
    cc = clean_and_fill_cc(cc)

    # --- C2: 类别编码 ---
    print("\n" + "-" * 40)
    cc = encode_cc(cc)

    # --- C3: 特征衍生 ---
    print("\n" + "-" * 40)
    cc = engineer_cc_features(cc)

    # --- C4: 聚合到 SK_ID_CURR ---
    print("\n" + "-" * 40)
    client_features = aggregate_cc_to_client(cc)

    # 保存 (不在此时标准化，统一交给队员 D 处理)
    out_path = f"{output_dir}/processed_credit_card_balance.csv"
    client_features.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({client_features.shape})")
    print("\n" + "=" * 60)
    print("credit_card_balance Pipeline 完成!")
    print("=" * 60)

    return client_features


if __name__ == "__main__":
    process_credit_card_table()
