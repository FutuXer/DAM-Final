"""
队员 A — installments_payments 表特征工程 Pipeline
处理 installments_payments.csv (分期还款流水)
输出: 按 SK_ID_CURR 聚合的分期还款行为特征
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# 配置
# ============================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

RECENT_INSTALLMENTS = 6


# ============================================================
# IP1: 缺失值处理
# ============================================================
def fill_missing_ip(df):
    df = df.copy()
    for col in df.columns:
        if col in ("SK_ID_PREV", "SK_ID_CURR"):
            continue
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == "object":
            modes = df[col].mode()
            df[col] = df[col].fillna(modes.iloc[0] if len(modes) > 0 else 0)
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


# ============================================================
# IP2: 行级特征衍生
# ============================================================
def engineer_ip_features(df):
    df = df.copy()

    # --- 还款延迟天数 ---
    df["IP_PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]

    # --- 还款差额 ---
    df["IP_PAYMENT_DIFF"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]

    # --- 还款比例 ---
    df["IP_PAYMENT_RATIO"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1)
    df["IP_PAYMENT_RATIO"] = df["IP_PAYMENT_RATIO"].clip(upper=10)

    # --- 逾期标记 ---
    df["IP_IS_LATE"] = (df["IP_PAYMENT_DELAY"] > 0).astype(np.uint8)
    df["IP_IS_LATE_OVER_30"] = (df["IP_PAYMENT_DELAY"] > 30).astype(np.uint8)
    df["IP_IS_LATE_OVER_60"] = (df["IP_PAYMENT_DELAY"] > 60).astype(np.uint8)

    # --- 欠款标记 ---
    df["IP_UNDERPAY"] = (df["AMT_PAYMENT"] < df["AMT_INSTALMENT"]).astype(np.uint8)

    # --- 零还款标记 ---
    df["IP_ZERO_PAYMENT"] = (df["AMT_PAYMENT"] == 0).astype(np.uint8)

    # --- 按账期版本标记版本变更 ---
    df["IP_INSTALMENT_PERIOD"] = df["DAYS_INSTALMENT"].abs() / 30.44

    print(f"[IP2] 行级衍生完成，维度: {df.shape[1]}")
    return df


# ============================================================
# IP3: 聚合到 SK_ID_PREV 级 (先聚到贷款级)
# ============================================================
def aggregate_to_prev(df):
    print(f"[IP3] 聚合到 SK_ID_PREV 级，输入: {df.shape}")

    # 全量聚合
    agg = df.groupby("SK_ID_PREV").agg(
        IP_INSTALMENTS_COUNT=("SK_ID_CURR", "count"),
        IP_PAYMENT_DELAY_MEAN=("IP_PAYMENT_DELAY", "mean"),
        IP_PAYMENT_DELAY_MAX=("IP_PAYMENT_DELAY", "max"),
        IP_PAYMENT_DELAY_MIN=("IP_PAYMENT_DELAY", "min"),
        IP_PAYMENT_DELAY_STD=("IP_PAYMENT_DELAY", "std"),
        IP_PAYMENT_DIFF_MEAN=("IP_PAYMENT_DIFF", "mean"),
        IP_PAYMENT_DIFF_SUM=("IP_PAYMENT_DIFF", "sum"),
        IP_PAYMENT_RATIO_MEAN=("IP_PAYMENT_RATIO", "mean"),
        IP_PAYMENT_RATIO_MIN=("IP_PAYMENT_RATIO", "min"),
        IP_PAYMENT_RATIO_MAX=("IP_PAYMENT_RATIO", "max"),
        IP_AMT_INSTALMENT_MEAN=("AMT_INSTALMENT", "mean"),
        IP_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
        IP_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
        IP_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        IP_AMT_PAYMENT_MAX=("AMT_PAYMENT", "max"),
        IP_IS_LATE_RATIO=("IP_IS_LATE", "mean"),
        IP_IS_LATE_OVER_30_RATIO=("IP_IS_LATE_OVER_30", "mean"),
        IP_IS_LATE_OVER_60_RATIO=("IP_IS_LATE_OVER_60", "mean"),
        IP_UNDERPAY_RATIO=("IP_UNDERPAY", "mean"),
        IP_ZERO_PAYMENT_RATIO=("IP_ZERO_PAYMENT", "mean"),
        IP_INSTALMENT_PERIOD_MEAN=("IP_INSTALMENT_PERIOD", "mean"),
        IP_INSTALMENT_VERSION_CHANGES=("NUM_INSTALMENT_VERSION", "nunique"),
    ).reset_index()

    # 首期/末期行为对比
    first = df.loc[df.groupby("SK_ID_PREV")["NUM_INSTALMENT_NUMBER"].idxmin()]
    last = df.loc[df.groupby("SK_ID_PREV")["NUM_INSTALMENT_NUMBER"].idxmax()]

    agg["IP_FIRST_PAYMENT_DELAY"] = first["IP_PAYMENT_DELAY"].values
    agg["IP_LAST_PAYMENT_DELAY"] = last["IP_PAYMENT_DELAY"].values
    agg["IP_FIRST_PAYMENT_RATIO"] = first["IP_PAYMENT_RATIO"].values
    agg["IP_LAST_PAYMENT_RATIO"] = last["IP_PAYMENT_RATIO"].values
    agg["IP_DELAY_TREND"] = agg["IP_LAST_PAYMENT_DELAY"] - agg["IP_FIRST_PAYMENT_DELAY"]

    # 最近 N 期行为
    sorted_df = df.sort_values("DAYS_INSTALMENT", ascending=False)
    recent = sorted_df.groupby("SK_ID_PREV").head(RECENT_INSTALLMENTS)
    agg_recent = recent.groupby("SK_ID_PREV").agg(
        IP_RECENT_DELAY_MEAN=("IP_PAYMENT_DELAY", "mean"),
        IP_RECENT_DELAY_MAX=("IP_PAYMENT_DELAY", "max"),
        IP_RECENT_PAYMENT_RATIO_MEAN=("IP_PAYMENT_RATIO", "mean"),
        IP_RECENT_LATE_RATIO=("IP_IS_LATE", "mean"),
        IP_RECENT_UNDERPAY_RATIO=("IP_UNDERPAY", "mean"),
        IP_RECENT_COUNT=("SK_ID_CURR", "count"),
    ).reset_index()
    agg = agg.merge(agg_recent, on="SK_ID_PREV", how="left")

    for col in agg.columns:
        if col == "SK_ID_PREV":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[IP3] 贷款级聚合完成: {agg.shape}")
    return agg


# ============================================================
# IP4: 聚合到 SK_ID_CURR (客户级)
# ============================================================
def aggregate_to_client(df_prev_agg, df_raw):
    print(f"[IP4] 聚合到 SK_ID_CURR，输入贷款级: {df_prev_agg.shape}")

    # 需要 SK_ID_CURR 的映射
    prev_to_curr = df_raw[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()
    df = df_prev_agg.merge(prev_to_curr, on="SK_ID_PREV", how="left")

    agg = df.groupby("SK_ID_CURR").agg(
        IP_LOANS_COUNT=("SK_ID_PREV", "count"),
        IP_TOTAL_INSTALMENTS=("IP_INSTALMENTS_COUNT", "sum"),
        IP_INSTALMENTS_PER_LOAN_MEAN=("IP_INSTALMENTS_COUNT", "mean"),
        IP_PAYMENT_DELAY_MEAN=("IP_PAYMENT_DELAY_MEAN", "mean"),
        IP_PAYMENT_DELAY_MAX=("IP_PAYMENT_DELAY_MAX", "max"),
        IP_PAYMENT_DELAY_STD=("IP_PAYMENT_DELAY_STD", "mean"),
        IP_PAYMENT_DIFF_SUM=("IP_PAYMENT_DIFF_SUM", "sum"),
        IP_PAYMENT_RATIO_MEAN=("IP_PAYMENT_RATIO_MEAN", "mean"),
        IP_PAYMENT_RATIO_MIN=("IP_PAYMENT_RATIO_MIN", "min"),
        IP_LATE_RATIO=("IP_IS_LATE_RATIO", "mean"),
        IP_LATE_OVER_30_RATIO=("IP_IS_LATE_OVER_30_RATIO", "mean"),
        IP_LATE_OVER_60_RATIO=("IP_IS_LATE_OVER_60_RATIO", "mean"),
        IP_UNDERPAY_RATIO=("IP_UNDERPAY_RATIO", "mean"),
        IP_ZERO_PAYMENT_RATIO=("IP_ZERO_PAYMENT_RATIO", "mean"),
        IP_FIRST_DELAY_MEAN=("IP_FIRST_PAYMENT_DELAY", "mean"),
        IP_LAST_DELAY_MEAN=("IP_LAST_PAYMENT_DELAY", "mean"),
        IP_DELAY_TREND_MEAN=("IP_DELAY_TREND", "mean"),
        IP_VERSION_CHANGES_MEAN=("IP_INSTALMENT_VERSION_CHANGES", "mean"),
        IP_RECENT_DELAY_MEAN=("IP_RECENT_DELAY_MEAN", "mean"),
        IP_RECENT_DELAY_MAX=("IP_RECENT_DELAY_MAX", "max"),
        IP_RECENT_LATE_RATIO_MEAN=("IP_RECENT_LATE_RATIO", "mean"),
        IP_RECENT_UNDERPAY_RATIO_MEAN=("IP_RECENT_UNDERPAY_RATIO", "mean"),
    ).reset_index()

    # 衍生比率
    agg["IP_TOTAL_PAYMENT_DIFF_RATIO"] = (
        agg["IP_PAYMENT_DIFF_SUM"] / (agg["IP_TOTAL_INSTALMENTS"] + 1)
    )
    agg["IP_HAS_ANY_LATE"] = (agg["IP_PAYMENT_DELAY_MAX"] > 0).astype(int)
    agg["IP_HAS_SEVERE_LATE"] = (agg["IP_LATE_OVER_60_RATIO"] > 0).astype(int)

    for col in agg.columns:
        if col == "SK_ID_CURR":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[IP4] 客户级聚合完成: {agg.shape}")
    return agg


# ============================================================
# IP5: 标准化
# ============================================================
def standardize_ip(df):
    df = df.copy()
    exclude = {"SK_ID_CURR"}
    scale_cols = [c for c in df.columns
                  if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]
                  and df[c].nunique() > 2]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[IP5] 标准化完成: {len(scale_cols)} 列")
    return df


# ============================================================
# 主 Pipeline
# ============================================================
def process_installments_table(
    ip_path=f"{RAW_DIR}/installments_payments.csv",
    output_dir=PROCESSED_DIR,
):
    print("=" * 60)
    print("队员 A — installments_payments 表特征工程 Pipeline")
    print("=" * 60)

    print(f"\n[加载] 读取 {ip_path}...")
    ip = pd.read_csv(ip_path)
    print(f"  installments_payments: {ip.shape}")

    # --- IP1: 缺失值 ---
    print("\n" + "-" * 40)
    ip = fill_missing_ip(ip)

    # --- IP2: 特征衍生 ---
    print("\n" + "-" * 40)
    ip = engineer_ip_features(ip)

    # --- IP3: 聚到贷款级 ---
    print("\n" + "-" * 40)
    prev_agg = aggregate_to_prev(ip)

    # --- IP4: 聚到客户级 ---
    print("\n" + "-" * 40)
    client_features = aggregate_to_client(prev_agg, ip)

    # --- IP5: 标准化 ---
    print("\n" + "-" * 40)
    client_features = standardize_ip(client_features)

    out_path = f"{output_dir}/processed_installments_payments.csv"
    client_features.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({client_features.shape})")
    print("\n" + "=" * 60)
    print("installments_payments Pipeline 完成!")
    print("=" * 60)

    return client_features


if __name__ == "__main__":
    process_installments_table()
