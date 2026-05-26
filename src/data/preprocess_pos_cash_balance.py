"""
队员 A — POS_CASH_balance 表特征工程 Pipeline
处理 POS_CASH_balance.csv (POS/现金贷月度快照)
输出: 按 SK_ID_CURR 聚合的特征矩阵
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================================
# 配置
# ============================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

CAT_COLS = ["NAME_CONTRACT_STATUS"]

RECENT_MONTHS_THRESHOLD = -6


# ============================================================
# PC1: 缺失值处理
# ============================================================
def fill_missing_pc(df):
    df = df.copy()
    for col in df.columns:
        if col in ("SK_ID_PREV", "SK_ID_CURR"):
            continue
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == "object":
            modes = df[col].mode()
            df[col] = df[col].fillna(modes.iloc[0] if len(modes) > 0 else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


# ============================================================
# PC2: 类别编码
# ============================================================
def encode_pc(df):
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    print(f"[PC2] 类别编码完成: {CAT_COLS}")
    return df


# ============================================================
# PC3: 行级特征衍生
# ============================================================
def engineer_pc_features(df):
    df = df.copy()

    # --- 剩余期数占比 ---
    df["PC_REMAINING_RATIO"] = (
        df["CNT_INSTALMENT_FUTURE"] / (df["CNT_INSTALMENT"] + 1)
    ).clip(upper=5)

    # --- 已完成期数 ---
    df["PC_COMPLETED_INSTALMENTS"] = (
        df["CNT_INSTALMENT"] - df["CNT_INSTALMENT_FUTURE"]
    ).clip(lower=0)

    # --- 还款进度 ---
    df["PC_PROGRESS_RATIO"] = (
        df["PC_COMPLETED_INSTALMENTS"] / (df["CNT_INSTALMENT"] + 1)
    )

    # --- 逾期标记 ---
    df["PC_HAS_DPD"] = (df["SK_DPD"] > 0).astype(np.uint8)
    df["PC_HAS_DEF_DPD"] = (df["SK_DPD_DEF"] > 0).astype(np.uint8)
    df["PC_DPD_OVER_30"] = (df["SK_DPD"] >= 30).astype(np.uint8)
    df["PC_DPD_OVER_60"] = (df["SK_DPD"] >= 60).astype(np.uint8)
    df["PC_DPD_OVER_90"] = (df["SK_DPD"] >= 90).astype(np.uint8)

    # --- 月数绝对值 ---
    df["PC_MONTHS_ABS"] = df["MONTHS_BALANCE"].abs()

    print(f"[PC3] 行级衍生完成，维度: {df.shape[1]}")
    return df


# ============================================================
# PC4: 聚合到 SK_ID_PREV (贷款级)
# ============================================================
def aggregate_to_prev(df):
    print(f"[PC4] 聚合到 SK_ID_PREV 级，输入: {df.shape}")

    agg = df.groupby("SK_ID_PREV").agg(
        PC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        PC_CNT_INSTALMENT_MAX=("CNT_INSTALMENT", "max"),
        PC_CNT_INSTALMENT_LAST=("CNT_INSTALMENT", "last"),
        PC_CNT_INSTALMENT_FUTURE_LAST=("CNT_INSTALMENT_FUTURE", "last"),
        PC_REMAINING_RATIO_MEAN=("PC_REMAINING_RATIO", "mean"),
        PC_REMAINING_RATIO_MIN=("PC_REMAINING_RATIO", "min"),
        PC_PROGRESS_RATIO_LAST=("PC_PROGRESS_RATIO", "last"),
        PC_SK_DPD_MAX=("SK_DPD", "max"),
        PC_SK_DPD_MEAN=("SK_DPD", "mean"),
        PC_SK_DPD_SUM=("SK_DPD", "sum"),
        PC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        PC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        PC_HAS_DPD_RATIO=("PC_HAS_DPD", "mean"),
        PC_HAS_DEF_DPD_RATIO=("PC_HAS_DEF_DPD", "mean"),
        PC_DPD_OVER_30_RATIO=("PC_DPD_OVER_30", "mean"),
        PC_DPD_OVER_60_RATIO=("PC_DPD_OVER_60", "mean"),
        PC_DPD_OVER_90_RATIO=("PC_DPD_OVER_90", "mean"),
        PC_MONTHS_ABS_MAX=("PC_MONTHS_ABS", "max"),
    ).reset_index()

    # 首月/末月趋势
    first = df.loc[df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()]
    last = df.loc[df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmin()]

    agg["PC_TREND_DPD"] = last["SK_DPD"].values - first["SK_DPD"].values
    agg["PC_TREND_REMAINING"] = (
        first["CNT_INSTALMENT_FUTURE"].values - last["CNT_INSTALMENT_FUTURE"].values
    )
    agg["PC_FIRST_DPD"] = first["SK_DPD"].values
    agg["PC_LAST_DPD"] = last["SK_DPD"].values

    # 还款速度 (每月完成的期数)
    agg["PC_PAYMENT_SPEED"] = (
        agg["PC_TREND_REMAINING"] / (agg["PC_MONTHS_COUNT"] + 1)
    ).clip(lower=0)

    for col in agg.columns:
        if col == "SK_ID_PREV":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[PC4] 贷款级聚合完成: {agg.shape}")
    return agg


# ============================================================
# PC5: 聚合到 SK_ID_CURR (客户级)
# ============================================================
def aggregate_to_client(df_prev_agg, df_raw):
    print(f"[PC5] 聚合到 SK_ID_CURR，输入贷款级: {df_prev_agg.shape}")

    prev_to_curr = df_raw[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()
    df = df_prev_agg.merge(prev_to_curr, on="SK_ID_PREV", how="left")

    agg = df.groupby("SK_ID_CURR").agg(
        PC_LOANS_COUNT=("SK_ID_PREV", "count"),
        PC_MONTHS_TOTAL=("PC_MONTHS_COUNT", "sum"),
        PC_MONTHS_PER_LOAN_MEAN=("PC_MONTHS_COUNT", "mean"),
        PC_CNT_INSTALMENT_MAX=("PC_CNT_INSTALMENT_MAX", "max"),
        PC_CNT_INSTALMENT_MEAN=("PC_CNT_INSTALMENT_LAST", "mean"),
        PC_REMAINING_RATIO_MEAN=("PC_REMAINING_RATIO_MEAN", "mean"),
        PC_PROGRESS_RATIO_MEAN=("PC_PROGRESS_RATIO_LAST", "mean"),
        PC_SK_DPD_MAX=("PC_SK_DPD_MAX", "max"),
        PC_SK_DPD_MEAN=("PC_SK_DPD_MEAN", "mean"),
        PC_SK_DPD_DEF_MAX=("PC_SK_DPD_DEF_MAX", "max"),
        PC_DPD_OVER_30_RATIO=("PC_DPD_OVER_30_RATIO", "mean"),
        PC_DPD_OVER_60_RATIO=("PC_DPD_OVER_60_RATIO", "mean"),
        PC_DPD_OVER_90_RATIO=("PC_DPD_OVER_90_RATIO", "mean"),
        PC_PAYMENT_SPEED_MEAN=("PC_PAYMENT_SPEED", "mean"),
        PC_TREND_DPD_MEAN=("PC_TREND_DPD", "mean"),
        PC_TREND_REMAINING_MEAN=("PC_TREND_REMAINING", "mean"),
        PC_FIRST_DPD_MEAN=("PC_FIRST_DPD", "mean"),
        PC_LAST_DPD_MEAN=("PC_LAST_DPD", "mean"),
    ).reset_index()

    # 近期行为 (最近 N 个月的所有贷款快照)
    if "SK_ID_PREV" in df_raw.columns and "SK_ID_CURR" in df_raw.columns:
        recent = df_raw[df_raw["MONTHS_BALANCE"] >= RECENT_MONTHS_THRESHOLD]
        if len(recent) > 0:
            agg_recent = recent.groupby("SK_ID_CURR").agg(
                PC_RECENT_MONTHS=("MONTHS_BALANCE", "count"),
                PC_RECENT_DPD_MAX=("SK_DPD", "max"),
                PC_RECENT_DPD_MEAN=("SK_DPD", "mean"),
                PC_RECENT_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            ).reset_index()
            agg = agg.merge(agg_recent, on="SK_ID_CURR", how="left")

    # 衍生比率
    agg["PC_HAS_ANY_DPD"] = (agg["PC_SK_DPD_MAX"] > 0).astype(int)
    agg["PC_HAS_SEVERE_DPD"] = (agg["PC_DPD_OVER_90_RATIO"] > 0).astype(int)
    agg["PC_DPD_PER_LOAN"] = (
        agg["PC_SK_DPD_MEAN"] / (agg["PC_LOANS_COUNT"] + 1)
    )

    for col in agg.columns:
        if col == "SK_ID_CURR":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[PC5] 客户级聚合完成: {agg.shape}")
    return agg


# ============================================================
# PC6: 标准化
# ============================================================
def standardize_pc(df):
    df = df.copy()
    exclude = {"SK_ID_CURR"}
    scale_cols = [c for c in df.columns
                  if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]
                  and df[c].nunique() > 2]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[PC6] 标准化完成: {len(scale_cols)} 列")
    return df


# ============================================================
# 主 Pipeline
# ============================================================
def process_pos_cash_table(
    pc_path=f"{RAW_DIR}/POS_CASH_balance.csv",
    output_dir=PROCESSED_DIR,
):
    print("=" * 60)
    print("队员 A — POS_CASH_balance 表特征工程 Pipeline")
    print("=" * 60)

    print(f"\n[加载] 读取 {pc_path}...")
    pc = pd.read_csv(pc_path)
    print(f"  POS_CASH_balance: {pc.shape}")

    # --- PC1: 缺失值 ---
    print("\n" + "-" * 40)
    pc = fill_missing_pc(pc)

    # --- PC2: 类别编码 ---
    print("\n" + "-" * 40)
    pc = encode_pc(pc)

    # --- PC3: 特征衍生 ---
    print("\n" + "-" * 40)
    pc = engineer_pc_features(pc)

    # --- PC4: 聚到贷款级 ---
    print("\n" + "-" * 40)
    prev_agg = aggregate_to_prev(pc)

    # --- PC5: 聚到客户级 ---
    print("\n" + "-" * 40)
    client_features = aggregate_to_client(prev_agg, pc)

    # --- PC6: 标准化 ---
    print("\n" + "-" * 40)
    client_features = standardize_pc(client_features)

    out_path = f"{output_dir}/processed_pos_cash_balance.csv"
    client_features.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({client_features.shape})")
    print("\n" + "=" * 60)
    print("POS_CASH_balance Pipeline 完成!")
    print("=" * 60)

    return client_features


if __name__ == "__main__":
    process_pos_cash_table()
