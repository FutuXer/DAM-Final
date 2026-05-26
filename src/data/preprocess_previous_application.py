"""
队员 A — previous_application 表特征工程 Pipeline
处理 previous_application.csv (历史申请记录)
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

CAT_COLS = [
    "NAME_CONTRACT_TYPE", "WEEKDAY_APPR_PROCESS_START",
    "NAME_CASH_LOAN_PURPOSE", "NAME_CONTRACT_STATUS",
    "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON",
    "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE",
    "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE", "CHANNEL_TYPE",
    "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP",
    "PRODUCT_COMBINATION",
]

FLAG_COLS = [
    "FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY",
    "NFLAG_INSURED_ON_APPROVAL",
]

DAYS_COLS = [
    "DAYS_DECISION", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

AMT_COLS = [
    "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT",
    "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
]

RATE_COLS = [
    "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED",
]


# ============================================================
# PA1: 缺失值处理
# ============================================================
def fill_missing_pa(df):
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
# PA2: 时间列转换
# ============================================================
def convert_days_pa(df):
    df = df.copy()
    for col in DAYS_COLS:
        if col not in df.columns:
            continue
        new_name = "YEARS_" + col[5:]
        df[new_name] = df[col].abs() / 365.25
    return df


# ============================================================
# PA3: 类别编码
# ============================================================
def encode_pa(df):
    df = df.copy()
    cat_present = [c for c in CAT_COLS if c in df.columns]
    for col in cat_present:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[PA3] 类别编码完成: {len(cat_present)} 列")
    return df


# ============================================================
# PA4: 行级特征衍生
# ============================================================
def engineer_pa_features(df):
    df = df.copy()

    # --- 申请金额差异 (审批后金额 vs 申请金额) ---
    df["PA_AMT_DIFF"] = df["AMT_CREDIT"] - df["AMT_APPLICATION"]
    df["PA_AMT_DIFF_RATIO"] = (
        df["PA_AMT_DIFF"] / (df["AMT_APPLICATION"] + 1)
    ).clip(lower=-1, upper=5)
    df["PA_AMT_REDUCED"] = (df["PA_AMT_DIFF"] < 0).astype(np.uint8)

    # --- 首付占比 ---
    df["PA_DOWN_PAYMENT_RATIO"] = (
        df["AMT_DOWN_PAYMENT"] / (df["AMT_CREDIT"] + 1)
    ).clip(upper=5)

    # --- 商品价格贷款比 ---
    if "AMT_GOODS_PRICE" in df.columns:
        df["PA_LOAN_TO_GOODS"] = (
            df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
        ).clip(upper=10)

    # --- 年金贷款比 ---
    df["PA_ANNUITY_CREDIT_RATIO"] = (
        df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    ).clip(upper=5)

    # --- 贷款期限 (年) ---
    if "YEARS_DECISION" in df.columns and "YEARS_TERMINATION" in df.columns:
        df["PA_CREDIT_DURATION"] = (
            df["YEARS_TERMINATION"] - df["YEARS_DECISION"]
        ).clip(lower=0)

    # --- 审批时长 ---
    if "YEARS_FIRST_DRAWING" in df.columns and "YEARS_DECISION" in df.columns:
        df["PA_DECISION_TO_DRAWING"] = (
            df["YEARS_FIRST_DRAWING"] - df["YEARS_DECISION"]
        ).abs().clip(upper=5)

    # --- 申请时间距当前时间 ---
    if "YEARS_DECISION" in df.columns:
        df["PA_RECENT_APPLICATION"] = (df["YEARS_DECISION"] < 1).astype(np.uint8)

    # --- 拒绝标记 ---
    df["PA_IS_REJECTED"] = df["NAME_CONTRACT_STATUS"].isin(
        ["Refused", "Canceled", "Cancelled"]
    ).astype(np.uint8)
    df["PA_IS_APPROVED"] = (~df["PA_IS_REJECTED"].astype(bool)).astype(np.uint8)

    # --- 费率组合 ---
    available_rates = [c for c in RATE_COLS if c in df.columns]
    if len(available_rates) >= 2:
        df["PA_RATE_MEAN"] = df[available_rates].mean(axis=1)
        df["PA_RATE_MAX"] = df[available_rates].max(axis=1)

    # --- 最后申请标记 ---
    for col in FLAG_COLS:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map({"Y": 1, "N": 0, "1": 1, "0": 0}).fillna(0)
            else:
                df[col] = df[col].fillna(0)
            df[col] = df[col].astype(np.uint8)

    # --- 申请时段 ---
    if "HOUR_APPR_PROCESS_START" in df.columns:
        hour = df["HOUR_APPR_PROCESS_START"].fillna(12)
        df["PA_APPLY_NIGHT"] = ((hour >= 22) | (hour <= 5)).astype(np.uint8)
        df["PA_APPLY_MORNING"] = ((hour >= 6) & (hour <= 11)).astype(np.uint8)

    print(f"[PA4] 行级衍生完成，维度: {df.shape[1]}")
    return df


# ============================================================
# PA5: 聚合到 SK_ID_CURR
# ============================================================
def aggregate_to_client(df):
    print(f"[PA5] 聚合到客户级，输入: {df.shape}")

    # 全量统计
    agg = df.groupby("SK_ID_CURR").agg(
        PA_LOANS_COUNT=("SK_ID_PREV", "count"),
        PA_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PA_AMT_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
        PA_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PA_AMT_APPLICATION_MIN=("AMT_APPLICATION", "min"),
        PA_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PA_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PA_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PA_AMT_DIFF_MEAN=("PA_AMT_DIFF", "mean"),
        PA_AMT_DIFF_RATIO_MEAN=("PA_AMT_DIFF_RATIO", "mean"),
        PA_AMT_REDUCED_RATIO=("PA_AMT_REDUCED", "mean"),
        PA_DOWN_PAYMENT_RATIO_MEAN=("PA_DOWN_PAYMENT_RATIO", "mean"),
        PA_ANNUITY_CREDIT_RATIO_MEAN=("PA_ANNUITY_CREDIT_RATIO", "mean"),
        PA_ANNUITY_CREDIT_RATIO_MAX=("PA_ANNUITY_CREDIT_RATIO", "max"),
        PA_CREDIT_DURATION_MEAN=("PA_CREDIT_DURATION", "mean"),
        PA_CREDIT_DURATION_MAX=("PA_CREDIT_DURATION", "max"),
        PA_DECISION_TO_DRAWING_MEAN=("PA_DECISION_TO_DRAWING", "mean"),
        PA_IS_REJECTED_RATIO=("PA_IS_REJECTED", "mean"),
        PA_IS_APPROVED_RATIO=("PA_IS_APPROVED", "mean"),
        PA_FLAG_LAST_APPL_RATIO=("FLAG_LAST_APPL_PER_CONTRACT", "mean"),
        PA_NFLAG_LAST_APPL_IN_DAY_RATIO=("NFLAG_LAST_APPL_IN_DAY", "mean"),
        PA_NFLAG_INSURED_RATIO=("NFLAG_INSURED_ON_APPROVAL", "mean"),
        PA_RATE_PRIMARY_MEAN=("RATE_INTEREST_PRIMARY", "mean"),
        PA_RATE_PRIMARY_MAX=("RATE_INTEREST_PRIMARY", "max"),
        PA_RATE_PRIVILEGED_MEAN=("RATE_INTEREST_PRIVILEGED", "mean"),
        PA_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PA_CNT_PAYMENT_MAX=("CNT_PAYMENT", "max"),
        PA_APPLY_NIGHT_RATIO=("PA_APPLY_NIGHT", "mean"),
    ).reset_index()

    # 最近一次申请的行为
    last_idx = df.groupby("SK_ID_CURR")["YEARS_DECISION"].idxmin()
    last_app = df.loc[last_idx, :]

    last_features = last_app[[
        "SK_ID_CURR", "AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY",
        "PA_DOWN_PAYMENT_RATIO", "PA_ANNUITY_CREDIT_RATIO",
        "PA_CREDIT_DURATION", "PA_IS_REJECTED",
    ]].copy()
    last_features.columns = ["SK_ID_CURR"] + [
        f"PA_LAST_{c}" for c in last_features.columns if c != "SK_ID_CURR"
    ]
    agg = agg.merge(last_features, on="SK_ID_CURR", how="left")

    # 不同产品的申请数量
    if "NAME_PORTFOLIO" in df.columns:
        portfolio_cnt = df.groupby(["SK_ID_CURR", "NAME_PORTFOLIO"]).size() \
            .unstack(fill_value=0).add_prefix("PA_PORTFOLIO_").reset_index()
        agg = agg.merge(portfolio_cnt, on="SK_ID_CURR", how="left")

    if "NAME_CONTRACT_TYPE" in df.columns:
        type_cnt = df.groupby(["SK_ID_CURR", "NAME_CONTRACT_TYPE"]).size() \
            .unstack(fill_value=0).add_prefix("PA_CONTRACT_TYPE_").reset_index()
        agg = agg.merge(type_cnt, on="SK_ID_CURR", how="left")

    # 衍生比率
    agg["PA_APPROVAL_RATE"] = agg["PA_IS_APPROVED_RATIO"]
    agg["PA_REJECTION_RATE"] = agg["PA_IS_REJECTED_RATIO"]
    agg["PA_TOTAL_CREDIT_SUM"] = agg["PA_AMT_CREDIT_SUM"]
    agg["PA_HAS_REJECTION"] = (agg["PA_REJECTION_RATE"] > 0).astype(int)

    # 申请频次
    agg["PA_APPLICATION_INTENSITY"] = (
        agg["PA_LOANS_COUNT"] / (agg["PA_CREDIT_DURATION_MEAN"] + 1)
    ).clip(upper=50)

    for col in agg.columns:
        if col == "SK_ID_CURR":
            continue
        if agg[col].dtype in ["float64", "float32", "int64", "int32"]:
            agg[col] = agg[col].fillna(0)

    print(f"[PA5] 客户级聚合完成: {agg.shape}")
    return agg


# ============================================================
# PA6: 标准化
# ============================================================
def standardize_pa(df):
    df = df.copy()
    exclude = {"SK_ID_CURR"}
    scale_cols = [c for c in df.columns
                  if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]
                  and df[c].nunique() > 2]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[PA6] 标准化完成: {len(scale_cols)} 列")
    return df


# ============================================================
# 主 Pipeline
# ============================================================
def process_previous_application_table(
    pa_path=f"{RAW_DIR}/previous_application.csv",
    output_dir=PROCESSED_DIR,
):
    print("=" * 60)
    print("队员 A — previous_application 表特征工程 Pipeline")
    print("=" * 60)

    print(f"\n[加载] 读取 {pa_path}...")
    pa = pd.read_csv(pa_path)
    print(f"  previous_application: {pa.shape}")

    # --- PA1: 缺失值 ---
    print("\n" + "-" * 40)
    pa = fill_missing_pa(pa)

    # --- PA2: 时间转换 ---
    print("\n" + "-" * 40)
    pa = convert_days_pa(pa)

    # --- PA3: 特征衍生 (在编码之前，利用原始字符串判断状态) ---
    print("\n" + "-" * 40)
    pa = engineer_pa_features(pa)

    # --- PA4: 类别编码 (在衍生之后) ---
    print("\n" + "-" * 40)
    pa = encode_pa(pa)

    # --- PA5: 聚合到客户级 ---
    print("\n" + "-" * 40)
    client_features = aggregate_to_client(pa)

    # --- PA6: 标准化 ---
    print("\n" + "-" * 40)
    client_features = standardize_pa(client_features)

    out_path = f"{output_dir}/processed_previous_application.csv"
    client_features.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({client_features.shape})")
    print("\n" + "=" * 60)
    print("previous_application Pipeline 完成!")
    print("=" * 60)

    return client_features


if __name__ == "__main__":
    process_previous_application_table()
