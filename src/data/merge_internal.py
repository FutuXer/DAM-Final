"""
队员 C — 内部流水特征合并脚本
将 4 张已聚合的客户级特征表合并为 features_internal.csv
"""
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def merge_internal_features(
    prev_path=f"{PROCESSED_DIR}/processed_previous_application.csv",
    cc_path=f"{PROCESSED_DIR}/processed_credit_card_balance.csv",
    ip_path=f"{PROCESSED_DIR}/processed_installments_payments.csv",
    pc_path=f"{PROCESSED_DIR}/processed_pos_cash_balance.csv",
    output_dir=PROCESSED_DIR,
):
    print("=" * 60)
    print("队员 C — 内部流水特征合并 (4 → 1)")
    print("=" * 60)

    # 以 previous_application 为基表（SK_ID_CURR 最全）
    print(f"\n[加载] 基表: {prev_path}")
    base = pd.read_csv(prev_path)
    print(f"  {base.shape}")

    for label, path in [
        ("credit_card_balance", cc_path),
        ("installments_payments", ip_path),
        ("pos_cash_balance", pc_path),
    ]:
        print(f"\n[合并] {label}: {path}")
        feats = pd.read_csv(path)
        print(f"  {feats.shape}")
        base = base.merge(feats, on="SK_ID_CURR", how="left")

    # 填充合并产生的新缺失值
    for col in base.columns:
        if col == "SK_ID_CURR":
            continue
        if base[col].dtype in ["float64", "float32", "int64", "int32"]:
            base[col] = base[col].fillna(base[col].median() if base[col].notna().any() else 0)

    out_path = f"{output_dir}/features_internal.csv"
    base.to_csv(out_path, index=False)
    print(f"\n[保存] → {out_path} ({base.shape})")
    print("=" * 60)
    return base


if __name__ == "__main__":
    merge_internal_features()
