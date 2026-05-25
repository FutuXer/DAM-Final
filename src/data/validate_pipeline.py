"""
验证 Pipeline: 用合成数据测试 preprocess_application 的每个步骤.
运行: python src/data/validate_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from src.data.preprocess_application import (
    fill_missing, treat_outliers, convert_days_features,
    engineer_features, encode_categorical,
)

np.random.seed(42)
N = 500


def make_synthetic_data(n=N):
    """生成与 Home Credit 主表结构相似的合成数据."""
    df = pd.DataFrame({
        "SK_ID_CURR": range(100000, 100000 + n),
        "TARGET": np.random.binomial(1, 0.08, n),
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": np.random.choice(["M", "F"], n),
        "FLAG_OWN_CAR": np.random.binomial(1, 0.35, n),
        "FLAG_OWN_REALTY": np.random.binomial(1, 0.65, n),
        "CNT_CHILDREN": np.random.poisson(1, n),
        "AMT_INCOME_TOTAL": np.random.lognormal(11.5, 0.5, n).astype(int),
        "AMT_CREDIT": np.random.lognormal(12.5, 0.4, n).astype(int),
        "AMT_ANNUITY": np.random.lognormal(10, 0.3, n).astype(int),
        "AMT_GOODS_PRICE": np.random.lognormal(12, 0.5, n).astype(int),
        "NAME_TYPE_SUITE": np.random.choice(["Unaccompanied", "Family", "Spouse"], n),
        "NAME_INCOME_TYPE": np.random.choice(
            ["Working", "Commercial associate", "State servant", "Pensioner"], n),
        "NAME_EDUCATION_TYPE": np.random.choice(
            ["Secondary", "Higher education", "Incomplete higher"], n),
        "NAME_FAMILY_STATUS": np.random.choice(
            ["Married", "Single", "Civil marriage", "Separated", "Widow"], n),
        "NAME_HOUSING_TYPE": np.random.choice(
            ["House / apartment", "With parents", "Rented apartment"], n),
        "ORGANIZATION_TYPE": np.random.choice(
            ["Business Entity Type 3", "School", "Self-employed", "Medicine",
             "Transport: type 3", "Construction"], n),
        "OCCUPATION_TYPE": np.random.choice(
            ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
             "Cleaning staff", "Cooking staff", np.nan], n),
        "REGION_POPULATION_RELATIVE": np.random.normal(0.02, 0.01, n).clip(0),
        "DAYS_BIRTH": -np.random.randint(7500, 24000, n),
        "DAYS_EMPLOYED": np.where(
            np.random.random(n) < 0.06,
            365243,
            -np.random.randint(0, 15000, n).astype(float)
        ),
        "DAYS_REGISTRATION": -np.random.randint(0, 10000, n).astype(float),
        "DAYS_ID_PUBLISH": -np.random.randint(0, 12000, n).astype(float),
        "DAYS_LAST_PHONE_CHANGE": -np.random.randint(0, 4000, n).astype(float),
        "OWN_CAR_AGE": np.where(
            np.random.random(n) < 0.65,
            np.random.randint(0, 30, n),
            np.nan
        ),
        "FLAG_MOBIL": np.ones(n, dtype=int),
        "FLAG_EMP_PHONE": np.random.binomial(1, 0.8, n),
        "FLAG_WORK_PHONE": np.random.binomial(1, 0.2, n),
        "FLAG_CONT_MOBILE": np.random.binomial(1, 0.95, n),
        "FLAG_PHONE": np.random.binomial(1, 0.3, n),
        "FLAG_EMAIL": np.random.binomial(1, 0.05, n),
        "REG_REGION_NOT_LIVE_REGION": np.zeros(n, dtype=int),
        "REG_REGION_NOT_WORK_REGION": np.zeros(n, dtype=int),
        "LIVE_REGION_NOT_WORK_REGION": np.zeros(n, dtype=int),
        "REG_CITY_NOT_LIVE_CITY": np.zeros(n, dtype=int),
        "REG_CITY_NOT_WORK_CITY": np.zeros(n, dtype=int),
        "LIVE_CITY_NOT_WORK_CITY": np.zeros(n, dtype=int),
        "WEEKDAY_APPR_PROCESS_START": np.random.choice(
            ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], n),
        "HOUR_APPR_PROCESS_START": np.random.randint(6, 22, n),
        "REGION_RATING_CLIENT": np.random.choice([1, 2, 3], n),
        "REGION_RATING_CLIENT_W_CITY": np.random.choice([1, 2, 3], n),
        "EXT_SOURCE_1": np.random.normal(0.5, 0.15, n).clip(0, 1),
        "EXT_SOURCE_2": np.where(
            np.random.random(n) < 0.3,
            np.nan,
            np.random.normal(0.5, 0.15, n).clip(0, 1)
        ),
        "EXT_SOURCE_3": np.where(
            np.random.random(n) < 0.2,
            np.nan,
            np.random.normal(0.5, 0.15, n).clip(0, 1)
        ),
        "CNT_FAM_MEMBERS": np.random.poisson(2.5, n) + 1,
        "OBS_30_CNT_SOCIAL_CIRCLE": np.random.poisson(2, n),
        "DEF_30_CNT_SOCIAL_CIRCLE": np.random.poisson(0.3, n),
        "OBS_60_CNT_SOCIAL_CIRCLE": np.random.poisson(2, n),
        "DEF_60_CNT_SOCIAL_CIRCLE": np.random.poisson(0.2, n),
        "AMT_REQ_CREDIT_BUREAU_HOUR": np.zeros(n, dtype=int),
        "AMT_REQ_CREDIT_BUREAU_DAY": np.random.poisson(0.1, n),
        "AMT_REQ_CREDIT_BUREAU_WEEK": np.random.poisson(0.3, n),
        "AMT_REQ_CREDIT_BUREAU_MON": np.random.poisson(0.8, n),
        "AMT_REQ_CREDIT_BUREAU_QRT": np.random.poisson(1.5, n),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.random.poisson(3, n),
    })

    # 添加 FLAG_DOCUMENT 列
    for i in range(2, 22):
        df[f"FLAG_DOCUMENT_{i}"] = np.random.binomial(1, 0.5 if i <= 5 else 0.3, n)

    # 添加建筑特征列 (AVG 部分)
    for base in ["APARTMENTS", "BASEMENTAREA", "ELEVATORS", "ENTRANCES",
                  "FLOORSMAX", "LIVINGAREA"]:
        for suf in ["AVG", "MODE", "MEDI"]:
            df[f"{base}_{suf}"] = np.random.normal(0.1, 0.03, n).clip(0)

    return df


def validate_pipeline():
    """分步骤验证并打印中间结果."""
    print("=" * 60)
    print("Pipeline 验证测试")
    print("=" * 60)

    df = make_synthetic_data()
    print(f"\n[初始化] 合成数据: {df.shape}")

    # Step 1: 缺失值
    df_1 = fill_missing(df.drop(columns=["TARGET"]))
    assert df_1.isnull().sum().sum() == 0, "仍有缺失值!"
    print("[A1] [PASS] 缺失值填充完成，0 列仍有缺失")

    # Step 2: 异常值
    df_2 = treat_outliers(df_1.copy())
    assert "FLAG_UNEMPLOYED" in df_2.columns, "缺少失业标记列"
    assert df_2["DAYS_EMPLOYED"].max() < 365243, "DAYS_EMPLOYED 异常值未处理"
    print("[A2] [PASS] 异常值处理完成")
    print(f"     无业人数: {df_2['FLAG_UNEMPLOYED'].sum()}")

    # Step 3: 时间转换
    df_3 = convert_days_features(df_2.copy())
    assert "AGE_YEARS" in df_3.columns, "缺少 AGE_YEARS 列"
    assert "YEARS_EMPLOYED" in df_3.columns, "缺少 YEARS_EMPLOYED 列"
    assert df_3["AGE_YEARS"].min() > 0, "年龄出现负值"
    print("[A4] [PASS] 时间转换完成")
    print(f"     年龄范围: {df_3['AGE_YEARS'].min():.1f} ~ {df_3['AGE_YEARS'].max():.1f}")

    # Step 4: 特征衍生
    df_4 = engineer_features(df_3.copy())
    expected_new = [
        "INCOME_CREDIT_RATIO", "ANNUITY_INCOME_RATIO",
        "EXT_SOURCE_WEIGHTED", "SOCIAL_DEF_30_RATIO",
        "AGE_YOUNG", "AGE_OLD", "CHILDREN_RATIO",
        "INCOME_PER_PERSON", "BUREAU_ENQUIRY_TOTAL",
        "DOCUMENTS_PROVIDED",
    ]
    for feat in expected_new:
        assert feat in df_4.columns, f"缺少衍生特征: {feat}"
    print("[A5] [PASS] 特征衍生完成")
    print(f"     新维度: {df_4.shape[1]} (原始 ~50 列)")

    # Step 5: 类别编码
    df_5 = encode_categorical(df_4.copy())
    obj_cols = df_5.select_dtypes(include=["object", "category"]).columns
    assert len(obj_cols) == 0, f"仍有未编码的类别列: {list(obj_cols)}"
    print(f"[A3] [PASS] 类别编码完成, 最终维度: {df_5.shape[1]}")

    # 最终检查
    assert df_5.isnull().sum().sum() == 0, "最终结果中存在缺失值!"

    print("\n" + "=" * 60)
    print("全部验证通过! Pipeline 可正常运行.")
    print("=" * 60)
    return df_5


if __name__ == "__main__":
    validate_pipeline()
