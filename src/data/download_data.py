"""
Home Credit 数据集下载脚本
使用 Kaggle API 下载: https://www.kaggle.com/competitions/home-credit-default-risk/data

前置条件:
    1. pip install kaggle
    2. 从 Kaggle 获取 API key (kaggle.json), 放到 ~/.kaggle/ 下
    3. 或者手动下载后放到 data/raw/ 目录

使用方法:
    python src/data/download_data.py
"""
import os
import sys
import zipfile
import shutil

RAW_DIR = "data/raw"
COMPETITION = "home-credit-default-risk"

FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "previous_application.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
]


def download_with_kaggle_api():
    """使用 Kaggle API 下载."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("请先安装 kaggle: pip install kaggle")
        print("然后配置 API key: https://www.kaggle.com/settings/account")
        return False

    print(f"[下载] 从 Kaggle 下载 {COMPETITION} 竞赛数据...")
    os.makedirs(RAW_DIR, exist_ok=True)

    ret = os.system(
        f"kaggle competitions download -c {COMPETITION} -p {RAW_DIR}"
    )
    if ret != 0:
        print("[错误] Kaggle API 下载失败，请检查 API key 配置")
        return False

    # 解压所有 zip 文件
    for item in os.listdir(RAW_DIR):
        if item.endswith(".zip"):
            zip_path = os.path.join(RAW_DIR, item)
            print(f"[解压] {item}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
            os.remove(zip_path)

    print("[完成] 数据下载完毕!")
    return True


def list_missing_files():
    """检查缺少哪些数据文件."""
    missing = []
    for f in FILES:
        path = os.path.join(RAW_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
    return missing


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    missing = list_missing_files()
    if not missing:
        print("[OK] 所有数据文件已就位")
        sys.exit(0)

    print(f"[缺失] {len(missing)} 个数据文件尚未下载:")
    for f in missing:
        print(f"   - {f}")

    print("\n尝试使用 Kaggle API 下载...")
    if not download_with_kaggle_api():
        print("\n备选方案: 手动下载")
        print(f"  1. 访问 https://www.kaggle.com/competitions/{COMPETITION}/data")
        print(f"  2. 下载所有 CSV 文件")
        print(f"  3. 将文件放到 {os.path.abspath(RAW_DIR)}/ 目录下")
