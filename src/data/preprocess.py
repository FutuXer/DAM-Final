"""数据预处理流水线：清洗 → 对齐 → 特征拼接 → 输出特征宽表"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/raw'
OUT_DIR = 'data/processed'


def load_poi_data():
    """加载并清洗 POI 数据"""
    print('[1/5] Loading POI data...')
    df = pd.read_csv(f'{DATA_DIR}/pois_raw.csv')

    # 去重
    df = df.drop_duplicates(subset=['osm_id'])

    # 只保留旅游相关类别
    target_classes = ['restaurant', 'cafe', 'hotel', 'museum', 'attraction',
                      'viewpoint', 'park', 'guest_house', 'hostel', 'motel',
                      'artwork', 'monument', 'archaeological_site', 'memorial',
                      'wayside_shrine', 'theme_park', 'zoo', 'aquarium', 'gallery']
    df = df[df['fclass'].isin(target_classes)].copy()

    # 过滤无效坐标
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] > 26) & (df['lat'] < 46) & (df['lon'] > 127) & (df['lon'] < 146)]

    # 坐标归一化
    scaler = MinMaxScaler()
    df['lat_norm'] = scaler.fit_transform(df[['lat']])
    df['lon_norm'] = scaler.fit_transform(df[['lon']])

    # 类别编码
    df['fclass_encoded'] = df['fclass'].astype('category').cat.codes

    print(f'  POI cleaned: {len(df):,} records ({df["fclass"].nunique()} categories)')
    return df


def load_reviews():
    """加载评论数据并与 POI 对齐"""
    print('[2/5] Loading review data...')

    # TripAdvisor CN 中文评论
    ta_cn = pd.read_csv(f'{DATA_DIR}/tripadvisor_cn_reviews.csv')
    print(f'  TA-CN: {len(ta_cn):,} reviews')

    # HuggingFace 日本景点数据（有坐标）
    hf_jp = pd.read_csv(f'{DATA_DIR}/reviews_japan.csv')
    print(f'  HF-JP: {len(hf_jp):,} records')

    # 从 HF 数据构建 attraction_code → 坐标 映射表
    # URL 中含 gXXXXXX-dXXXXXX 编码
    code_to_coords = {}
    for _, row in hf_jp.iterrows():
        url = str(row.get('url', ''))
        m = re.search(r'(g\d+-d\d+)', url)
        if m and pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            code = m.group(1)
            if code not in code_to_coords:
                code_to_coords[code] = {
                    'lat': row['lat'], 'lon': row['lon'],
                    'name': row.get('name', ''), 'rating': row.get('rating', 0),
                }
    print(f'  Code→Coord mapping: {len(code_to_coords)} attractions')

    # 给 TA-CN 评论关联坐标
    ta_cn['lat'] = ta_cn['attraction_code'].map(
        lambda c: code_to_coords.get(c, {}).get('lat', np.nan))
    ta_cn['lon'] = ta_cn['attraction_code'].map(
        lambda c: code_to_coords.get(c, {}).get('lon', np.nan))
    ta_cn['poi_name'] = ta_cn['attraction_code'].map(
        lambda c: code_to_coords.get(c, {}).get('name', ''))

    # 有坐标的比例
    has_coords = ta_cn['lat'].notna().sum()
    print(f'  TA-CN with coords: {has_coords}/{len(ta_cn)} ({has_coords/len(ta_cn)*100:.1f}%)')

    # 聚合：每个 POI 的所有评论拼接
    poi_reviews = ta_cn.groupby('attraction_code').agg({
        'text': lambda x: ' '.join(x.dropna()),
        'lat': 'first',
        'lon': 'first',
        'poi_name': 'first',
    }).reset_index()
    poi_reviews['review_count'] = ta_cn.groupby('attraction_code').size().values
    poi_reviews['avg_text_len'] = ta_cn.groupby('attraction_code')['text_len'].mean().values
    poi_reviews = poi_reviews.dropna(subset=['lat', 'lon'])

    print(f'  POI-level reviews: {len(poi_reviews)} POIs with coords')
    return poi_reviews, ta_cn


def compute_transit_features(poi_df):
    """计算交通可达性特征"""
    print('[3/5] Computing transit features...')
    stations = pd.read_csv(f'{DATA_DIR}/stations_raw.csv')
    stations = stations.dropna(subset=['lat', 'lon'])

    from scipy.spatial import cKDTree

    # 为每个 POI 找最近车站
    station_coords = np.radians(stations[['lat', 'lon']].values)
    poi_coords = np.radians(poi_df[['lat', 'lon']].values)

    tree = cKDTree(station_coords)
    distances_rad, indices = tree.query(poi_coords, k=1)

    # Haversine 距离 (km)
    poi_df['nearest_station_km'] = distances_rad * 6371

    # 1km 内车站数量
    counts = tree.query_ball_point(poi_coords, r=1.0/6371.0)  # 1km in radians
    poi_df['stations_within_1km'] = [len(c) for c in counts]

    print(f'  Avg nearest station: {poi_df["nearest_station_km"].mean():.2f} km')
    print(f'  Avg stations in 1km: {poi_df["stations_within_1km"].mean():.2f}')
    return poi_df


def merge_demographics(poi_df):
    """关联区域统计数据"""
    print('[4/5] Merging demographics...')
    census = pd.read_csv(f'{DATA_DIR}/census_raw.csv')

    # 根据纬度判断所属地区（粗略映射）
    # 北海道 41-46, 东北 38-41, 关东 35-37, 中部 34-37, 近畿 33-35, 中国 33-35, 四国 32-34, 九州 26-34
    region_map = {
        'Hokkaido': (41, 46), 'Aomori': (40, 41.5), 'Iwate': (38.5, 40.5),
        'Miyagi': (37.5, 39), 'Akita': (39, 41), 'Fukushima': (36.5, 38),
        'Tokyo': (35.4, 36.1), 'Kanagawa': (35.1, 35.7), 'Chiba': (35, 36),
        'Saitama': (35.7, 36.2), 'Aichi': (34.7, 35.5), 'Osaka': (34.3, 35),
        'Kyoto': (34.7, 35.5), 'Hyogo': (34.3, 35.5), 'Nara': (34.3, 34.9),
        'Hiroshima': (34, 35), 'Fukuoka': (33, 34.3), 'Okinawa': (26, 27.5),
    }

    # 简化：用 nearest prefecture by lat
    pref_list = census['prefecture'].tolist()
    pref_stats = census.set_index('prefecture').to_dict('index')

    # 只为 demo 目的：随机分配人口密度（实际项目中应该用行政边界精确匹配）
    # 这里用区域特征为每个 POI 添加
    for pref, stats in pref_stats.items():
        mask = None
        if pref in region_map:
            lo, hi = region_map[pref]
            mask = (poi_df['lat'] >= lo) & (poi_df['lat'] < hi)
        if mask is not None and mask.sum() > 0:
            poi_df.loc[mask, 'pop_density'] = stats['pop_density_km2']
            poi_df.loc[mask, 'tourism_index'] = stats['tourism_index']

    # 填充缺失值
    poi_df['pop_density'] = poi_df['pop_density'].fillna(poi_df['pop_density'].median())
    poi_df['tourism_index'] = poi_df['tourism_index'].fillna(poi_df['tourism_index'].median())

    print(f'  Pop density range: [{poi_df["pop_density"].min():.0f}, {poi_df["pop_density"].max():.0f}]')
    print(f'  Tourism index range: [{poi_df["tourism_index"].min():.0f}, {poi_df["tourism_index"].max():.0f}]')
    return poi_df


def build_feature_matrix():
    """主流程：构建四源特征宽表"""
    # 1. POI 基础数据
    poi = load_poi_data()

    # 2. 加载评论并与 POI 做空间匹配
    poi_reviews, _ = load_reviews()

    # L0 空间匹配：评论 ↔ 最近 POI (50m 阈值)
    from scipy.spatial import cKDTree
    poi_coords = np.radians(poi[['lat', 'lon']].values)
    review_coords = np.radians(poi_reviews[['lat', 'lon']].values)

    tree = cKDTree(poi_coords)
    distances_rad, indices = tree.query(review_coords, k=1)
    distances_km = distances_rad * 6371

    # 建立映射：每个 POI 对应的评论
    poi['review_text'] = ''
    poi['review_count'] = 0
    poi['avg_rating'] = 0.0

    matched = 0
    for i, (dist, idx) in enumerate(zip(distances_km, indices)):
        if dist < 0.5:  # 500m 阈值——城市POI的合理匹配范围
            matched += 1
            pid = idx
            poi.at[poi.index[pid], 'review_text'] = \
                str(poi.at[poi.index[pid], 'review_text']) + ' ' + str(poi_reviews.iloc[i]['text'])
            poi.at[poi.index[pid], 'review_count'] += 1

    print(f'  Spatial match: {matched}/{len(poi_reviews)} reviews matched to POIs ({matched/len(poi_reviews)*100:.0f}%)')

    # 只保留有评论的 POI
    poi_with_reviews = poi[poi['review_count'] > 0].copy()
    print(f'  POIs with reviews: {len(poi_with_reviews)}')

    # 3. 交通特征
    poi_with_reviews = compute_transit_features(poi_with_reviews)

    # 4. 人口统计特征
    poi_with_reviews = merge_demographics(poi_with_reviews)

    # 5. 输出特征宽表
    feature_cols = [
        'osm_id', 'name', 'fclass', 'fclass_encoded',
        'lat', 'lon', 'lat_norm', 'lon_norm',
        'review_text', 'review_count', 'avg_rating',
        'nearest_station_km', 'stations_within_1km',
        'pop_density', 'tourism_index',
    ]
    available = [c for c in feature_cols if c in poi_with_reviews.columns]
    feature_matrix = poi_with_reviews[available].copy()

    # 归一化数值特征
    num_cols = ['nearest_station_km', 'stations_within_1km', 'pop_density', 'tourism_index', 'review_count']
    for col in num_cols:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())
            feature_matrix[f'{col}_norm'] = MinMaxScaler().fit_transform(feature_matrix[[col]])

    # 保存
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    feature_matrix.to_csv(f'{OUT_DIR}/feature_matrix.csv', index=False, encoding='utf-8')

    print(f'\n[5/5] Feature matrix saved!')
    print(f'  Shape: {feature_matrix.shape}')
    print(f'  POIs with reviews: {len(feature_matrix)}')
    print(f'  Features: {list(feature_matrix.columns)}')

    return feature_matrix


if __name__ == '__main__':
    fm = build_feature_matrix()
