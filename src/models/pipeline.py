"""NLP 向量化 + 聚类分析流水线"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import os

DATA_DIR = 'data/processed'
OUT_DIR = 'outputs'


def haversine_distance(p1, p2):
    """计算两点间的球面距离 (归一化到 [0,1])"""
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371 * c


def joint_distance_matrix(geo_features, text_vectors, alpha=0.5):
    """联合距离矩阵: D_joint = alpha * D_geo_norm + (1-alpha) * (1 - Sim_text)"""
    n = geo_features.shape[0]

    # 地理距离矩阵 (Haversine)
    D_geo = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_distance(geo_features[i], geo_features[j])
            D_geo[i, j] = d
            D_geo[j, i] = d
    # 归一化
    if D_geo.max() > 0:
        D_geo = D_geo / D_geo.max()

    # 文本相似度矩阵 (余弦相似度)
    from sklearn.metrics.pairwise import cosine_similarity
    if text_vectors.shape[1] > 0:
        Sim_text = cosine_similarity(text_vectors)
        D_text = 1 - Sim_text  # 距离 = 1 - 相似度
    else:
        D_text = np.eye(n)

    # 联合距离
    D_joint = alpha * D_geo + (1 - alpha) * D_text
    return D_joint


def run_experiment(name, feature_matrix, text_vectors, alphas, k_range=(3, 12)):
    """运行一组聚类实验"""
    results = []

    # 提取地理特征 (lat_norm, lon_norm)
    geo_cols = ['lat_norm', 'lon_norm']
    geo = feature_matrix[geo_cols].values

    for alpha in alphas:
        D = joint_distance_matrix(geo, text_vectors, alpha=alpha)

        # K-Means (需要欧式空间，用多维缩放近似)
        from sklearn.manifold import MDS
        # 用 MDS 把距离矩阵映射到欧式空间
        try:
            mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42, max_iter=300)
            coords = mds.fit_transform(D)
        except:
            continue

        for k in range(k_range[0], k_range[1] + 1):
            for cluster_algo in ['kmeans', 'agglomerative']:
                try:
                    if cluster_algo == 'kmeans':
                        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
                    else:
                        labels = AgglomerativeClustering(
                            n_clusters=k, metric='precomputed', linkage='average'
                        ).fit_predict(D)

                    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_labels < 2:
                        continue

                    sil = silhouette_score(D, labels, metric='precomputed')
                    # DBI on coords
                    try:
                        dbi = davies_bouldin_score(coords, labels)
                    except:
                        dbi = np.nan
                    try:
                        ch = calinski_harabasz_score(coords, labels)
                    except:
                        ch = np.nan

                    results.append({
                        'experiment': name, 'alpha': alpha, 'k': k,
                        'algorithm': cluster_algo, 'silhouette': sil,
                        'dbi': dbi, 'ch_index': ch, 'n_clusters_found': n_labels,
                    })
                except Exception as e:
                    pass

    return pd.DataFrame(results)


def main():
    print('=' * 60)
    print('  NLP VECTORIZATION + CLUSTERING PIPELINE')
    print('=' * 60)

    # 1. 加载特征矩阵
    fm = pd.read_csv(f'{DATA_DIR}/feature_matrix.csv')
    print(f'\n[1] Loaded feature matrix: {fm.shape}')

    # 2. NLP 向量化
    print('\n[2] NLP Vectorization...')
    texts = fm['review_text'].fillna('').values

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_vectors = tfidf.fit_transform(texts).toarray()
    print(f'  TF-IDF:   {tfidf_vectors.shape}')

    # 标准化
    scaler = StandardScaler()
    tfidf_norm = scaler.fit_transform(tfidf_vectors)

    # 3. Baseline: 纯地理聚类
    print('\n[3] Running experiments...')
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 纯地理 baseline (alpha=1.0)
    geo_only = np.ones((len(fm), 1))  # 纯地理不需要文本向量
    D_pure_geo = joint_distance_matrix(fm[['lat_norm', 'lon_norm']].values, geo_only, alpha=1.0)

    # 纯文本 baseline (alpha=0.0)
    D_pure_text = joint_distance_matrix(fm[['lat_norm', 'lon_norm']].values, tfidf_norm, alpha=0.0)

    print('  Running TF-IDF experiments...')
    results_tfidf = run_experiment('TF-IDF', fm, tfidf_norm, alphas)
    print(f'  TF-IDF: {len(results_tfidf)} results')

    # 4. 找最优配置
    all_results = results_tfidf
    if len(all_results) > 0:
        best = all_results.loc[all_results['silhouette'].idxmax()]
        print(f'\n[4] Best configuration:')
        print(f'  Alpha: {best["alpha"]}, K: {best["k"]:.0f}, Algo: {best["algorithm"]}')
        print(f'  Silhouette: {best["silhouette"]:.4f}')
        print(f'  DBI: {best["dbi"]:.4f}')

    # 5. 保存结果
    os.makedirs(f'{OUT_DIR}/results', exist_ok=True)
    all_results.to_csv(f'{OUT_DIR}/results/experiment_results.csv', index=False)
    print(f'\n[5] Results saved to {OUT_DIR}/results/experiment_results.csv')

    # 6. 用最优配置生成最终聚类标签
    optimal_alpha = best['alpha'] if len(all_results) > 0 else 0.5
    optimal_k = int(best['k']) if len(all_results) > 0 else 6

    D_final = joint_distance_matrix(fm[['lat_norm', 'lon_norm']].values, tfidf_norm, alpha=optimal_alpha)
    from sklearn.manifold import MDS
    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42, max_iter=300)
    coords_final = mds.fit_transform(D_final)
    labels_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(coords_final)

    fm['cluster'] = labels_final
    fm.to_csv(f'{DATA_DIR}/feature_matrix_clustered.csv', index=False, encoding='utf-8')
    print(f'  Clustered data saved: {len(fm)} POIs × {len(fm.columns)} features')

    # 聚类分布
    print(f'\n[6] Cluster distribution:')
    for c in sorted(fm['cluster'].unique()):
        n = (fm['cluster'] == c).sum()
        sample_names = fm[fm['cluster'] == c]['name'].dropna().head(5).tolist()
        print(f'  Cluster {c}: {n:3d} POIs  e.g. {sample_names[:3]}')

    return fm


if __name__ == '__main__':
    main()
