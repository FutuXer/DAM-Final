"""Folium 交互式聚类地图"""
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os

def generate_map():
    fm = pd.read_csv('data/processed/feature_matrix_clustered.csv')

    # 日本中心
    m = folium.Map(location=[35.68, 139.76], zoom_start=7, tiles='CartoDB positron')

    # 配色方案
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#ffff33', '#a65628', '#f781bf', '#66c2a5', '#fc8d62']

    # 为每个簇创建图层
    clusters = sorted(fm['cluster'].unique())
    for c in clusters:
        cluster_data = fm[fm['cluster'] == c]
        color = colors[c % len(colors)] if c >= 0 else '#999999'

        fg = folium.FeatureGroup(
            name=f'Cluster {c} ({len(cluster_data)} POIs)' if c >= 0 else f'Noise ({len(cluster_data)} POIs)'
        )

        for _, row in cluster_data.iterrows():
            # 弹窗内容
            name = str(row.get('name', 'N/A'))[:80]
            fclass = row.get('fclass', 'N/A')
            review_n = int(row.get('review_count', 0))

            popup_html = f"""
            <b>{name}</b><br>
            Category: {fclass}<br>
            Reviews: {review_n}<br>
            Cluster: {c}
            """

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4 + review_n * 0.05,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg)

        fg.add_to(m)

    # 车站叠加层（只显示东京+京都的主要车站）
    try:
        stations = pd.read_csv('data/raw/stations_raw.csv')
        stations = stations.dropna(subset=['lat', 'lon'])
        # 筛选关东+关西主要车站
        mask = ((stations['lat'] > 34.5) & (stations['lat'] < 36.5) &
                (stations['lon'] > 135) & (stations['lon'] < 140))
        major = stations[mask].iloc[::10]  # 每10个取1个，避免太密

        st_fg = folium.FeatureGroup(name='Railway Stations', show=False)
        for _, s in major.iterrows():
            folium.CircleMarker(
                location=[s['lat'], s['lon']],
                radius=2, color='#333333', fill=True, fill_opacity=0.5,
            ).add_to(st_fg)
        st_fg.add_to(m)
    except:
        pass

    folium.LayerControl().add_to(m)

    os.makedirs('outputs/maps', exist_ok=True)
    m.save('outputs/maps/cluster_map.html')
    print(f'Map saved: outputs/maps/cluster_map.html')
    print(f'Clusters: {len(clusters)}, Total POIs: {len(fm)}')

    return m


if __name__ == '__main__':
    generate_map()
