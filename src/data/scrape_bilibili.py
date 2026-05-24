"""B站日本旅游评论爬虫 —— 补充中文评论数据源 v2 (高容错版)"""

import requests, pandas as pd, time, os
from datetime import datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.bilibili.com/',
}

SEARCH_KEYWORDS = [
    '京都旅游','大阪旅游','东京旅游','奈良旅游',
    '北海道旅游','福冈旅游','名古屋旅游','神户旅游',
    '京都美食','大阪美食','东京美食','日本美食探店',
    '清水寺','大阪城','东京塔','富士山','浅草寺','环球影城',
    '日本自由行','日本旅行攻略','日本vlog','关西旅游',
    '京都vlog','大阪vlog','东京vlog','日本旅游攻略',
    '京都旅行','大阪旅行','东京旅行',
    '日本探店','日本打卡','日本小众',
    '关西攻略','京都攻略','大阪攻略',
    '箱根','镰仓','北海道旅游攻略',
]

OUTPUT_DIR = 'data/raw/bilibili'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_videos(keyword, max_pages=3):
    videos = []
    for page in range(1, max_pages + 1):
        ok = False
        for _ in range(3):
            try:
                r = requests.get(
                    'https://api.bilibili.com/x/web-interface/search/type',
                    params={'search_type': 'video', 'keyword': keyword, 'page': page, 'order': 'click'},
                    headers=HEADERS, timeout=30)
                data = r.json()
                if data.get('code') == 0:
                    for v in data['data'].get('result', []):
                        videos.append({
                            'bvid': v.get('bvid'), 'aid': v.get('aid'),
                            'title': v.get('title', '').replace('<em class="keyword">','').replace('</em>',''),
                            'play': v.get('play', 0), 'search_keyword': keyword,
                        })
                    ok = True
                break
            except:
                time.sleep(2)
        if not ok:
            break
        time.sleep(0.8)
    return videos


def get_comments(oid, max_pages=5):
    comments = []
    for pn in range(1, max_pages + 1):
        ok = False
        replies = None
        for _ in range(2):
            try:
                r = requests.get('https://api.bilibili.com/x/v2/reply',
                    params={'type': 1, 'oid': oid, 'pn': pn, 'sort': 1},
                    headers=HEADERS, timeout=30)
                data = r.json()
                if data.get('code') == 0:
                    replies = data['data'].get('replies')
                    if replies:
                        for reply in replies:
                            comments.append({
                                'oid': oid, 'rpid': reply.get('rpid'),
                                'message': reply.get('content', {}).get('message', ''),
                                'like': reply.get('like', 0),
                                'ctime': reply.get('ctime', 0),
                            })
                    ok = True
                break
            except:
                time.sleep(1)
        if not ok or not replies or len(replies) < 20:
            break
        time.sleep(0.5)
    return comments


def main():
    print(f'Bilibili Scraper v2 | {len(SEARCH_KEYWORDS)} keywords | {datetime.now():%H:%M:%S}')

    # 1. Search
    all_videos, seen = [], set()
    for ki, kw in enumerate(SEARCH_KEYWORDS):
        videos = search_videos(kw)
        new = [v for v in videos if v['aid'] not in seen]
        for v in new:
            seen.add(v['aid'])
        all_videos.extend(new)
        if (ki + 1) % 10 == 0:
            print(f'  Search {ki+1}/{len(SEARCH_KEYWORDS)}: {len(all_videos)} unique videos')
        pd.DataFrame(all_videos).to_csv(f'{OUTPUT_DIR}/videos.csv', index=False, encoding='utf-8')

    print(f'Videos: {len(all_videos)}')

    # 2. Comments (resume if exists)
    cf = f'{OUTPUT_DIR}/comments.csv'
    comments, done = [], set()
    if os.path.exists(cf):
        existing = pd.read_csv(cf)
        comments = existing.to_dict('records')
        done = set(existing['oid'].unique())
        print(f'Resume: {len(done)} videos done, {len(comments)} comments')

    for vi, v in enumerate(all_videos):
        if v['aid'] in done:
            continue
        batch = get_comments(v['aid'])
        for c in batch:
            c['bvid'] = v['bvid']; c['video_title'] = v['title']; c['search_keyword'] = v['search_keyword']
        comments.extend(batch)
        if (vi + 1) % 100 == 0:
            print(f'  [{vi+1}/{len(all_videos)}] {len(comments)} comments')
            pd.DataFrame(comments).to_csv(cf, index=False, encoding='utf-8')

    # 3. Save
    df = pd.DataFrame(comments)
    df.to_csv(cf, index=False, encoding='utf-8')
    print(f'Done: {len(df)} comments from {df["oid"].nunique()} videos')
    return df


if __name__ == '__main__':
    main()
