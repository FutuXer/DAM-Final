"""
TripAdvisor CN 日本景点中文评论爬虫
- 服务器端渲染，直接解析 HTML
- 高质量中文评论（长篇、含评分、日期、旅游类型）
- 覆盖东京/京都/大阪核心景点
"""

import requests, re, json, time
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 核心日本景点列表（TripAdvisor CN 的 g(地理ID)-d(景点ID)）
ATTRACTIONS = {
    # 京都
    '清水寺':      'g298564-d321405',
    '伏见稻荷':    'g14135148-d1380174',
    '金阁寺':      'g298564-d321400',
    '岚山':        'g298564-d1169613',
    '二条城':      'g298564-d321403',
    '银阁寺':      'g298564-d321401',
    '祇园':        'g298564-d321406',
    '京都御所':    'g298564-d319863',
    '南禅寺':      'g298564-d321402',
    '三十三间堂':  'g298564-d321404',

    # 大阪
    '大阪城':      'g14135149-d1380173',
    '道顿堀':      'g298566-d546423',
    '环球影城':    'g298566-d324657',
    '心斋桥':      'g298566-d546422',
    '通天阁':      'g298566-d546424',
    '黑门市场':    'g298566-d3171158',

    # 东京
    '浅草寺':      'g14133691-d1380175',
    '东京塔':      'g14133692-d1380172',
    '明治神宫':    'g1066443-d320296',
    '上野公园':    'g14133692-d555022',
    '秋叶原':      'g1066443-d320311',
    '涩谷':        'g1066443-d320315',
    '新宿御苑':    'g1066443-d320316',
    '筑地市场':    'g1066443-d320310',

    # 奈良/神户
    '东大寺':      'g298561-d324062',
    '奈良公园':    'g298561-d319862',
    '神户港':      'g298562-d324387',
}


def get_reviews(gid, did, max_pages=5):
    """获取单个景点的中文评论"""
    all_reviews = []
    base_url = f'https://www.tripadvisor.cn/Attraction_Review-{gid}-{did}-Reviews'

    for page in range(max_pages):
        # TripAdvisor 分页: orXX- 表示 offset
        if page == 0:
            url = f'{base_url}.html'
        else:
            offset = page * 10
            url = f'{base_url}-or{offset}.html'

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break

            soup = BeautifulSoup(r.text, 'html.parser')
            cards = soup.select('[data-automation=\"reviewCard\"]')
            if not cards:
                break

            for card in cards:
                text = card.get_text(strip=True)
                review = {'gid': gid, 'did': did, 'page': page}

                # 提取旅游类型
                type_match = re.search(r'(情侣游|家庭|独自旅行|商务|好友|情侣|夫妻|朋友)', text)
                review['trip_type'] = type_match.group(1) if type_match else ''

                # 提取日期
                date_match = re.search(r'撰写日期[：:]?\s*(\d{4}-\d{2}-\d{2})', text)
                review['date'] = date_match.group(1) if date_match else ''

                # 提取评分（bubble 数字）
                rating_match = re.search(r'bubble_(\d+)', str(card))
                review['rating'] = int(rating_match.group(1)) // 10 if rating_match else None

                # 评论文本（去掉元数据后的部分）
                # 去掉 "Tripadvisor用户发布于:海外" 等前缀
                text_clean = re.sub(r'Tripadvisor[^:]*[:：]', '', text)
                text_clean = re.sub(r'撰写日期[：:]?\d{4}-\d{2}-\d{2}', '', text_clean)
                text_clean = re.sub(r'此点评为.*?观点[。.]?', '', text_clean)
                text_clean = re.sub(r'阅读更多.*?收起全文', '', text_clean)
                text_clean = re.sub(r'(情侣游|家庭|独自旅行|商务|好友|情侣|夫妻|朋友|海外)', '', text_clean)
                text_clean = text_clean.strip()
                review['text'] = text_clean

                if len(text_clean) > 30:
                    all_reviews.append(review)

            print(f'  Page {page+1}: {len(all_reviews)} reviews total')
            time.sleep(1.5)  # 慢一点，避免限流

        except Exception as e:
            print(f'  Page {page+1} error: {e}')
            break

    return all_reviews


def main():
    print(f'TripAdvisor CN Scraper | {len(ATTRACTIONS)} attractions | {datetime.now():%H:%M:%S}')
    print()

    all_reviews = []
    for i, (name, code) in enumerate(ATTRACTIONS.items()):
        gid, did = code.split('-')
        print(f'[{i+1}/{len(ATTRACTIONS)}] {name} ({code})')
        reviews = get_reviews(gid, did, max_pages=5)
        for r in reviews:
            r['attraction'] = name
        all_reviews.extend(reviews)
        print(f'  Total: {len(reviews)} reviews for {name}')

        # 每 5 个景点保存一次
        if (i + 1) % 5 == 0:
            pd.DataFrame(all_reviews).to_csv(
                'data/raw/tripadvisor_cn_reviews.csv', index=False, encoding='utf-8')
            print(f'  [Saved {len(all_reviews)} reviews so far]')

    # 最终保存
    df = pd.DataFrame(all_reviews)
    df.to_csv('data/raw/tripadvisor_cn_reviews.csv', index=False, encoding='utf-8')
    print(f'\nDone: {len(df)} reviews from {df[\"attraction\"].nunique()} attractions')
    print(f'Avg text length: {df[\"text\"].str.len().mean():.0f} chars')
    return df


if __name__ == '__main__':
    main()
