import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager, ChromeType
# Chromium　起動用
from selenium.webdriver.common.by import By
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.chrome import ChromeDriverManager

import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
import pandas as pd
import time
import base64

def run_scraping_and_analysis(ASIN):
    review_data = []
    for page in range(1, 3):  # 最大2ページまでスクレイピング
        url = f'https://www.amazon.co.jp/product-reviews/{ASIN}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={page}'

        # 　ヘッドレスモードでブラウザを起動
        webdriver_path = ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        s = Service(webdriver_path)
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        # ブラウザーを起動
        driver = webdriver.Chrome(service=s, options=options)

        driver.get(url)
        time.sleep(3)  # 読み込み待ち
        driver.implicitly_wait(3)  # 見つからないときは、3秒まで待つ

        # ページをスクロールして追加の商品を読み込む
        for k in range(2):
            driver.execute_script(
                'window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(3)
            
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        # レビュー要素をすべて取得
        review_elements = soup.find_all('div', {'data-hook': 'review'})

        # レビュー要素がなければループを終了
        if not review_elements:
            break
        # 感情分析のパイプラインを作成
        sentiment_analysis = pipeline('sentiment-analysis', model="cl-tohoku/bert-base-japanese-whole-word-masking")
        # 各レビュー要素からテキストと評価を取得し、感情分析を実行
        for index, element in enumerate(review_elements, start=1):
            review_text = element.find('span', {'data-hook': 'review-body'}).text.strip()
            review_rating = float(element.find('i', {'data-hook': 'review-star-rating'}).text.strip()[-3:])
            review_text = re.sub(r'[^\w\s。、]', '', review_text, flags=re.UNICODE)
            review_sentiment = sentiment_analysis(review_text)[0]

            # レビューデータを辞書に格納し、リストに追加
            review_data.append({
                'index': len(review_data) + 1,  # 'index'が重複しないように修正
                'body': review_text,
                'rating': review_rating,
                'sentiment': round(review_sentiment['score'], 4)
            })

    # データフレームを作成し、感情スコアでソート
    df = pd.DataFrame(review_data)
    df.sort_values(by='sentiment', ascending=False, inplace=True)
    return df

def main():
    st.title('Amazon Review Scraper and Analyzer')
    ASIN = st.text_input('Enter ASIN Code', '')
    if st.button('Run Scraper and Analyzer') and ASIN:
        with st.spinner('Processing...'):
            df = run_scraping_and_analysis(ASIN)
        st.write(df.to_html(index=False), unsafe_allow_html=True)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # ここで必要ないくつかの文字列 <-> バイトの変換
        href = f'<a href="data:file/csv;base64,{b64}" download="reviews.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()