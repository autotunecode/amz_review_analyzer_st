# Amazon Review Scraper and Analyzer

## 概要

このプログラムは、指定された ASIN コードに対応する Amazon 商品のレビューをスクレイピングし、感情分析を行います。結果はデータフレームとして表示され、CSV ファイルとしてダウンロードすることも可能です。
使用技術

- Python
- Streamlit
- Selenium
- BeautifulSoup
- Transformers

## 機能

レビュースクレイピング
指定された ASIN コードに対応する Amazon 商品のレビューをスクレイピングします。最大 2 ページまでスクレイピングします。
感情分析
スクレイピングしたレビューテキストに対して感情分析を行います。感情分析には Transformers ライブラリの sentiment-analysis パイプラインを使用します。
結果表示
スクレイピングと感情分析の結果をデータフレームとして表示します。データフレームは感情スコアでソートされます。
CSV ダウンロード
分析結果を CSV ファイルとしてダウンロードすることができます。

## 使用方法

1. Streamlit アプリケーションを起動します。
2. 'Enter ASIN Code'フィールドに ASIN コードを入力します。
3. 'Run Scraper and Analyzer'ボタンをクリックします。
4. 処理が完了すると、結果がデータフレームとして表示されます。
5. 'Download CSV file'リンクをクリックすると、結果を CSV ファイルとしてダウンロードできます。

## 注意事項

- ASIN コードは Amazon 商品ページの URL に含まれる一意の識別子です。
- このプログラムは感情分析のために cl-tohoku/bert-base-japanese-whole-word-masking モデルを使用しています。このモデルは日本語テキストの感情分析に最適化されています。
- スクレイピングは Amazon の利用規約に違反する可能性があります。このプログラムを使用する際は自己責任で行ってください。

### アプリケーションのセットアップ

このアプリケーションは Python の`transformers`ライブラリを使用しています。そのため、`transformers`ライブラリが正しくインストールされていることを確認してください。

### ライブラリのインストール

以下のコマンドを実行して、必要なライブラリをインストールまたはアップデートしてください：

```bash
pip install --upgrade transformers

これにより、最新バージョンの`transformers`ライブラリがインストールされ、アプリケーション内で`pipeline`関数を正しくインポートできるようになります。
```
