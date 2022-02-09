# 簡単な説明です
公開する予定がなかったので見づらいコードになっているかと思います。申し訳ありません。
## 学習用データ収集について
まず、yahoo様の株価データを収集させていただきます。そのままだとスクレイピングできないので、一度htmlをコピーしてローカルサーバーにミラーさせてからそれをスクレイピングする形をとります。  
自分の場合はXAMPPというソフトウェアを使用してローカルにサーバーを立てました。  
yahoo様の株価サイトはurl内の企業コードだけ変更すれば各企業を見られるようになっているので、それを利用します。  
`codelist.js`には企業コードの一覧が入っています。(こちらのスクレイピング用のプログラムは一度しか使わなかったので消してしまいました…)  
`test.html`を実行することで株価データのスクレイピングを行います。ページには検索ボックスが表示されますが、特に意味はありません。検索ボタンが開始ボタンになります。Enterキーでも開始できます。`test.js`での処理は以下の通りです。 
 
1.企業コード配列から企業コードを抽出  
2.その企業の株価データページをミラーリング  
3.必要なデータを収集  
4.次の企業コードで繰り返す  
5.抽出したデータからcsvを作成し、ダウンロード  

こうしてダウンロードしたものが`data_*.csv`となります。(実際に行ったのは1年ほど前なので、データは2021年4月ごろのものになります)

## データの学習について
データは20日分取ってあるので、20日前~1日前までのデータから当日のデータ、つまり前日比がプラスかマイナスかを推論させるようにして学習を行います。学習用のアルゴリズムとして`Adam`と呼ばれる手法を用いていますが、詳細は、オライリー社「ゼロから作るDeep Learning」を参考にしました。こちらに関してはアルゴリズムを見直して改良していく必要があります。