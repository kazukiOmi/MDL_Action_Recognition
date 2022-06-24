# 時空間アダプタを用いた動作認識のためのマルチドメイン学習

動作認識モデルのマルチドメイン学習を行うコード

## コード

- datasets/dataset.py: 全てのデータセットのロードを管理するファイル
- datasets/hmdb51.py: HMDB51を読み込むためのファイル
- datasets/multiview.py: データセットをマルチビューでロードするためのファイル
- models/model.py: 複数ドメインを処理するモデルの構築
- train.py: マルチドメイン学習を実行
- main.py: 実行ファイル
- plot.py: 実行結果をプロット

## 準備
[comet](https://www.comet-ml.com/docs/)でログを残すためにアカウント作成
以下のコマンドでcometのapiをホームディレクトリ以下に置けばデフォルトでそれを参照する
```bash
comet init --api-key
```

環境は[こちら](https://hub.docker.com/r/tttamaki/docker-ssh/)のDockerイメージからコンテナ作成
必要に応じて以下のコマンドで実行環境をそろえる
```bash
pip install -r requirements.txt
```

## 実行
```bash
python main.py -m "train"
               -i num_itarasions
               -dn dataset_name_list
               -bsl batch_size_list
```
#### オプション
- `-m`：モデルを学習させる(`train`)か評価するか(`val`)を選択．マルチビューで評価するなら`multiview_val`
- `-i`：イテレーション数
- `-dn`：使用するデータセットのリスト
- `-bsl`：それぞれのデータセットのバッチサイズ
- `-lr`：初期学習率
- `-ap`：アダプタを入れる位置
  - `stages`:ResStage間
  - `blocks`：ResBlock間
  - `all`：ResStage間とResBlock間
  - `No`：アダプタなし
- `--adp_num`：`-ap`で`stages`を選んだ場合のみアダプタを何個入れるか指定
- `-adp_pos`：`-ap`で`stages`を選んだ場合のみResStageのどこにアダプタ入れるか指定
  - `top`:出力層側のResStageから`-adp_num`個
  - `bottom`：入力層側のResStageから`-adp_num`個
- `--fix_shared_params`：ドメイン非依存パラメータを固定するオプション
- `-ex_name`：実験名（モデルの保存場所とcometのログを紐づけるため，また評価時はここで指定したモデルの評価が行われる）


## 学会発表
[SSII2022](https://confit.atlas.jp/guide/event/ssii2022/subject/IS1-02/category?cryptoId=)

![SSII2022_omi_poster-1](https://user-images.githubusercontent.com/80406150/175548611-646b8328-db7c-478e-b43b-40ea7fb4c40e.png)