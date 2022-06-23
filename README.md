# 202109_omi_multi_domain

動作認識モデルのマルチドメイン学習を行うコード

## コード

- datasets/dataset.py: 全てのデータセットのロードを管理するファイル
- datasets/hmdb51.py: HMDB51を読み込むためのファイル
- datasets/multiview.py: データセットをマルチビューでロードするためのファイル
- models/model.py: 複数ドメインを処理するモデルの構築
- train.py: マルチドメイン学習を実行
- main.py: 実行ファイル
- plot.py: 実行結果をプロット

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
- `-lr`：学習率
- `-ap`：アダプタを入れる位置
  - `stages`:ResStage間
  - `blocks`：ResBlock間
  - `all`：ResStage間とResBlock間
  - `No`：アダプタなし
- `--adp_num`：`-ap`で`stages`を選んだ場合のみアダプタを何個入れるか指定
- `-adp_pos`：`-ap`で`stages`を選んだ場合のみResStageのどこにアダプタ入れるか指定
  - `top`:出力層側のResStageから`-adp_num`個
  - `bottom`：入力層側のResStageから`-adp_num`個
- `-ex_name`：実験名（モデルの保存場所とcometのログを紐づけるため）
