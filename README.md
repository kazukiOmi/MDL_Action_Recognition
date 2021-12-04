# 202109_omi_multi_domain

動作認識モデルのマルチドメイン学習を行うコード

# コード

- datasets/dataset.py: データセットのローダーを作成
- models/model.py: 複数ドメインを処理するモデルの構築
- train.py: マルチドメイン学習を実行
- main.py: 実行ファイル

# 実行

```bash
python main.py --iteration num_itarasions
               --batch_size_list batch_size_list
               --num_workers num_workers
               --pretrained True 
               --adp_place stages
               --adp_mode
               --dataset_names dataset_name_list
```
