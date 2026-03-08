# kotenseki-dataset

Hugging Face Datasets形式でくずし字画像とアノテーション（文字 bbox / 列 bbox / セグメント bbox）を提供するスクリプト。

## セットアップ

```bash
uv sync
```

## 使用方法

### 基本的な使用法（ローカル保存）

```bash
# COCO形式（デフォルト）
uv run python convert_dataset.py --bbox-format coco --raw-dir ./raw --column-annotations-dir ./output --segment-annotations-dir ./output_seg

# YOLO形式（正規化済み）
uv run python convert_dataset.py --bbox-format yolo --raw-dir ./raw --column-annotations-dir ./output --segment-annotations-dir ./output_seg
```

### Hugging Face Hubへのアップロード

```bash
# Hubにプッシュ
uv run python convert_dataset.py --bbox-format coco --push-to-hub --hub-username your_username
```

`hf auth login` 済みなら、`HF_TOKEN` を明示しなくてもローカル保存済みトークンを自動利用します。明示したい場合は `--hub-token` または `HF_TOKEN` も使えます。
`--hub-username` を省略した場合は、ログイン中ユーザー名を自動解決して `{username}/{dataset_name}` 形式で push します。

### テスト実行（1ディレクトリのみ）

```bash
uv run python convert_dataset.py --bbox-format coco --raw-dir ./raw --dry-run
```

## CLI引数

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--bbox-format` | bbox形式（coco, yolo） | coco |
| `--dataset-name-prefix` | データセット名のプレフィックス | kuzushiji-dataset |
| `--raw-dir` | 生データのディレクトリ | ./raw |
| `--output-dir` | 出力ディレクトリ | ./output |
| `--column-annotations-dir` | 列アノテーションCSVの親ディレクトリ | ./output |
| `--segment-annotations-dir` | セグメントアノテーションCSVの親ディレクトリ | ./output_seg |
| `--push-to-hub` | Hugging Face Hubにプッシュ | False |
| `--hub-token` | Hubトークン（未指定時は HF_TOKEN → `hf auth login` 保存トークンの順で参照） | None |
| `--hub-username` | Hubユーザー名/組織名 | None |
| `--max-shard-size` | Parquetシャードサイズ | 500MB |
| `--dry-run` | 最初の1ディレクトリのみ処理 | False |

## bbox形式

| 形式 | フォーマット | 正規化 |
|------|--------------|--------|
| COCO | [x_min, y_min, width, height] | なし（ピクセル値） |
| YOLO | [x_center, y_center, width, height] | 0-1に正規化 |

## データセット構造

```python
features = Features({
    "image": Image(),                    # 画像データ
    "image_id": Value("string"),         # 画像ID（例: 100241706_00004_2）
    "book_id": Value("string"),          # 書籍ID（例: 100241706）
    "width": Value("int32"),             # 画像幅
    "height": Value("int32"),            # 画像高さ
    "objects": {
        "bbox": Sequence(Sequence(Value("float32"), length=4)),
        "category": Sequence(Value("string")),      # Unicode文字列（例: U+3042）
        "category_id": Sequence(Value("int32")),    # カテゴリID
        "char": Sequence(Value("string")),          # 実際の文字（例: あ）
    },
    "columns": {
        "bbox": Sequence(Sequence(Value("float32"), length=4)),
        "column_id": Sequence(Value("string")),
        "char_ids": Sequence(Sequence(Value("string"))),
        "segment_id": Sequence(Value("string")),
    },
    "segments": {
        "bbox": Sequence(Sequence(Value("float32"), length=4)),
        "segment_id": Sequence(Value("string")),
        "column_ids": Sequence(Sequence(Value("string"))),
    },
})
```

## 出力ファイル

- `output/label2id.json` - Unicode → ID マッピング
- `output/id2label.json` - ID → Unicode マッピング
- `output/kuzushiji-dataset-{形式}/` - ローカル保存時のデータセット

## データソース

本データセットは「**日本古典籍くずし字データセット**」を基にしています。

- **提供**: 国文学研究資料館ほか所蔵
- **加工**: ROIS-DS 人文学オープンデータ共同利用センター（CODH）
- **DOI**: [10.20676/00000340](https://doi.org/10.20676/00000340)
- **Website**: [https://codh.rois.ac.jp/char-shape/](https://codh.rois.ac.jp/char-shape/)

### 引用

```
『日本古典籍くずし字データセット』（国文研ほか所蔵／CODH加工）doi:10.20676/00000340
```

## raw/ディレクトリ構造

`raw/` ディレクトリには以下の構造でデータが格納されています：

```
raw/
├── {ID}/ (複数ディレクトリ)
│   ├── {ID}_coordinate.csv  # Unicode, Image, X, Y, Block ID, Char ID, Width, Height
│   ├── images/              # ページ画像 (*.jpg)
│   └── characters/          # 切り出し文字画像（本スクリプトでは未使用）

output/
└── {ID}/
    └── column_annotation.csv

output_seg/
└── {ID}/
    └── column_annotation.csv
```

## ライセンス

CC BY-SA 4.0
