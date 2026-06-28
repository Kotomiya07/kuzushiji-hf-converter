# kotenseki-dataset

Hugging Face Datasets形式で、ページ単位 dataset と文字クロップ単位 dataset を生成するスクリプト。

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

`--export-format hf` では `--dataset-type` で出力対象を切り替えられます。

- `--dataset-type both` : ページ単位と文字単位の両方を生成
- `--dataset-type page` : ページ単位だけを生成
- `--dataset-type character` : 文字単位だけを生成

- ページ単位 dataset: `output/kuzushiji-dataset-{bbox形式}/`
- 文字単位 dataset: `output/kuzushiji-dataset-characters/`

文字単位 dataset の画像は `raw/*/characters/` を使わず、`raw/*/images/*.jpg` と `*_coordinate.csv` の bbox から毎回新規クロップします。画像サイズはリサイズせず、クロップ後の原寸をそのまま保持します。

### Roboflow 向け YOLOv8 形式で書き出し

```bash
uv run python convert_dataset.py \
  --export-format roboflow \
  --raw-dir ./raw \
  --column-annotations-dir ./output \
  --segment-annotations-dir ./output_seg
```

Roboflow 出力では、列アノテーションが1件以上ある画像だけを対象に、単一クラス `column` の YOLOv8 検出データセットを書き出します。  
出力先は `output/kuzushiji-dataset-roboflow-yolov8-columns/` で、`train/images/`, `train/labels/`, `data.yaml` を生成します。

### Hugging Face Hubへのアップロード

```bash
# Hubにプッシュ
uv run python convert_dataset.py --bbox-format coco --push-to-hub --hub-username your_username

# character dataset だけをHubにプッシュ
uv run python convert_dataset.py --dataset-type character --push-to-hub --hub-username your_username
```

`hf auth login` 済みなら、`HF_TOKEN` を明示しなくてもローカル保存済みトークンを自動利用します。明示したい場合は `--hub-token` または `HF_TOKEN` も使えます。
`--hub-username` を省略した場合は、ログイン中ユーザー名を自動解決して `{username}/{dataset_name}` 形式で push します。

### 変換結果の bbox 可視化

変換後の Dataset を使って、ページ画像上に以下の bbox を重ね描きして確認できます。

- 文字 bbox: 青
- 列 bbox: 緑
- セグメント bbox / polygon: 赤（半透明塗りつぶしあり）

```bash
# ローカル保存した Dataset を可視化
uv run --project . python visualize_dataset_annotations.py ./output/kuzushiji-dataset-coco --split train --max-samples 10

# YOLO 形式を明示して可視化
uv run --project . python visualize_dataset_annotations.py ./output/kuzushiji-dataset-yolo --split train --bbox-format yolo --max-samples 10

# Hugging Face Hub 上の dataset を直接可視化
uv run --project . python visualize_dataset_annotations.py your-username/kuzushiji-dataset-coco --split train --max-samples 10
```

デフォルトでは `output/visualizations/` に `00000_{image_id}.png` の形式で保存されます。
`--bbox-format auto` では、bbox 値がすべて 0-1 に収まる場合は YOLO、それ以外は COCO とみなして描画します。
`segments.polygon` / `segments.polygons` / `segments.segmentation` が存在する場合は、セグメント領域も合わせて描画します。

### テスト実行（1ディレクトリのみ）

```bash
uv run python convert_dataset.py --bbox-format coco --raw-dir ./raw --dry-run
```

## CLI引数

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--export-format` | 出力形式（hf, roboflow） | hf |
| `--dataset-type` | HF出力対象（page, character, both） | both |
| `--bbox-format` | bbox形式（coco, yolo） | coco |
| `--dataset-name-prefix` | データセット名のプレフィックス | kuzushiji-dataset |
| `--raw-dir` | 生データのディレクトリ | ./raw |
| `--output-dir` | 出力ディレクトリ | ./output |
| `--column-annotations-dir` | 列アノテーションCSVの親ディレクトリ | ./output |
| `--segment-annotations-dir` | セグメントアノテーションCSVの親ディレクトリ | ./output_seg |
| `--pua-metadata-path` | アノテーターの `pua_characters.json` へのパス（未指定時は自動探索） | None |
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

### ページ単位 dataset

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
        "is_pua": Sequence(Value("bool")),          # category が私用領域コードか
        "pua_code": Sequence(Value("string")),      # PUAコード（例: U+E000、通常文字は空文字）
        "pua_reading": Sequence(Value("string")),   # pua_metadata.json 由来の読み
        "pua_memo": Sequence(Value("string")),      # pua_metadata.json 由来のメモ
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

### 文字単位 dataset

```python
features = Features({
    "image": Image(),                # クロップ済み文字画像
    "source_image_id": Value("string"),
    "book_id": Value("string"),
    "char_id": Value("string"),
    "block_id": Value("string"),
    "category": Value("string"),     # Unicode文字列（例: U+3042）
    "category_id": Value("int32"),
    "is_pua": Value("bool"),         # category が私用領域コードか
    "pua_code": Value("string"),     # PUAコード（例: U+E000、通常文字は空文字）
    "pua_reading": Value("string"),  # pua_metadata.json 由来の読み
    "pua_memo": Value("string"),     # pua_metadata.json 由来のメモ
    "char": Value("string"),         # 実際の文字（例: あ）
    "bbox": Sequence(Value("int32"), length=4),       # 元ページ上の bbox [x, y, w, h]
    "crop_bbox": Sequence(Value("int32"), length=4),  # 実際に使ったクロップ bbox [x, y, w, h]
    "width": Value("int32"),
    "height": Value("int32"),
})
```

## 出力ファイル

- `output/label2id.json` - Unicode → ID マッピング
- `output/id2label.json` - ID → Unicode マッピング
- `output/pua_metadata.json` - PUAコード → 読み・メモのマッピング（メタデータが見つかった場合のみ）
- `output/kuzushiji-dataset-{形式}/` - ローカル保存時のデータセット
- `output/kuzushiji-dataset-characters/` - 文字単位のクロップ画像データセット
- `output/kuzushiji-dataset-roboflow-yolov8-columns/` - Roboflow 向け YOLOv8 データセット

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
│   └── characters/          # 既存切り出し文字画像（本スクリプトでは使用しない）

output/
└── {ID}/
    └── column_annotation.csv

output_seg/
└── {ID}/
    └── column_annotation.csv
```

## ライセンス

CC BY-SA 4.0
