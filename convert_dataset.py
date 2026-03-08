# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets", "Pillow", "pandas", "huggingface_hub"]
# ///
"""Hugging Face Datasets アップロードスクリプト.

raw/ ディレクトリのページ画像とアノテーション（バウンディングボックス）を
Hugging Face Datasets にアップロードする。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import DatasetCard, DatasetCardData
from PIL import Image as PILImage


BboxFormat = Literal["coco", "yolo"]


@dataclass
class CharAnnotation:
    """文字アノテーション."""

    unicode: str  # U+XXXX形式
    x: int
    y: int
    width: int
    height: int
    block_id: str
    char_id: str


@dataclass
class ImageAnnotation:
    """画像のアノテーション情報."""

    image_id: str
    book_id: str
    image_path: Path
    width: int
    height: int
    characters: list[CharAnnotation]
    columns: list[ColumnAnnotation]
    segments: list[SegmentAnnotation]


@dataclass
class ColumnAnnotation:
    """列アノテーション."""

    column_id: str
    bbox: list[float]
    char_ids: list[str]
    segment_id: str


@dataclass
class SegmentAnnotation:
    """セグメントアノテーション."""

    segment_id: str
    bbox: list[float]
    column_ids: list[str]


def parse_unicode_to_char(unicode_str: str) -> str:
    """Unicode文字列（U+XXXX）を実際の文字に変換する."""
    if unicode_str.startswith("U+"):
        code_point = int(unicode_str[2:], 16)
        return chr(code_point)
    return unicode_str


def convert_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    img_width: int,
    img_height: int,
    bbox_format: BboxFormat,
) -> list[float]:
    """バウンディングボックスを指定形式に変換する.

    Args:
        x: 左上X座標（ピクセル）
        y: 左上Y座標（ピクセル）
        width: 幅（ピクセル）
        height: 高さ（ピクセル）
        img_width: 画像幅
        img_height: 画像高さ
        bbox_format: 出力形式（coco, yolo）

    Returns:
        変換されたバウンディングボックス [4要素のリスト]
    """
    if bbox_format == "coco":
        # [x_min, y_min, width, height]
        return [float(x), float(y), float(width), float(height)]
    elif bbox_format == "yolo":
        # [x_center, y_center, width, height] 正規化
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        return [x_center, y_center, norm_width, norm_height]
    else:
        msg = f"Unknown bbox format: {bbox_format}"
        raise ValueError(msg)


def parse_sort_key(value: str) -> tuple[str, int]:
    """ID 末尾の数値をソートキーに変換する."""
    matched = re.search(r"(\d+)$", value)
    if matched is None:
        return value, 0
    return value, int(matched.group(1))


def build_bbox_from_frame(
    df: pd.DataFrame,
    bbox_format: BboxFormat,
    img_width: int,
    img_height: int,
) -> list[float]:
    """DataFrame から外接矩形 bbox を作る."""
    x_min = int(df["X"].min())
    y_min = int(df["Y"].min())
    x_max = int((df["X"] + df["Width"]).max())
    y_max = int((df["Y"] + df["Height"]).max())
    return convert_bbox(
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min,
        img_width,
        img_height,
        bbox_format,
    )


def scan_raw_directory(raw_dir: Path) -> Iterator[tuple[str, Path, Path]]:
    """rawディレクトリをスキャンし、各書籍のCSVと画像ディレクトリを返す.

    Yields:
        (book_id, csv_path, images_dir)
    """
    for book_dir in sorted(raw_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        book_id = book_dir.name
        csv_path = book_dir / f"{book_id}_coordinate.csv"
        images_dir = book_dir / "images"

        if csv_path.exists() and images_dir.exists():
            yield book_id, csv_path, images_dir


def load_column_segment_annotations(
    book_id: str,
    column_dir: Path | None,
    segment_dir: Path | None,
    bbox_format: BboxFormat,
) -> dict[str, tuple[list[ColumnAnnotation], list[SegmentAnnotation]]]:
    """列/セグメントアノテーションをページ単位で読み込む."""
    page_map: dict[str, tuple[list[ColumnAnnotation], list[SegmentAnnotation]]] = {}

    column_csv = None if column_dir is None else column_dir / book_id / "column_annotation.csv"
    segment_csv = None if segment_dir is None else segment_dir / book_id / "column_annotation.csv"

    column_df = (
        pd.read_csv(column_csv)
        if column_csv is not None and column_csv.exists()
        else pd.DataFrame()
    )
    segment_df = (
        pd.read_csv(segment_csv)
        if segment_csv is not None and segment_csv.exists()
        else pd.DataFrame()
    )

    page_ids = sorted(
        {
            str(image_id)
            for image_id in column_df.get("Image", pd.Series(dtype="string")).dropna().tolist()
        }
        | {
            str(image_id)
            for image_id in segment_df.get("Image", pd.Series(dtype="string")).dropna().tolist()
        }
    )

    for page_id in page_ids:
        source_df = segment_df[segment_df["Image"] == page_id].copy() if not segment_df.empty else pd.DataFrame()
        if source_df.empty and not column_df.empty:
            source_df = column_df[column_df["Image"] == page_id].copy()
        if source_df.empty:
            continue

        img_width = int(source_df["X"].add(source_df["Width"]).max())
        img_height = int(source_df["Y"].add(source_df["Height"]).max())

        columns: list[ColumnAnnotation] = []
        if "Column ID" in source_df.columns:
            work_columns = source_df.dropna(subset=["Column ID"]).copy()
            for column_id, group in work_columns.groupby("Column ID", sort=False):
                segment_id = ""
                if "Segment ID" in group.columns:
                    segment_ids = [
                        str(value).strip()
                        for value in group["Segment ID"].dropna().tolist()
                        if str(value).strip()
                    ]
                    segment_id = segment_ids[0] if segment_ids else ""
                char_ids = sorted(
                    [str(char_id) for char_id in group["Char ID"].dropna().tolist()],
                    key=parse_sort_key,
                )
                columns.append(
                    ColumnAnnotation(
                        column_id=str(column_id),
                        bbox=build_bbox_from_frame(group, bbox_format, img_width, img_height),
                        char_ids=char_ids,
                        segment_id=segment_id,
                    )
                )
        columns.sort(key=lambda ann: parse_sort_key(ann.column_id))

        segments: list[SegmentAnnotation] = []
        if "Segment ID" in source_df.columns:
            work_segments = source_df.dropna(subset=["Segment ID"]).copy()
            for segment_id, group in work_segments.groupby("Segment ID", sort=False):
                column_ids = sorted(
                    {
                        str(column_id)
                        for column_id in group["Column ID"].dropna().tolist()
                        if str(column_id).strip()
                    },
                    key=parse_sort_key,
                )
                segments.append(
                    SegmentAnnotation(
                        segment_id=str(segment_id),
                        bbox=build_bbox_from_frame(group, bbox_format, img_width, img_height),
                        column_ids=column_ids,
                    )
                )
        segments.sort(key=lambda ann: parse_sort_key(ann.segment_id))

        page_map[page_id] = (columns, segments)

    return page_map


def load_annotations(
    csv_path: Path,
    images_dir: Path,
    book_id: str,
    bbox_format: BboxFormat,
    column_dir: Path | None,
    segment_dir: Path | None,
) -> list[ImageAnnotation]:
    """CSVファイルからアノテーションを読み込む."""
    df = pd.read_csv(csv_path)
    page_level_annotations = load_column_segment_annotations(
        book_id,
        column_dir,
        segment_dir,
        bbox_format,
    )

    # 画像ごとにグループ化
    image_annotations: dict[str, ImageAnnotation] = {}

    for _, row in df.iterrows():
        image_name = row["Image"]
        image_path = images_dir / f"{image_name}.jpg"

        if not image_path.exists():
            continue

        if image_name not in image_annotations:
            # 画像サイズを取得
            with PILImage.open(image_path) as img:
                img_width, img_height = img.size

            image_annotations[image_name] = ImageAnnotation(
                image_id=image_name,
                book_id=book_id,
                image_path=image_path,
                width=img_width,
                height=img_height,
                characters=[],
                columns=[],
                segments=[],
            )

        char_ann = CharAnnotation(
            unicode=row["Unicode"],
            x=int(row["X"]),
            y=int(row["Y"]),
            width=int(row["Width"]),
            height=int(row["Height"]),
            block_id=row["Block ID"],
            char_id=row["Char ID"],
        )
        image_annotations[image_name].characters.append(char_ann)

    for image_name, annotation in image_annotations.items():
        columns, segments = page_level_annotations.get(image_name, ([], []))
        annotation.columns = columns
        annotation.segments = segments

    return list(image_annotations.values())


def build_category_mapping(
    all_annotations: list[ImageAnnotation],
) -> tuple[dict[str, int], dict[int, str]]:
    """Unicode文字列からカテゴリIDへのマッピングを構築する."""
    unique_unicodes: set[str] = set()
    for ann in all_annotations:
        for char in ann.characters:
            unique_unicodes.add(char.unicode)

    sorted_unicodes = sorted(unique_unicodes)
    label2id = {unicode: idx for idx, unicode in enumerate(sorted_unicodes)}
    id2label = {idx: unicode for unicode, idx in label2id.items()}

    return label2id, id2label


def generate_dataset_records(
    annotations: list[ImageAnnotation],
    label2id: dict[str, int],
    bbox_format: BboxFormat,
) -> Iterator[dict[str, Any]]:
    """アノテーションをデータセット形式に変換するジェネレータ.

    メモリ効率のため、1レコードずつyieldする。
    """
    for ann in annotations:
        # 画像データを読み込み（1枚ずつ）
        image_bytes = ann.image_path.read_bytes()

        objects = {
            "bbox": [],
            "category": [],
            "category_id": [],
            "char": [],
        }
        columns = {
            "bbox": [],
            "column_id": [],
            "char_ids": [],
            "segment_id": [],
        }
        segments = {
            "bbox": [],
            "segment_id": [],
            "column_ids": [],
        }

        for char in ann.characters:
            bbox = convert_bbox(
                char.x,
                char.y,
                char.width,
                char.height,
                ann.width,
                ann.height,
                bbox_format,
            )
            objects["bbox"].append(bbox)
            objects["category"].append(char.unicode)
            objects["category_id"].append(label2id[char.unicode])
            objects["char"].append(parse_unicode_to_char(char.unicode))

        for column in ann.columns:
            columns["bbox"].append(column.bbox)
            columns["column_id"].append(column.column_id)
            columns["char_ids"].append(column.char_ids)
            columns["segment_id"].append(column.segment_id)

        for segment in ann.segments:
            segments["bbox"].append(segment.bbox)
            segments["segment_id"].append(segment.segment_id)
            segments["column_ids"].append(segment.column_ids)

        yield {
            "image": {"bytes": image_bytes, "path": ann.image_path.name},
            "image_id": ann.image_id,
            "book_id": ann.book_id,
            "width": ann.width,
            "height": ann.height,
            "objects": objects,
            "columns": columns,
            "segments": segments,
        }


def create_dataset_features() -> Features:
    """データセットのフィーチャー定義を作成する."""
    return Features(
        {
            "image": Image(),
            "image_id": Value("string"),
            "book_id": Value("string"),
            "width": Value("int32"),
            "height": Value("int32"),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"), length=4)),
                "category": Sequence(Value("string")),
                "category_id": Sequence(Value("int32")),
                "char": Sequence(Value("string")),
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
        }
    )


def save_label_mapping(
    label2id: dict[str, int],
    id2label: dict[int, str],
    output_dir: Path,
) -> None:
    """ラベルマッピングをJSONファイルに保存する."""
    output_dir.mkdir(parents=True, exist_ok=True)

    label2id_path = output_dir / "label2id.json"
    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    id2label_path = output_dir / "id2label.json"
    # JSONのキーは文字列でなければならない
    id2label_str_keys = {str(k): v for k, v in id2label.items()}
    with open(id2label_path, "w", encoding="utf-8") as f:
        json.dump(id2label_str_keys, f, ensure_ascii=False, indent=2)

    print(f"Saved label mappings to {output_dir}")


def create_dataset_card(
    bbox_format: BboxFormat,
    num_images: int,
    num_books: int,
    num_categories: int,
) -> DatasetCard:
    """データセットカードを作成する."""
    bbox_descriptions = {
        "coco": "[x_min, y_min, width, height] (pixels)",
        "yolo": "[x_center, y_center, width, height] (normalized 0-1)",
    }

    card_data = DatasetCardData(
        language=["ja"],
        license="cc-by-sa-4.0",
        task_categories=["object-detection"],
        tags=["kuzushiji", "japanese", "historical-documents", "ocr", bbox_format],
        size_categories=["1K<n<10K"] if num_images < 10000 else ["10K<n<100K"],
    )

    content = f"""---
{card_data.to_yaml()}
---

# Kuzushiji Dataset ({bbox_format.upper()} format)

This dataset contains page images from historical Japanese documents (Kotenseki)
with character-level bounding box annotations for Kuzushiji (cursive Japanese) recognition.

## Dataset Description

- **Number of images**: {num_images:,}
- **Number of books**: {num_books}
- **Number of character categories**: {num_categories:,}
- **Bounding box format**: {bbox_descriptions[bbox_format]}
- **Additional annotations**: optional column / segment bounding boxes

## Dataset Structure

```python
{{
    "image": Image(),                    # Page image
    "image_id": str,                     # Image ID (e.g., 100241706_00004_2)
    "book_id": str,                      # Book ID (e.g., 100241706)
    "width": int,                        # Image width in pixels
    "height": int,                       # Image height in pixels
    "objects": {{
        "bbox": List[List[float]],       # Bounding boxes ({bbox_descriptions[bbox_format]})
        "category": List[str],           # Unicode strings (e.g., U+3042)
        "category_id": List[int],        # Category IDs
        "char": List[str],               # Actual characters (e.g., あ)
    }},
    "columns": {{
        "bbox": List[List[float]],       # Column boxes
        "column_id": List[str],          # Column IDs (e.g., COL0001)
        "char_ids": List[List[str]],     # Member Char IDs
        "segment_id": List[str],         # Parent Segment IDs if available
    }},
    "segments": {{
        "bbox": List[List[float]],       # Segment boxes
        "segment_id": List[str],         # Segment IDs (e.g., SEG0001)
        "column_ids": List[List[str]],   # Member Column IDs
    }}
}}
```

## Bounding Box Formats

| Format | Description | Normalized |
|--------|-------------|------------|
| COCO | [x_min, y_min, width, height] | No (pixels) |
| YOLO | [x_center, y_center, width, height] | Yes (0-1) |

## Usage

```python
from datasets import load_dataset
import json

dataset = load_dataset("your-username/kuzushiji-dataset-{bbox_format}")

# Access first example
example = dataset["train"][0]
print(f"Image ID: {{example['image_id']}}")
print(f"Number of characters: {{len(example['objects']['bbox'])}}")

# Load label mappings
from huggingface_hub import hf_hub_download

label2id_path = hf_hub_download(
    repo_id="your-username/kuzushiji-dataset-{bbox_format}",
    filename="label2id.json",
    repo_type="dataset"
)
with open(label2id_path) as f:
    label2id = json.load(f)

print(f"Number of categories: {{len(label2id)}}")
```

## Label Mappings

This dataset includes the following mapping files:
- `label2id.json`: Unicode string (e.g., "U+3042") to category ID mapping
- `id2label.json`: Category ID to Unicode string mapping

## License

This dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Data Source

This dataset is derived from the **日本古典籍くずし字データセット (Japanese Historical Character Dataset)**.

- **Provider**: National Institute of Japanese Literature and other institutions (国文学研究資料館ほか所蔵)
- **Processing**: ROIS-DS Center for Open Data in the Humanities (CODH)
- **DOI**: [10.20676/00000340](https://doi.org/10.20676/00000340)
- **Website**: [https://codh.rois.ac.jp/char-shape/](https://codh.rois.ac.jp/char-shape/)

## Citation

If you use this dataset, please cite:

```
『日本古典籍くずし字データセット』（国文研ほか所蔵／CODH加工）doi:10.20676/00000340
```

English:
```
"Japanese Historical Character Dataset" (Owned by NIJL and others, Processed by CODH) doi:10.20676/00000340
```

## Acknowledgments

Data provided by: ROIS-DS Center for Open Data in the Humanities (人文学オープンデータ共同利用センター)
"""
    return DatasetCard(content)


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Convert raw data to Hugging Face Dataset format"
    )
    parser.add_argument(
        "--bbox-format",
        type=str,
        choices=["coco", "yolo"],
        default="coco",
        help="Bounding box format (default: coco)",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        type=str,
        default="kuzushiji-dataset",
        help="Dataset name prefix (default: kuzushiji-dataset)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./raw"),
        help="Raw data directory (default: ./raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory for local files (default: ./output)",
    )
    parser.add_argument(
        "--column-annotations-dir",
        type=Path,
        default=Path("./output"),
        help="Directory containing per-book column_annotation.csv files for columns",
    )
    parser.add_argument(
        "--segment-annotations-dir",
        type=Path,
        default=Path("./output_seg"),
        help="Directory containing per-book column_annotation.csv files for segments",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face Hub token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hub-username",
        type=str,
        default=None,
        help="Hugging Face Hub username/organization",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="500MB",
        help="Maximum shard size (default: 500MB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: only process first directory",
    )

    args = parser.parse_args()

    bbox_format: BboxFormat = args.bbox_format
    raw_dir: Path = args.raw_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    column_annotations_dir = args.column_annotations_dir.resolve()
    segment_annotations_dir = args.segment_annotations_dir.resolve()

    if not raw_dir.exists():
        print(f"Error: Raw directory not found: {raw_dir}")
        return

    # トークン取得
    hub_token = args.hub_token or os.environ.get("HF_TOKEN")

    print(f"Scanning raw directory: {raw_dir}")
    print(f"Bbox format: {bbox_format}")

    # 全アノテーションを収集
    all_annotations: list[ImageAnnotation] = []
    book_count = 0

    for book_id, csv_path, images_dir in scan_raw_directory(raw_dir):
        print(f"Processing book: {book_id}")
        annotations = load_annotations(
            csv_path,
            images_dir,
            book_id,
            bbox_format,
            column_annotations_dir,
            segment_annotations_dir,
        )
        all_annotations.extend(annotations)
        book_count += 1

        if args.dry_run:
            print("Dry run: stopping after first book")
            break

    print(f"Processed {book_count} books, {len(all_annotations)} images")

    if not all_annotations:
        print("No annotations found!")
        return

    # カテゴリマッピング構築
    print("Building category mapping...")
    label2id, id2label = build_category_mapping(all_annotations)
    print(f"Found {len(label2id)} unique characters")

    # ラベルマッピング保存
    save_label_mapping(label2id, id2label, output_dir)

    # データセット作成（ジェネレータを使用してメモリ効率化）
    print("Creating dataset with generator...")
    features = create_dataset_features()

    def gen():
        yield from generate_dataset_records(all_annotations, label2id, bbox_format)

    dataset = Dataset.from_generator(gen, features=features)

    print(f"Dataset created: {dataset}")
    print(f"Sample record keys: {list(dataset[0].keys())}")

    # Hubにプッシュ
    if args.push_to_hub:
        dataset_name = f"{args.dataset_name_prefix}-{bbox_format}"

        if args.hub_username:
            repo_id = f"{args.hub_username}/{dataset_name}"
        else:
            repo_id = dataset_name

        print(f"Pushing to Hub: {repo_id}")
        dataset.push_to_hub(
            repo_id,
            token=hub_token,
            max_shard_size=args.max_shard_size,
        )

        # label2id.json と id2label.json をHubにアップロード
        from huggingface_hub import HfApi

        api = HfApi()
        print("Uploading label mappings...")

        # label2id.json
        label2id_path = output_dir / "label2id.json"
        api.upload_file(
            path_or_fileobj=str(label2id_path),
            path_in_repo="label2id.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=hub_token,
        )

        # id2label.json
        id2label_path = output_dir / "id2label.json"
        api.upload_file(
            path_or_fileobj=str(id2label_path),
            path_in_repo="id2label.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=hub_token,
        )

        # データセットカードをアップロード
        print("Uploading dataset card...")
        card = create_dataset_card(
            bbox_format=bbox_format,
            num_images=len(all_annotations),
            num_books=book_count,
            num_categories=len(label2id),
        )
        card.push_to_hub(repo_id, token=hub_token)

        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
    else:
        # ローカルに保存
        local_path = output_dir / f"{args.dataset_name_prefix}-{bbox_format}"
        print(f"Saving dataset locally: {local_path}")
        dataset.save_to_disk(str(local_path))
        print(f"Dataset saved to: {local_path}")


if __name__ == "__main__":
    main()
