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
import io
import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, get_token
from PIL import Image as PILImage


BboxFormat = Literal["coco", "yolo"]
ExportFormat = Literal["hf", "roboflow"]
DatasetType = Literal["page", "character", "both"]
PuaMetadata = dict[str, dict[str, str]]
PUA_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


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
    is_pua: bool = False
    pua_code: str = ""
    pua_reading: str = ""
    pua_memo: str = ""


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


def unicode_codepoint(unicode_str: str) -> int | None:
    """U+XXXX形式の文字列をコードポイントに変換する."""
    if not unicode_str.startswith("U+"):
        return None
    try:
        return int(unicode_str[2:], 16)
    except ValueError:
        return None


def is_pua_unicode(unicode_str: str) -> bool:
    """Unicode文字列が私用領域のコードポイントか判定する."""
    code_point = unicode_codepoint(unicode_str)
    if code_point is None:
        return False
    return any(start <= code_point <= end for start, end in PUA_RANGES)


def build_pua_fields(unicode_str: str, pua_metadata: PuaMetadata) -> dict[str, str | bool]:
    """Unicode文字列からPUA補助フィールドを作る."""
    if not is_pua_unicode(unicode_str):
        return {
            "is_pua": False,
            "pua_code": "",
            "pua_reading": "",
            "pua_memo": "",
        }

    metadata = pua_metadata.get(unicode_str, {})
    return {
        "is_pua": True,
        "pua_code": unicode_str,
        "pua_reading": metadata.get("reading", ""),
        "pua_memo": metadata.get("memo", ""),
    }


def load_pua_metadata(metadata_path: Path | None) -> PuaMetadata:
    """アノテーターの pua_characters.json から読み・メモを読み込む."""
    if metadata_path is None or not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    characters = data.get("pua_characters", {})
    if not isinstance(characters, dict):
        return {}

    metadata: PuaMetadata = {}
    for code, info in characters.items():
        if not isinstance(code, str) or not isinstance(info, dict):
            continue
        metadata[code] = {
            "reading": str(info.get("reading", "")),
            "memo": str(info.get("memo", "")),
        }
    return metadata


def load_pua_metadata_files(metadata_paths: list[Path]) -> PuaMetadata:
    """複数のPUAメタデータファイルを読み込んでマージする."""
    merged: PuaMetadata = {}
    for metadata_path in metadata_paths:
        merged.update(load_pua_metadata(metadata_path))
    return merged


def resolve_pua_metadata_paths(raw_dir: Path, metadata_path: Path | None) -> list[Path]:
    """PUAメタデータファイルのパス候補を解決する."""
    if metadata_path is not None:
        resolved = metadata_path.resolve()
        return [resolved] if resolved.exists() else []

    candidates = [
        raw_dir.parent.parent / "kotenseki-annotator-web" / "pua_characters.json",
        raw_dir / "pua_characters.json",
        raw_dir.parent / "pua_characters.json",
    ]
    resolved_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in seen:
                resolved_paths.append(resolved)
                seen.add(resolved)
    return resolved_paths


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


def infer_annotation_extent(df: pd.DataFrame) -> tuple[int, int]:
    """注釈の広がりから暫定画像サイズを推定する."""
    img_width = int(df["X"].add(df["Width"]).max())
    img_height = int(df["Y"].add(df["Height"]).max())
    return img_width, img_height


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
    image_sizes: dict[str, tuple[int, int]] | None = None,
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

        img_width, img_height = (
            image_sizes[page_id]
            if image_sizes is not None and page_id in image_sizes
            else infer_annotation_extent(source_df)
        )

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
    pua_metadata: PuaMetadata | None = None,
) -> list[ImageAnnotation]:
    """CSVファイルからアノテーションを読み込む."""
    df = pd.read_csv(csv_path)
    pua_metadata = pua_metadata or {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_name in sorted({str(value) for value in df["Image"].dropna().tolist()}):
        image_path = images_dir / f"{image_name}.jpg"
        if not image_path.exists():
            continue
        with PILImage.open(image_path) as img:
            image_sizes[image_name] = img.size

    page_level_annotations = load_column_segment_annotations(
        book_id,
        column_dir,
        segment_dir,
        bbox_format,
        image_sizes=image_sizes,
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

        pua_fields = build_pua_fields(str(row["Unicode"]), pua_metadata)
        char_ann = CharAnnotation(
            unicode=row["Unicode"],
            x=int(row["X"]),
            y=int(row["Y"]),
            width=int(row["Width"]),
            height=int(row["Height"]),
            block_id=row["Block ID"],
            char_id=row["Char ID"],
            is_pua=bool(pua_fields["is_pua"]),
            pua_code=str(pua_fields["pua_code"]),
            pua_reading=str(pua_fields["pua_reading"]),
            pua_memo=str(pua_fields["pua_memo"]),
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
            "is_pua": [],
            "pua_code": [],
            "pua_reading": [],
            "pua_memo": [],
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
            objects["is_pua"].append(char.is_pua)
            objects["pua_code"].append(char.pua_code)
            objects["pua_reading"].append(char.pua_reading)
            objects["pua_memo"].append(char.pua_memo)
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


def clamp_bbox_to_image(
    x: int,
    y: int,
    width: int,
    height: int,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """画像境界内に bbox を収める."""
    left = max(0, min(x, img_width))
    top = max(0, min(y, img_height))
    right = max(left, min(x + width, img_width))
    bottom = max(top, min(y + height, img_height))
    return left, top, right, bottom


def encode_pil_image_to_png_bytes(image: PILImage.Image) -> bytes:
    """PIL Image を PNG bytes に変換する."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_character_dataset_records(
    annotations: list[ImageAnnotation],
    label2id: dict[str, int],
) -> Iterator[dict[str, Any]]:
    """文字単位のクロップ画像データセットを生成する."""
    for ann in annotations:
        with PILImage.open(ann.image_path) as page_image:
            for char in ann.characters:
                left, top, right, bottom = clamp_bbox_to_image(
                    char.x,
                    char.y,
                    char.width,
                    char.height,
                    ann.width,
                    ann.height,
                )
                cropped = page_image.crop((left, top, right, bottom))
                crop_width = right - left
                crop_height = bottom - top
                crop_filename = f"{ann.image_id}_{char.char_id}.png"

                yield {
                    "image": {
                        "bytes": encode_pil_image_to_png_bytes(cropped),
                        "path": crop_filename,
                    },
                    "source_image_id": ann.image_id,
                    "book_id": ann.book_id,
                    "char_id": char.char_id,
                    "block_id": str(char.block_id),
                    "category": char.unicode,
                    "category_id": label2id[char.unicode],
                    "is_pua": char.is_pua,
                    "pua_code": char.pua_code,
                    "pua_reading": char.pua_reading,
                    "pua_memo": char.pua_memo,
                    "char": parse_unicode_to_char(char.unicode),
                    "bbox": [char.x, char.y, char.width, char.height],
                    "crop_bbox": [left, top, crop_width, crop_height],
                    "width": crop_width,
                    "height": crop_height,
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
                "is_pua": Sequence(Value("bool")),
                "pua_code": Sequence(Value("string")),
                "pua_reading": Sequence(Value("string")),
                "pua_memo": Sequence(Value("string")),
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


def create_character_dataset_features() -> Features:
    """文字単位データセットのフィーチャー定義を作成する."""
    return Features(
        {
            "image": Image(),
            "source_image_id": Value("string"),
            "book_id": Value("string"),
            "char_id": Value("string"),
            "block_id": Value("string"),
            "category": Value("string"),
            "category_id": Value("int32"),
            "is_pua": Value("bool"),
            "pua_code": Value("string"),
            "pua_reading": Value("string"),
            "pua_memo": Value("string"),
            "char": Value("string"),
            "bbox": Sequence(Value("int32"), length=4),
            "crop_bbox": Sequence(Value("int32"), length=4),
            "width": Value("int32"),
            "height": Value("int32"),
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


def save_pua_metadata(pua_metadata: PuaMetadata, output_dir: Path) -> Path | None:
    """PUAメタデータをJSONファイルに保存する."""
    if not pua_metadata:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "pua_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(pua_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved PUA metadata to {metadata_path}")
    return metadata_path


def format_yolo_label_line(bbox: list[float]) -> str:
    """YOLO bbox 1件分をラベル行に変換する."""
    x_center, y_center, width, height = bbox
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def export_roboflow_yolov8_dataset(
    annotations: list[ImageAnnotation],
    output_dir: Path,
    dataset_name_prefix: str,
) -> Path:
    """Roboflow 向け YOLOv8 検出データセットを書き出す."""
    dataset_dir = output_dir / f"{dataset_name_prefix}-roboflow-yolov8-columns"
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for ann in annotations:
        if not ann.columns:
            continue

        dst_image_path = images_dir / ann.image_path.name
        shutil.copy2(ann.image_path, dst_image_path)

        yolo_bboxes = [
            column.bbox if all(0.0 <= value <= 1.0 for value in column.bbox)
            else convert_bbox(
                int(column.bbox[0]),
                int(column.bbox[1]),
                int(column.bbox[2]),
                int(column.bbox[3]),
                ann.width,
                ann.height,
                "yolo",
            )
            for column in ann.columns
        ]
        label_lines = [format_yolo_label_line(bbox) for bbox in yolo_bboxes]
        label_path = labels_dir / f"{ann.image_path.stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        exported_count += 1

    data_yaml = "\n".join(
        [
            f"path: {dataset_dir}",
            "train: train/images",
            "val: ''",
            "test: ''",
            "nc: 1",
            "names:",
            "  - column",
            "",
        ]
    )
    (dataset_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Roboflow dataset exported: {dataset_dir}")
    print(f"Exported {exported_count} annotated images")
    return dataset_dir


def create_dataset_card(
    repo_id: str,
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
        "is_pua": List[bool],            # Whether category is a Private Use Area code
        "pua_code": List[str],           # PUA code strings if applicable
        "pua_reading": List[str],        # PUA readings from pua_metadata.json if available
        "pua_memo": List[str],           # PUA notes from pua_metadata.json if available
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

dataset = load_dataset("{repo_id}")

# Access first example
example = dataset["train"][0]
print(f"Image ID: {{example['image_id']}}")
print(f"Number of characters: {{len(example['objects']['bbox'])}}")

# Load label mappings
from huggingface_hub import hf_hub_download

label2id_path = hf_hub_download(
    repo_id="{repo_id}",
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
- `pua_metadata.json`: PUA code to reading / memo mapping when annotator metadata is available

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


def create_character_dataset_card(
    repo_id: str,
    num_characters: int,
    num_books: int,
    num_categories: int,
) -> DatasetCard:
    """文字単位データセットカードを作成する."""
    card_data = DatasetCardData(
        language=["ja"],
        license="cc-by-sa-4.0",
        task_categories=["image-classification"],
        tags=["kuzushiji", "japanese", "historical-documents", "ocr", "character-crops"],
        size_categories=["1K<n<10K"] if num_characters < 10000 else ["10K<n<100K"],
    )

    content = f"""---
{card_data.to_yaml()}
---

# Kuzushiji Character Dataset

This dataset contains character crops generated directly from page images using raw character annotations.

## Dataset Description

- **Number of character images**: {num_characters:,}
- **Number of books**: {num_books}
- **Number of character categories**: {num_categories:,}
- **Crop source**: raw page image + annotation CSV
- **Image size**: original cropped size (no resize)

## Dataset Structure

```python
{{
    "image": Image(),                # Cropped character image
    "source_image_id": str,          # Source page image ID
    "book_id": str,                  # Book ID
    "char_id": str,                  # Character annotation ID
    "block_id": str,                 # Block ID
    "category": str,                 # Unicode string (e.g., U+3042)
    "category_id": int,              # Category ID
    "is_pua": bool,                  # Whether category is a Private Use Area code
    "pua_code": str,                 # PUA code string if applicable
    "pua_reading": str,              # PUA reading from pua_metadata.json if available
    "pua_memo": str,                 # PUA note from pua_metadata.json if available
    "char": str,                     # Actual character
    "bbox": List[int],               # Original bbox on the source page [x, y, w, h]
    "crop_bbox": List[int],          # Clamped bbox used for cropping [x, y, w, h]
    "width": int,                    # Crop width in pixels
    "height": int,                   # Crop height in pixels
}}
```
"""
    return DatasetCard(content)


def should_generate_page_dataset(dataset_type: DatasetType) -> bool:
    """ページ単位 dataset を生成するか判定する."""
    return dataset_type in {"page", "both"}


def should_generate_character_dataset(dataset_type: DatasetType) -> bool:
    """文字単位 dataset を生成するか判定する."""
    return dataset_type in {"character", "both"}


def resolve_repo_id(
    dataset_name: str,
    hub_username: str | None,
    hub_token: str | None,
) -> str:
    """完全修飾の repo_id を返す."""
    if hub_username:
        return f"{hub_username}/{dataset_name}"

    api = HfApi(token=hub_token)
    user_info = api.whoami(token=hub_token)
    username = str(user_info["name"])
    return f"{username}/{dataset_name}"


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Convert raw data to Hugging Face Dataset format"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["hf", "roboflow"],
        default="hf",
        help="Export format (default: hf)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["page", "character", "both"],
        default="both",
        help="Dataset target for HF export (default: both)",
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
        "--pua-metadata-path",
        type=Path,
        default=None,
        help="Path to annotator pua_characters.json for PUA reading/memo metadata",
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
    export_format: ExportFormat = args.export_format
    dataset_type: DatasetType = args.dataset_type
    raw_dir: Path = args.raw_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    column_annotations_dir = args.column_annotations_dir.resolve()
    segment_annotations_dir = args.segment_annotations_dir.resolve()
    pua_metadata_paths = resolve_pua_metadata_paths(raw_dir, args.pua_metadata_path)
    pua_metadata = load_pua_metadata_files(pua_metadata_paths)

    if export_format == "roboflow" and args.push_to_hub:
        msg = "--push-to-hub は --export-format roboflow と同時に使用できません"
        raise SystemExit(msg)
    if export_format == "roboflow" and dataset_type == "character":
        msg = "--dataset-type character は --export-format roboflow と同時に使用できません"
        raise SystemExit(msg)

    if not raw_dir.exists():
        print(f"Error: Raw directory not found: {raw_dir}")
        return

    # トークン取得
    hub_token = args.hub_token or os.environ.get("HF_TOKEN") or get_token()

    print(f"Scanning raw directory: {raw_dir}")
    print(f"Export format: {export_format}")
    print(f"Dataset type: {dataset_type}")
    print(f"Bbox format: {bbox_format}")
    if pua_metadata_paths:
        print(f"PUA metadata: {len(pua_metadata)} entries from:")
        for metadata_path in pua_metadata_paths:
            print(f"  - {metadata_path}")
    else:
        print("PUA metadata: not found")

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
            pua_metadata,
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

    if export_format == "roboflow":
        export_roboflow_yolov8_dataset(
            annotations=all_annotations,
            output_dir=output_dir,
            dataset_name_prefix=args.dataset_name_prefix,
        )
        return

    # カテゴリマッピング構築
    print("Building category mapping...")
    label2id, id2label = build_category_mapping(all_annotations)
    print(f"Found {len(label2id)} unique characters")

    # ラベルマッピング保存
    save_label_mapping(label2id, id2label, output_dir)
    pua_metadata_output_path = save_pua_metadata(pua_metadata, output_dir)

    # データセット作成（ジェネレータを使用してメモリ効率化）
    dataset: Dataset | None = None
    character_dataset: Dataset | None = None

    if should_generate_page_dataset(dataset_type):
        print("Creating page dataset with generator...")
        features = create_dataset_features()

        def gen():
            yield from generate_dataset_records(all_annotations, label2id, bbox_format)

        dataset = Dataset.from_generator(gen, features=features)

        print(f"Page dataset created: {dataset}")
        print(f"Sample record keys: {list(dataset[0].keys())}")

    if should_generate_character_dataset(dataset_type):
        print("Creating character crop dataset with generator...")
        character_features = create_character_dataset_features()

        def character_gen():
            yield from generate_character_dataset_records(all_annotations, label2id)

        character_dataset = Dataset.from_generator(character_gen, features=character_features)

        print(f"Character dataset created: {character_dataset}")
        print(f"Character sample record keys: {list(character_dataset[0].keys())}")

    # Hubにプッシュ
    if args.push_to_hub:
        repo_id: str | None = None
        character_repo_id: str | None = None

        if dataset is not None:
            dataset_name = f"{args.dataset_name_prefix}-{bbox_format}"
            repo_id = resolve_repo_id(dataset_name, args.hub_username, hub_token)
            print(f"Pushing to Hub: {repo_id}")
            dataset.push_to_hub(
                repo_id,
                token=hub_token,
                max_shard_size=args.max_shard_size,
            )

        if character_dataset is not None:
            character_dataset_name = f"{args.dataset_name_prefix}-characters"
            character_repo_id = resolve_repo_id(
                character_dataset_name,
                args.hub_username,
                hub_token,
            )
            print(f"Pushing to Hub: {character_repo_id}")
            character_dataset.push_to_hub(
                character_repo_id,
                token=hub_token,
                max_shard_size=args.max_shard_size,
            )

        # label2id.json と id2label.json をHubにアップロード
        api = HfApi(token=hub_token)
        print("Uploading label mappings...")

        # label2id.json
        label2id_path = output_dir / "label2id.json"
        if repo_id is not None:
            api.upload_file(
                path_or_fileobj=str(label2id_path),
                path_in_repo="label2id.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hub_token,
            )
        if character_repo_id is not None:
            api.upload_file(
                path_or_fileobj=str(label2id_path),
                path_in_repo="label2id.json",
                repo_id=character_repo_id,
                repo_type="dataset",
                token=hub_token,
            )

        # id2label.json
        id2label_path = output_dir / "id2label.json"
        if repo_id is not None:
            api.upload_file(
                path_or_fileobj=str(id2label_path),
                path_in_repo="id2label.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hub_token,
            )
        if character_repo_id is not None:
            api.upload_file(
                path_or_fileobj=str(id2label_path),
                path_in_repo="id2label.json",
                repo_id=character_repo_id,
                repo_type="dataset",
                token=hub_token,
            )

        if pua_metadata_output_path is not None:
            if repo_id is not None:
                api.upload_file(
                    path_or_fileobj=str(pua_metadata_output_path),
                    path_in_repo="pua_metadata.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hub_token,
                )
            if character_repo_id is not None:
                api.upload_file(
                    path_or_fileobj=str(pua_metadata_output_path),
                    path_in_repo="pua_metadata.json",
                    repo_id=character_repo_id,
                    repo_type="dataset",
                    token=hub_token,
                )

        # データセットカードをアップロード
        print("Uploading dataset card...")
        if repo_id is not None:
            card = create_dataset_card(
                repo_id=repo_id,
                bbox_format=bbox_format,
                num_images=len(all_annotations),
                num_books=book_count,
                num_categories=len(label2id),
            )
            card.push_to_hub(repo_id, token=hub_token)
            print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
        if character_repo_id is not None:
            character_card = create_character_dataset_card(
                repo_id=character_repo_id,
                num_characters=sum(len(ann.characters) for ann in all_annotations),
                num_books=book_count,
                num_categories=len(label2id),
            )
            character_card.push_to_hub(character_repo_id, token=hub_token)
            print(f"Dataset pushed to: https://huggingface.co/datasets/{character_repo_id}")
    else:
        # ローカルに保存
        if dataset is not None:
            local_path = output_dir / f"{args.dataset_name_prefix}-{bbox_format}"
            print(f"Saving dataset locally: {local_path}")
            dataset.save_to_disk(str(local_path))
            print(f"Dataset saved to: {local_path}")
        if character_dataset is not None:
            character_local_path = output_dir / f"{args.dataset_name_prefix}-characters"
            print(f"Saving dataset locally: {character_local_path}")
            character_dataset.save_to_disk(str(character_local_path))
            print(f"Dataset saved to: {character_local_path}")


if __name__ == "__main__":
    main()
