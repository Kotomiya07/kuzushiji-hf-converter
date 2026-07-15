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


def get_size_category(num_examples: int) -> str:
    """Hugging Face の件数カテゴリを返す."""
    thresholds = (
        (1_000, "n<1K"),
        (10_000, "1K<n<10K"),
        (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"),
        (10_000_000, "1M<n<10M"),
        (100_000_000, "10M<n<100M"),
        (1_000_000_000, "100M<n<1B"),
    )
    for upper_bound, category in thresholds:
        if num_examples < upper_bound:
            return category
    return "n>1B"


PUA_DATASET_CARD_SECTION = """## PUA (Private Use Area) Characters

Some annotations use PUA (Private Use Area) code points for glyphs without a standard
Unicode representation. The original code point is retained in `category`; the derived
fields `is_pua`, `pua_code`, `pua_reading`, and `pua_memo` make these records explicit.
Readings and notes are empty when no PUA metadata was supplied to the converter.

A notable example is the four-way distinction around the kōto ligature:

| Character / code | Variant | `is_pua` | `pua_reading` |
|---|---|---:|---|
| ヿ (`U+30FF`) | Katakana ヿ (standard Unicode) | false | — |
| `U+E009` | Hiragana ヿ | true | こと (koto) |
| `U+E00A` | Hiragana ヿ with dakuten | true | ごと (goto) |
| `U+E00B` | Katakana ヿ with dakuten | true | ゴト (goto) |

Depending on the research question, users may merge these forms, preserve them as
separate classes, or exclude them. This choice is not made by the dataset converter.
"""


def create_dataset_card(
    repo_id: str,
    bbox_format: BboxFormat,
    num_images: int,
    num_books: int,
    num_categories: int,
) -> DatasetCard:
    """ページ単位データセットの詳細なカードを作成する."""
    bbox_descriptions = {
        "coco": "[x_min, y_min, width, height] in pixels",
        "yolo": "[x_center, y_center, width, height], normalized to 0–1",
    }
    pretty_format = bbox_format.upper()
    card_data = DatasetCardData(
        pretty_name=f"Kuzushiji Page Dataset ({pretty_format})",
        language=["ja"],
        license="cc-by-sa-4.0",
        task_categories=["object-detection"],
        tags=[
            "kuzushiji",
            "japanese",
            "historical-documents",
            "ocr",
            "document-layout-analysis",
            bbox_format,
        ],
        size_categories=[get_size_category(num_images)],
    )

    content = f"""---
{card_data.to_yaml()}
---

# Dataset Card for Kuzushiji Page Dataset

## Dataset Summary

The Kuzushiji Page Dataset packages page images from Japanese historical books with
character-level bounding boxes and labels. When supplied during conversion, it also
includes reading-column and segment annotations for document-layout analysis. This
repository contains the **{pretty_format}** variant; bounding boxes use
`{bbox_descriptions[bbox_format]}`.

This card describes the generated repository `{repo_id}`. Its statistics are calculated
at conversion time rather than copied from the upstream collection:

| Statistic | Value |
|---|---:|
| Page images | {num_images:,} |
| Books | {num_books:,} |
| Character categories | {num_categories:,} |
| Split | `train` only |

The source material is the **日本古典籍くずし字データセット (Japanese Historical
Character Dataset)**, owned by the National Institute of Japanese Literature (NIJL)
and other institutions and processed by the ROIS-DS Center for Open Data in the
Humanities (CODH).

## Supported Tasks and Leaderboards

- **Character detection / Kuzushiji recognition**: use `objects.bbox`, `category`, and
  `category_id` to locate and classify cursive Japanese characters.
- **Document layout analysis**: use the optional `columns` and `segments` fields to
  detect reading columns and larger text regions.
- **OCR preprocessing and evaluation**: use the page image, book identifier, Unicode
  labels, and layout hierarchy to construct recognition pipelines.

There is no official train/evaluation split, benchmark protocol, or leaderboard for this
converted dataset. Users must define evaluation splits appropriate to their task.

## Languages

The documents are in Japanese (`ja`), primarily historical written Japanese represented
in Kuzushiji. The metadata does not provide a verified language distribution by book,
period, script type, or genre.

## Dataset Structure

### Data Instances

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
example = dataset["train"][0]

print(example["image_id"], example["book_id"])
print(example["objects"]["bbox"][:3])
```

A record has the following shape (sequences may be empty):

```python
{{
    "image": Image(),
    "image_id": str,
    "book_id": str,
    "width": int,
    "height": int,
    "objects": {{
        "bbox": list[list[float]],
        "category": list[str],
        "category_id": list[int],
        "is_pua": list[bool],
        "pua_code": list[str],
        "pua_reading": list[str],
        "pua_memo": list[str],
        "char": list[str],
    }},
    "columns": {{
        "bbox": list[list[float]],
        "column_id": list[str],
        "char_ids": list[list[str]],
        "segment_id": list[str],
    }},
    "segments": {{
        "bbox": list[list[float]],
        "segment_id": list[str],
        "column_ids": list[list[str]],
    }},
}}
```

### Data Fields

| Field | Type | Description |
|---|---|---|
| `image` | `Image` | Full page image. |
| `image_id` | `string` | Source page identifier, without the image extension. |
| `book_id` | `string` | Identifier of the source book. |
| `width`, `height` | `int32` | Page dimensions in pixels. |
| `objects.bbox` | sequence of 4 floats | Character boxes in the repository's declared bbox format. |
| `objects.category` | sequence of strings | Unicode labels such as `U+3042`; PUA labels are preserved. |
| `objects.category_id` | sequence of `int32` | Integer IDs defined by `label2id.json`. |
| `objects.is_pua` | sequence of booleans | Whether each label is in a Unicode Private Use Area. |
| `objects.pua_code` | sequence of strings | PUA code, or an empty string for a standard Unicode label. |
| `objects.pua_reading` | sequence of strings | Optional reading imported from PUA metadata. |
| `objects.pua_memo` | sequence of strings | Optional note imported from PUA metadata. |
| `objects.char` | sequence of strings | Character obtained from the Unicode code point. |
| `columns.bbox` | sequence of 4 floats | Union box of the characters assigned to a reading column. |
| `columns.column_id` | sequence of strings | Column identifiers such as `COL0001`. |
| `columns.char_ids` | sequence of string sequences | Character IDs belonging to each column. |
| `columns.segment_id` | sequence of strings | Parent segment ID, or an empty string when unavailable. |
| `segments.bbox` | sequence of 4 floats | Union box of columns/characters assigned to a segment. |
| `segments.segment_id` | sequence of strings | Segment identifiers such as `SEG0001`. |
| `segments.column_ids` | sequence of string sequences | Column IDs belonging to each segment. |

The parallel sequences within `objects`, `columns`, and `segments` are positionally
aligned. `columns` and `segments` can be empty when the corresponding optional annotation
files were not provided or a page was not annotated.

### Data Splits

| Split | Number of rows |
|---|---:|
| train | {num_images:,} |

No validation or test split is generated. To reduce leakage between pages from the same
work, create downstream splits by `book_id`, not by randomly splitting individual rows.

### Bounding Box Formats

| Variant | Coordinates | Normalized |
|---|---|---:|
| COCO | `[x_min, y_min, width, height]` | No; pixel units |
| YOLO | `[x_center, y_center, width, height]` | Yes; values in 0–1 |

All character, column, and segment boxes in this repository use the **{pretty_format}**
variant. Do not infer the format from the values alone.

### Label Mapping Files

- `label2id.json`: Unicode label to category ID.
- `id2label.json`: category ID to Unicode label.
- `pua_metadata.json`: PUA code to reading and memo, only when metadata was available.

{PUA_DATASET_CARD_SECTION}

## Dataset Creation

### Curation Rationale

This conversion makes the upstream character coordinates directly usable with Hugging
Face Datasets and preserves page/book provenance. Optional column and segment structures
support layout-aware OCR and reading-order research without replacing the original
character annotations.

### Source Data

#### Initial Data Collection and Normalization

The converter reads each source page image and its coordinate CSV. It validates image
availability, converts Unicode code points to display characters, assigns deterministic
category IDs, and converts boxes to the selected COCO or YOLO representation. Column and
segment boxes are derived as unions of their member character boxes when matching local
annotation CSV files are supplied. The images themselves are not resized by this step.

Upstream source:

- **Dataset**: 日本古典籍くずし字データセット
- **Owners**: National Institute of Japanese Literature and other institutions
- **Processing**: ROIS-DS Center for Open Data in the Humanities (CODH)
- **DOI**: [10.20676/00000340](https://doi.org/10.20676/00000340)
- **Website**: [codh.rois.ac.jp/char-shape](https://codh.rois.ac.jp/char-shape/)

#### Who Are the Source Language Producers?

The text was produced by historical Japanese authors, scribes, printers, and publishers.
Their identities and demographic attributes are not encoded in this converted dataset.
The holding institutions and CODH provide and process the digitized source material.

### Annotations

#### Annotation Process

Character labels and coordinates originate from the upstream dataset. The converter does
not re-transcribe or independently verify them. Column annotations are optional local
assignments of characters to reading columns. Segment annotations are optional derived or
human-corrected groupings of columns. Their availability can therefore differ by book and
page; empty sequences do not mean that the page contains no text.

#### Who Are the Annotators?

See the upstream dataset documentation for the provenance of character annotations. The
converter does not store annotator identities for optional column/segment annotations, so
their annotator composition and inter-annotator agreement cannot be determined from this
repository alone.

### Personal and Sensitive Information

Historical pages can contain personal names, addresses, ownership marks, or other
information about historical individuals. No dedicated personal-information or sensitive-
content audit is performed during conversion. Users should inspect the source material for
their intended publication context and follow the policies of the holding institutions.

## Considerations for Using the Data

### Social Impact of the Dataset

The dataset can support preservation, search, transcription, and accessibility of
historical Japanese materials. Automated recognition may also produce plausible but
incorrect readings; outputs should not be treated as authoritative transcriptions without
review, especially in historical, genealogical, or identity-related research.

### Discussion of Biases

The collection reflects the books selected, preserved, digitized, and annotated by the
source institutions rather than the full distribution of historical Japanese writing.
Character frequencies, genres, periods, hands, print styles, page conditions, and
institutions may be uneven. Rare characters and PUA labels are likely to be especially
sparse. No demographic, geographic, genre, or performance fairness audit is included.

### Other Known Limitations

- Only a `train` split is provided; reported row counts describe this generated revision.
- Annotation completeness and accuracy are inherited from upstream data and optional local
  column/segment files; they are not independently audited by the converter.
- Categories can be highly imbalanced, and PUA semantics depend on optional metadata.
- Boxes are axis-aligned and cannot fully describe rotated, touching, damaged, or highly
  irregular glyphs and regions.
- Column and segment annotations may be absent or partially covered across books/pages.
- `category_id` values are repository-specific; use the included mapping files rather than
  assuming IDs are stable across independently generated versions.
- A random page-level split can leak book-specific visual characteristics; split by book
  for a stronger estimate of generalization.

## Additional Information

### Dataset Curators

The source collection is curated and processed by NIJL, other holding institutions, and
CODH. This Hugging Face packaging is generated by the `kuzushiji-hf-converter` project;
consult the repository history for the maintainers of a particular published revision.

### Licensing Information

This generated dataset is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Users remain responsible
for checking the upstream dataset terms, providing attribution, indicating modifications,
and applying ShareAlike requirements to adaptations. This card is descriptive and is not
legal advice.

### Citation Information

Please cite the source dataset:

```text
『日本古典籍くずし字データセット』（国文研ほか所蔵／CODH加工）
doi:10.20676/00000340
```

```text
"Japanese Historical Character Dataset"
(Owned by NIJL and others, Processed by CODH)
doi:10.20676/00000340
```

When publishing a derived dataset or model, also cite the exact Hugging Face repository
revision used (`{repo_id}`) so that the generated schema and statistics are reproducible.

### Contributions and Acknowledgments

Data is provided by the ROIS-DS Center for Open Data in the Humanities, the National
Institute of Japanese Literature, and the other holding institutions credited by the
upstream dataset.
"""
    return DatasetCard(content)


def create_character_dataset_card(
    repo_id: str,
    num_characters: int,
    num_books: int,
    num_categories: int,
) -> DatasetCard:
    """文字単位データセットの詳細なカードを作成する."""
    card_data = DatasetCardData(
        pretty_name="Kuzushiji Character Dataset",
        language=["ja"],
        license="cc-by-sa-4.0",
        task_categories=["image-classification"],
        tags=[
            "kuzushiji",
            "japanese",
            "historical-documents",
            "ocr",
            "character-crops",
        ],
        size_categories=[get_size_category(num_characters)],
    )

    content = f"""---
{card_data.to_yaml()}
---

# Dataset Card for Kuzushiji Character Dataset

## Dataset Summary

The Kuzushiji Character Dataset contains individual character crops generated from full
page images and character-coordinate annotations. Crops retain their original pixel size;
they are not resized or padded. Each record keeps the source book, page, character ID,
Unicode label, original page box, and the clamped box actually used for cropping.

This card describes the generated repository `{repo_id}`. Statistics are calculated at
conversion time:

| Statistic | Value |
|---|---:|
| Character images | {num_characters:,} |
| Books | {num_books:,} |
| Character categories | {num_categories:,} |
| Split | `train` only |

The source material is the **日本古典籍くずし字データセット (Japanese Historical
Character Dataset)**, owned by NIJL and other institutions and processed by CODH.

## Supported Tasks and Leaderboards

- **Image classification / Kuzushiji recognition**: predict `category` or `category_id`
  from a cropped character image.
- **Representation learning and retrieval**: learn glyph embeddings while retaining
  `book_id` and `source_image_id` for provenance-aware evaluation.
- **OCR component evaluation**: evaluate isolated-character recognizers before integrating
  them into page-level detection and transcription systems.

There is no official validation/test split, benchmark protocol, or leaderboard for this
converted dataset.

## Languages

The labels represent characters used in historical Japanese (`ja`). A label is a Unicode
code-point string, not a modern-Japanese reading or complete transcription. Language,
period, genre, and script distributions are not provided at record level.

## Dataset Structure

### Data Instances

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
example = dataset["train"][0]

print(example["category"], example["char"])
print(example["source_image_id"], example["crop_bbox"])
```

```python
{{
    "image": Image(),
    "source_image_id": str,
    "book_id": str,
    "char_id": str,
    "block_id": str,
    "category": str,
    "category_id": int,
    "is_pua": bool,
    "pua_code": str,
    "pua_reading": str,
    "pua_memo": str,
    "char": str,
    "bbox": list[int],
    "crop_bbox": list[int],
    "width": int,
    "height": int,
}}
```

### Data Fields

| Field | Type | Description |
|---|---|---|
| `image` | `Image` | Character crop encoded from the source page; no resize or padding. |
| `source_image_id` | `string` | Identifier of the page from which the crop was extracted. |
| `book_id` | `string` | Identifier of the source book. |
| `char_id` | `string` | Character annotation identifier within the source data. |
| `block_id` | `string` | Optional source block identifier; may be empty. |
| `category` | `string` | Unicode label such as `U+3042`; PUA labels are preserved. |
| `category_id` | `int32` | Integer class ID defined by `label2id.json`. |
| `is_pua` | `bool` | Whether `category` is in a Unicode Private Use Area. |
| `pua_code` | `string` | PUA code, or an empty string for a standard Unicode label. |
| `pua_reading` | `string` | Optional reading imported from PUA metadata. |
| `pua_memo` | `string` | Optional note imported from PUA metadata. |
| `char` | `string` | Character obtained from the Unicode code point. |
| `bbox` | 4 `int32` values | Original page coordinates `[x, y, width, height]`. |
| `crop_bbox` | 4 `int32` values | Boundary-clamped page coordinates actually used for cropping. |
| `width`, `height` | `int32` | Resulting crop dimensions in pixels. |

### Data Splits

| Split | Number of rows |
|---|---:|
| train | {num_characters:,} |

No validation or test split is generated. Multiple characters from the same page and book
share visual and material characteristics. Downstream evaluation should therefore split by
`book_id` (or at least `source_image_id`) before training rather than randomly splitting
individual character rows.

### Label Mapping Files

- `label2id.json`: Unicode label to category ID.
- `id2label.json`: category ID to Unicode label.
- `pua_metadata.json`: PUA code to reading and memo, only when metadata was available.

{PUA_DATASET_CARD_SECTION}

## Dataset Creation

### Curation Rationale

This derived view supports isolated-character classification and retrieval without
requiring users to reproduce page cropping. It preserves links to the source page and book
so users can build leakage-resistant splits or return to the full context.

### Source Data

#### Initial Data Collection and Normalization

For each character annotation, the converter reads `[x, y, width, height]` from the source
CSV, intersects that rectangle with the source image bounds, and encodes the resulting
crop. `bbox` records the original annotation; `crop_bbox` records the rectangle actually
used. Empty intersections are skipped. Crops are not resized, padded, deskewed, denoised,
or contrast-normalized.

Upstream source:

- **Dataset**: 日本古典籍くずし字データセット
- **Owners**: National Institute of Japanese Literature and other institutions
- **Processing**: ROIS-DS Center for Open Data in the Humanities (CODH)
- **DOI**: [10.20676/00000340](https://doi.org/10.20676/00000340)
- **Website**: [codh.rois.ac.jp/char-shape](https://codh.rois.ac.jp/char-shape/)

#### Who Are the Source Language Producers?

The source text was produced by historical Japanese authors, scribes, printers, and
publishers. Their identities and demographic attributes are not represented as structured
fields in this derived dataset.

### Annotations

#### Annotation Process

Character labels and page coordinates come from the upstream dataset. The converter
performs deterministic cropping and derives PUA helper fields, but it does not re-label,
transcribe, or independently validate each glyph. Category IDs are created from the labels
present during conversion and can differ between generated repositories.

#### Who Are the Annotators?

See the upstream dataset documentation for annotation provenance. Annotator identities,
agreement scores, and per-record confidence values are not included in this converted
view.

### Personal and Sensitive Information

Although each image is a small glyph crop, labels and source identifiers link it back to a
historical page that may contain personal names or other information about historical
individuals. The converter performs no personal-information or sensitive-content audit.

## Considerations for Using the Data

### Social Impact of the Dataset

Character-level recognition can improve transcription and access to Japanese historical
collections. Predictions remain uncertain for rare, damaged, or context-dependent forms;
using isolated predictions as authoritative readings can introduce errors into historical
records. Human review and page context are important for high-stakes interpretation.

### Discussion of Biases

The class distribution follows the selected and annotated source books and is expected to
be long-tailed. Preserved works, institutions, genres, periods, hands, print styles, and
page conditions may be unevenly represented. PUA classes and rare variants can have very
few samples. No demographic, geographic, genre, or class-wise performance audit is
included.

### Other Known Limitations

- Only a `train` split is provided; counts describe the current generated revision.
- Crops inherit annotation errors and may include neighboring marks, partial glyphs,
  degradation, or background artifacts.
- Crop dimensions vary, so models generally need an explicit resize/pad policy.
- Isolated crops omit page and linguistic context needed to disambiguate many Kuzushiji
  forms.
- Random row-level splitting causes leakage because crops from the same page/book are
  visually related; group splits by `book_id` or `source_image_id`.
- Categories are imbalanced; accuracy alone can conceal poor rare-character performance.
- PUA readings/notes are optional, and `category_id` is not guaranteed stable across
  independently generated versions.

## Additional Information

### Dataset Curators

The source collection is curated and processed by NIJL, other holding institutions, and
CODH. This character-crop view is generated by the `kuzushiji-hf-converter` project;
consult the repository history for maintainers of a particular published revision.

### Licensing Information

This generated dataset is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Users remain responsible
for checking the upstream terms, providing attribution, indicating modifications, and
applying ShareAlike requirements to adaptations. This card is not legal advice.

### Citation Information

Please cite the source dataset:

```text
『日本古典籍くずし字データセット』（国文研ほか所蔵／CODH加工）
doi:10.20676/00000340
```

```text
"Japanese Historical Character Dataset"
(Owned by NIJL and others, Processed by CODH)
doi:10.20676/00000340
```

When publishing a derived dataset or model, also cite the exact Hugging Face repository
revision used (`{repo_id}`) so the generated schema and statistics can be reproduced.

### Contributions and Acknowledgments

Data is provided by the ROIS-DS Center for Open Data in the Humanities, the National
Institute of Japanese Literature, and the other holding institutions credited by the
upstream dataset.
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
