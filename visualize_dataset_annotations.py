"""変換済みデータセットの bbox 可視化スクリプト."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image as PILImage
from PIL import ImageColor, ImageDraw


BboxFormat = Literal["coco", "yolo"]
InputBboxFormat = Literal["auto", "coco", "yolo"]
Record = dict[str, object]

CHAR_COLOR = ImageColor.getrgb("#2563eb")
COLUMN_COLOR = ImageColor.getrgb("#16a34a")
SEGMENT_COLOR = ImageColor.getrgb("#dc2626")
SEGMENT_FILL = (220, 38, 38, 56)
LABEL_BG = ImageColor.getrgb("#111827")
LABEL_FG = ImageColor.getrgb("#f9fafb")


def bbox_to_xyxy(
    bbox: Sequence[float],
    bbox_format: BboxFormat,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """bbox を描画用のピクセル xyxy に変換する。"""
    if len(bbox) != 4:
        msg = f"bbox length must be 4: {bbox}"
        raise ValueError(msg)

    if bbox_format == "coco":
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
    else:
        x_center, y_center, width, height = bbox
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
        x_min = x_center_px - (width_px / 2)
        y_min = y_center_px - (height_px / 2)
        x_max = x_center_px + (width_px / 2)
        y_max = y_center_px + (height_px / 2)

    return (
        max(0, round(x_min)),
        max(0, round(y_min)),
        min(image_width, round(x_max)),
        min(image_height, round(y_max)),
    )


def detect_bbox_format(record: Mapping[str, object]) -> BboxFormat:
    """レコード内容から bbox 形式を推定する。"""
    for key in ("objects", "columns", "segments"):
        group = record.get(key)
        if not isinstance(group, Mapping):
            continue
        bbox_values = group.get("bbox")
        if not isinstance(bbox_values, Sequence) or len(bbox_values) == 0:
            continue

        first_bbox = bbox_values[0]
        if not isinstance(first_bbox, Sequence) or len(first_bbox) != 4:
            continue

        numeric_bbox = [float(value) for value in first_bbox]
        if all(0.0 <= value <= 1.0 for value in numeric_bbox):
            return "yolo"
        return "coco"

    return "coco"


def point_to_xy(
    x: float,
    y: float,
    bbox_format: BboxFormat,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """点座標を描画用ピクセル xy に変換する。"""
    if bbox_format == "yolo":
        x *= image_width
        y *= image_height

    return (
        min(image_width, max(0, round(x))),
        min(image_height, max(0, round(y))),
    )


def polygon_to_xy(
    polygon: Sequence[float],
    bbox_format: BboxFormat,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int]]:
    """平坦化された polygon 座標列を描画用ピクセル座標へ変換する。"""
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        msg = f"polygon must contain at least 3 xy pairs: {polygon}"
        raise ValueError(msg)

    points: list[tuple[int, int]] = []
    for index in range(0, len(polygon), 2):
        points.append(
            point_to_xy(
                float(polygon[index]),
                float(polygon[index + 1]),
                bbox_format,
                image_width,
                image_height,
            )
        )
    return points


def load_split(dataset_source: Path | str, split: str) -> Dataset:
    """ローカル保存済み Dataset または Hub 上の dataset split を読み込む。"""
    source_path = Path(dataset_source)
    if source_path.exists():
        loaded = load_from_disk(str(source_path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded

    return cast(Dataset, load_dataset(str(dataset_source), split=split))


def load_image(image_value: object) -> PILImage.Image:
    """Dataset の image カラムを PIL.Image に正規化する。"""
    if isinstance(image_value, PILImage.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, Mapping):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")
        if isinstance(image_bytes, bytes):
            return PILImage.open(BytesIO(image_bytes)).convert("RGB")
        if isinstance(image_path, str):
            return PILImage.open(image_path).convert("RGB")

    if isinstance(image_value, str):
        return PILImage.open(image_value).convert("RGB")

    msg = f"Unsupported image payload: {type(image_value)!r}"
    raise TypeError(msg)


def draw_box(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    line_width: int,
) -> None:
    """単一 bbox とラベルを描画する。"""
    draw.rectangle(xyxy, outline=color, width=line_width)

    if not label:
        return

    label_pos = (xyxy[0] + 2, max(0, xyxy[1] - 14))
    text_bbox = draw.textbbox(label_pos, label)
    draw.rectangle(text_bbox, fill=LABEL_BG)
    draw.text(label_pos, label, fill=LABEL_FG)


def draw_annotation_group(
    image: PILImage.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    bbox_format: BboxFormat,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    """同種の bbox 群を画像へ描画する。"""
    draw = ImageDraw.Draw(image)
    for index, bbox in enumerate(boxes):
        xyxy = bbox_to_xyxy(bbox, bbox_format, image.width, image.height)
        label = labels[index] if index < len(labels) else ""
        draw_box(draw, xyxy, color, label, line_width)


def resolve_segment_polygons(segments: Mapping[str, object]) -> Sequence[Sequence[float]]:
    """セグメント polygon 候補フィールドを取り出す。"""
    for key in ("polygon", "polygons", "segmentation"):
        value = segments.get(key)
        if isinstance(value, Sequence):
            return cast(Sequence[Sequence[float]], value)
    return []


def draw_segment_overlay(
    image: PILImage.Image,
    segments: Mapping[str, object],
    bbox_format: BboxFormat,
    line_width: int,
) -> None:
    """セグメント領域を半透明塗りで描画する。"""
    boxes = cast(Sequence[Sequence[float]], segments.get("bbox", []))
    labels = cast(Sequence[str], segments.get("segment_id", []))
    polygons = resolve_segment_polygons(segments)

    overlay = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for index, polygon in enumerate(polygons):
        if index >= len(boxes):
            break
        if len(polygon) < 6:
            continue
        overlay_draw.polygon(
            polygon_to_xy(polygon, bbox_format, image.width, image.height),
            fill=SEGMENT_FILL,
            outline=SEGMENT_COLOR,
            width=line_width,
        )

    for index in range(len(polygons), len(boxes)):
        overlay_draw.rectangle(
            bbox_to_xyxy(boxes[index], bbox_format, image.width, image.height),
            fill=SEGMENT_FILL,
            outline=None,
        )

    composited = PILImage.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.paste(composited)
    draw_annotation_group(
        image,
        boxes,
        labels,
        bbox_format,
        SEGMENT_COLOR,
        line_width,
    )


def visualize_record(
    record: Mapping[str, object],
    bbox_format: InputBboxFormat,
    line_width: int = 3,
) -> PILImage.Image:
    """1 レコード分の画像に bbox を重ね描きする。"""
    image = load_image(record["image"]).copy()
    resolved_bbox_format: BboxFormat = (
        detect_bbox_format(record) if bbox_format == "auto" else bbox_format
    )

    objects = cast(Mapping[str, object], record.get("objects", {}))
    columns = cast(Mapping[str, object], record.get("columns", {}))
    segments = cast(Mapping[str, object], record.get("segments", {}))

    draw_annotation_group(
        image,
        cast(Sequence[Sequence[float]], objects.get("bbox", [])),
        cast(Sequence[str], objects.get("char", [])),
        resolved_bbox_format,
        CHAR_COLOR,
        line_width,
    )
    draw_annotation_group(
        image,
        cast(Sequence[Sequence[float]], columns.get("bbox", [])),
        cast(Sequence[str], columns.get("column_id", [])),
        resolved_bbox_format,
        COLUMN_COLOR,
        line_width,
    )
    draw_segment_overlay(image, segments, resolved_bbox_format, line_width)

    return image


def visualize_dataset_split(
    dataset_source: Path | str,
    output_dir: Path,
    split: str,
    bbox_format: InputBboxFormat,
    start_index: int,
    max_samples: int | None,
    line_width: int = 3,
) -> list[Path]:
    """指定 split のレコードを可視化して保存する。"""
    dataset = load_split(dataset_source, split)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    end_index = total if max_samples is None else min(total, start_index + max_samples)
    saved_paths: list[Path] = []

    for index in range(start_index, end_index):
        record = cast(Record, dataset[index])
        rendered = visualize_record(record, bbox_format=bbox_format, line_width=line_width)
        image_id = str(record.get("image_id", f"index_{index:05d}"))
        output_path = output_dir / f"{index:05d}_{image_id}.png"
        rendered.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="変換済みデータセットの bbox を描画する")
    parser.add_argument("dataset_source", help="ローカル Dataset パスまたは Hugging Face repo id")
    parser.add_argument("--split", default="train", help="対象 split 名")
    parser.add_argument(
        "--bbox-format",
        choices=["auto", "coco", "yolo"],
        default="auto",
        help="bbox 形式。auto の場合はレコードから推定",
    )
    parser.add_argument("--start-index", type=int, default=0, help="開始インデックス")
    parser.add_argument("--max-samples", type=int, default=10, help="描画する件数")
    parser.add_argument("--line-width", type=int, default=3, help="矩形線の太さ")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "visualizations",
        help="描画結果の保存先",
    )
    return parser.parse_args()


def main() -> None:
    """CLI エントリーポイント。"""
    args = parse_args()
    saved_paths = visualize_dataset_split(
        dataset_source=args.dataset_source,
        output_dir=args.output_dir,
        split=args.split,
        bbox_format=cast(InputBboxFormat, args.bbox_format),
        start_index=args.start_index,
        max_samples=args.max_samples,
        line_width=args.line_width,
    )
    print(f"saved {len(saved_paths)} visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
