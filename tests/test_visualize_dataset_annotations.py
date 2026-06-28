from pathlib import Path
import sys

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parents[1]))

import visualize_dataset_annotations as visualizer


def build_features() -> Features:
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


def test_bbox_to_xyxy_supports_coco_and_yolo() -> None:
    assert visualizer.bbox_to_xyxy([10.0, 20.0, 30.0, 40.0], "coco", 200, 100) == (10, 20, 40, 60)
    assert visualizer.bbox_to_xyxy([0.25, 0.4, 0.1, 0.2], "yolo", 200, 100) == (40, 30, 60, 50)


def test_polygon_to_xy_supports_coco_and_yolo() -> None:
    assert visualizer.polygon_to_xy([10.0, 20.0, 40.0, 20.0, 25.0, 60.0], "coco", 200, 100) == [
        (10, 20),
        (40, 20),
        (25, 60),
    ]
    assert visualizer.polygon_to_xy([0.1, 0.2, 0.2, 0.2, 0.15, 0.6], "yolo", 200, 100) == [
        (20, 20),
        (40, 20),
        (30, 60),
    ]


def test_detect_bbox_format_prefers_yolo_for_normalized_boxes() -> None:
    record = {
        "objects": {"bbox": [[0.25, 0.4, 0.1, 0.2]]},
        "columns": {"bbox": []},
        "segments": {"bbox": []},
    }

    assert visualizer.detect_bbox_format(record) == "yolo"


def test_visualize_dataset_split_saves_overlay_image(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    PILImage.new("RGB", (120, 80), color="white").save(image_path)

    dataset = Dataset.from_dict(
        {
            "image": [str(image_path)],
            "image_id": ["page_001"],
            "book_id": ["book1"],
            "width": [120],
            "height": [80],
            "objects": [
                {
                    "bbox": [[10.0, 10.0, 20.0, 20.0]],
                    "category": ["U+4E00"],
                    "category_id": [0],
                    "char": ["一"],
                }
            ],
            "columns": [
                {
                    "bbox": [[8.0, 8.0, 24.0, 30.0]],
                    "column_id": ["COL0001"],
                    "char_ids": [["C0001"]],
                    "segment_id": ["SEG0001"],
                }
            ],
            "segments": [
                {
                    "bbox": [[5.0, 5.0, 30.0, 40.0]],
                    "segment_id": ["SEG0001"],
                    "column_ids": [["COL0001"]],
                }
            ],
        },
        features=build_features(),
    )
    dataset_dir = tmp_path / "dataset"
    dataset.save_to_disk(str(dataset_dir))

    output_dir = tmp_path / "visualizations"
    saved_paths = visualizer.visualize_dataset_split(
        dataset_source=dataset_dir,
        output_dir=output_dir,
        split="train",
        bbox_format="auto",
        start_index=0,
        max_samples=1,
    )

    assert len(saved_paths) == 1
    assert saved_paths[0].exists()

    rendered = PILImage.open(saved_paths[0])
    assert rendered.size == (120, 80)
    assert rendered.getbbox() is not None


def test_visualize_record_fills_segment_polygon() -> None:
    image = PILImage.new("RGB", (120, 80), color="white")
    record = {
        "image": image,
        "objects": {"bbox": [], "char": []},
        "columns": {"bbox": [], "column_id": []},
        "segments": {
            "bbox": [[10.0, 10.0, 30.0, 30.0]],
            "segment_id": ["SEG0001"],
            "polygon": [[10.0, 10.0, 40.0, 10.0, 25.0, 40.0]],
        },
    }

    rendered = visualizer.visualize_record(record, bbox_format="coco")

    assert rendered.getpixel((25, 20)) != (255, 255, 255)
