from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parents[1]))

import convert_dataset


def test_load_column_segment_annotations_builds_page_groups(tmp_path: Path) -> None:
    book_id = "book1"
    column_dir = tmp_path / "output" / book_id
    segment_dir = tmp_path / "output_seg" / book_id
    column_dir.mkdir(parents=True)
    segment_dir.mkdir(parents=True)

    rows = pd.DataFrame(
        [
            {
                "Image": "page_001",
                "Unicode": "U+4E00",
                "X": 100,
                "Y": 20,
                "Width": 20,
                "Height": 40,
                "Char ID": "C0001",
                "Block ID": "",
                "Column ID": "COL0001",
                "Segment ID": "SEG0001",
            },
            {
                "Image": "page_001",
                "Unicode": "U+4E01",
                "X": 100,
                "Y": 80,
                "Width": 20,
                "Height": 40,
                "Char ID": "C0002",
                "Block ID": "",
                "Column ID": "COL0001",
                "Segment ID": "SEG0001",
            },
            {
                "Image": "page_001",
                "Unicode": "U+4E02",
                "X": 60,
                "Y": 25,
                "Width": 20,
                "Height": 40,
                "Char ID": "C0003",
                "Block ID": "",
                "Column ID": "COL0002",
                "Segment ID": "SEG0002",
            },
        ]
    )
    rows.drop(columns=["Segment ID"]).to_csv(column_dir / "column_annotation.csv", index=False)
    rows.to_csv(segment_dir / "column_annotation.csv", index=False)

    page_map = convert_dataset.load_column_segment_annotations(
        book_id,
        tmp_path / "output",
        tmp_path / "output_seg",
        "coco",
    )

    columns, segments = page_map["page_001"]
    assert [column.column_id for column in columns] == ["COL0001", "COL0002"]
    assert columns[0].char_ids == ["C0001", "C0002"]
    assert [segment.segment_id for segment in segments] == ["SEG0001", "SEG0002"]
    assert segments[0].column_ids == ["COL0001"]


def test_load_column_segment_annotations_uses_actual_image_size_for_yolo(tmp_path: Path) -> None:
    book_id = "book1"
    segment_dir = tmp_path / "output_seg" / book_id
    segment_dir.mkdir(parents=True)

    rows = pd.DataFrame(
        [
            {
                "Image": "page_001",
                "Unicode": "U+4E00",
                "X": 100,
                "Y": 20,
                "Width": 20,
                "Height": 40,
                "Char ID": "C0001",
                "Block ID": "",
                "Column ID": "COL0001",
                "Segment ID": "SEG0001",
            }
        ]
    )
    rows.to_csv(segment_dir / "column_annotation.csv", index=False)

    page_map = convert_dataset.load_column_segment_annotations(
        book_id,
        None,
        tmp_path / "output_seg",
        "yolo",
        image_sizes={"page_001": (200, 100)},
    )

    columns, segments = page_map["page_001"]
    assert columns[0].bbox == [0.55, 0.4, 0.1, 0.4]
    assert segments[0].bbox == [0.55, 0.4, 0.1, 0.4]


def test_generate_dataset_records_includes_columns_and_segments(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake-image")

    annotation = convert_dataset.ImageAnnotation(
        image_id="page_001",
        book_id="book1",
        image_path=image_path,
        width=200,
        height=300,
        characters=[
            convert_dataset.CharAnnotation(
                unicode="U+4E00",
                x=100,
                y=20,
                width=20,
                height=40,
                block_id="",
                char_id="C0001",
            )
        ],
        columns=[
            convert_dataset.ColumnAnnotation(
                column_id="COL0001",
                bbox=[100.0, 20.0, 20.0, 40.0],
                char_ids=["C0001"],
                segment_id="SEG0001",
            )
        ],
        segments=[
            convert_dataset.SegmentAnnotation(
                segment_id="SEG0001",
                bbox=[100.0, 20.0, 20.0, 40.0],
                column_ids=["COL0001"],
            )
        ],
    )

    records = list(
        convert_dataset.generate_dataset_records(
            [annotation],
            {"U+4E00": 0},
            "coco",
        )
    )

    assert records[0]["columns"]["column_id"] == ["COL0001"]
    assert records[0]["segments"]["segment_id"] == ["SEG0001"]


def test_load_annotations_derives_pua_info_from_unicode_column(tmp_path: Path) -> None:
    book_id = "book1"
    images_dir = tmp_path / "raw" / book_id / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "page_001.jpg"
    PILImage.new("RGB", (20, 20), color="white").save(image_path)

    csv_path = tmp_path / "raw" / book_id / f"{book_id}_coordinate.csv"
    pd.DataFrame(
        [
            {
                "Image": "page_001",
                "Unicode": "U+E000",
                "X": 2,
                "Y": 3,
                "Width": 4,
                "Height": 5,
                "Char ID": "C0001",
                "Block ID": "",
            }
        ]
    ).to_csv(csv_path, index=False)
    pua_metadata = {
        "U+E000": {
            "reading": "き",
            "memo": "竹かんむりに車へん",
        }
    }

    annotations = convert_dataset.load_annotations(
        csv_path,
        images_dir,
        book_id,
        "coco",
        None,
        None,
        pua_metadata,
    )

    char = annotations[0].characters[0]
    assert char.is_pua is True
    assert char.pua_code == "U+E000"
    assert char.pua_reading == "き"
    assert char.pua_memo == "竹かんむりに車へん"


def test_load_pua_metadata_reads_annotator_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "pua_characters.json"
    metadata_path.write_text(
        """
{
  "pua_characters": {
    "U+E000": {
      "reading": "き",
      "memo": "竹かんむりに車へん",
      "usage_count": 1
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    metadata = convert_dataset.load_pua_metadata(metadata_path)

    assert metadata == {
        "U+E000": {
            "reading": "き",
            "memo": "竹かんむりに車へん",
        }
    }


def test_generate_dataset_records_includes_pua_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake-image")

    annotation = convert_dataset.ImageAnnotation(
        image_id="page_001",
        book_id="book1",
        image_path=image_path,
        width=200,
        height=300,
        characters=[
            convert_dataset.CharAnnotation(
                unicode="U+E000",
                x=100,
                y=20,
                width=20,
                height=40,
                block_id="",
                char_id="C0001",
                is_pua=True,
                pua_code="U+E000",
                pua_reading="き",
                pua_memo="竹かんむりに車へん",
            )
        ],
        columns=[],
        segments=[],
    )

    records = list(
        convert_dataset.generate_dataset_records(
            [annotation],
            {"U+E000": 0},
            "coco",
        )
    )

    assert records[0]["objects"]["is_pua"] == [True]
    assert records[0]["objects"]["pua_code"] == ["U+E000"]
    assert records[0]["objects"]["pua_reading"] == ["き"]
    assert records[0]["objects"]["pua_memo"] == ["竹かんむりに車へん"]


def test_generate_character_dataset_records_crops_from_page_image(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    page = PILImage.new("RGB", (10, 10), color="white")
    for x in range(2, 6):
        for y in range(3, 8):
            page.putpixel((x, y), (255, 0, 0))
    page.save(image_path)

    annotation = convert_dataset.ImageAnnotation(
        image_id="page_001",
        book_id="book1",
        image_path=image_path,
        width=10,
        height=10,
        characters=[
            convert_dataset.CharAnnotation(
                unicode="U+E000",
                x=2,
                y=3,
                width=4,
                height=5,
                block_id="B0001",
                char_id="C0001",
                is_pua=True,
                pua_code="U+E000",
                pua_reading="き",
                pua_memo="竹かんむりに車へん",
            )
        ],
        columns=[],
        segments=[],
    )

    records = list(
        convert_dataset.generate_character_dataset_records(
            [annotation],
            {"U+E000": 0},
        )
    )

    assert len(records) == 1
    assert records[0]["source_image_id"] == "page_001"
    assert records[0]["char_id"] == "C0001"
    assert records[0]["category"] == "U+E000"
    assert records[0]["is_pua"] is True
    assert records[0]["pua_code"] == "U+E000"
    assert records[0]["pua_reading"] == "き"
    assert records[0]["pua_memo"] == "竹かんむりに車へん"
    assert records[0]["bbox"] == [2, 3, 4, 5]
    assert records[0]["crop_bbox"] == [2, 3, 4, 5]
    assert records[0]["width"] == 4
    assert records[0]["height"] == 5

    cropped = PILImage.open(convert_dataset.io.BytesIO(records[0]["image"]["bytes"]))
    assert cropped.size == (4, 5)
    assert cropped.getpixel((1, 1)) == (255, 0, 0)


def test_generate_character_dataset_records_clamps_crop_to_image_bounds(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    PILImage.new("RGB", (10, 10), color="black").save(image_path)

    annotation = convert_dataset.ImageAnnotation(
        image_id="page_001",
        book_id="book1",
        image_path=image_path,
        width=10,
        height=10,
        characters=[
            convert_dataset.CharAnnotation(
                unicode="U+4E00",
                x=8,
                y=7,
                width=5,
                height=6,
                block_id="",
                char_id="C0001",
            )
        ],
        columns=[],
        segments=[],
    )

    records = list(
        convert_dataset.generate_character_dataset_records(
            [annotation],
            {"U+4E00": 0},
        )
    )

    assert records[0]["crop_bbox"] == [8, 7, 2, 3]
    assert records[0]["width"] == 2
    assert records[0]["height"] == 3


def test_export_roboflow_yolov8_writes_only_images_with_column_annotations(tmp_path: Path) -> None:
    image_with_columns = tmp_path / "page_with_columns.jpg"
    image_without_columns = tmp_path / "page_without_columns.jpg"
    PILImage.new("RGB", (200, 100), color="white").save(image_with_columns)
    PILImage.new("RGB", (200, 100), color="white").save(image_without_columns)

    annotations = [
        convert_dataset.ImageAnnotation(
            image_id="page_with_columns",
            book_id="book1",
            image_path=image_with_columns,
            width=200,
            height=100,
            characters=[],
            columns=[
                convert_dataset.ColumnAnnotation(
                    column_id="COL0001",
                    bbox=[0.55, 0.4, 0.1, 0.4],
                    char_ids=["C0001"],
                    segment_id="SEG0001",
                )
            ],
            segments=[],
        ),
        convert_dataset.ImageAnnotation(
            image_id="page_without_columns",
            book_id="book1",
            image_path=image_without_columns,
            width=200,
            height=100,
            characters=[],
            columns=[],
            segments=[],
        ),
    ]

    output_dir = tmp_path / "roboflow-output"
    convert_dataset.export_roboflow_yolov8_dataset(
        annotations=annotations,
        output_dir=output_dir,
        dataset_name_prefix="kuzushiji-dataset",
    )

    dataset_dir = output_dir / "kuzushiji-dataset-roboflow-yolov8-columns"
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"

    assert (train_images / "page_with_columns.jpg").exists()
    assert (train_labels / "page_with_columns.txt").exists()
    assert not (train_images / "page_without_columns.jpg").exists()
    assert not (train_labels / "page_without_columns.txt").exists()
    assert (train_labels / "page_with_columns.txt").read_text(encoding="utf-8").strip() == "0 0.550000 0.400000 0.100000 0.400000"

    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == 1
    assert data_yaml["names"] == ["column"]
    assert data_yaml["train"] == "train/images"


def test_main_rejects_push_to_hub_for_roboflow_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_dataset.py",
            "--export-format",
            "roboflow",
            "--push-to-hub",
        ],
    )

    with pytest.raises(SystemExit):
        convert_dataset.main()


def test_create_character_dataset_features_has_expected_schema() -> None:
    features = convert_dataset.create_character_dataset_features()

    assert set(features.keys()) == {
        "image",
        "source_image_id",
        "book_id",
        "char_id",
        "block_id",
        "category",
        "category_id",
        "is_pua",
        "pua_code",
        "pua_reading",
        "pua_memo",
        "char",
        "bbox",
        "crop_bbox",
        "width",
        "height",
    }


def test_dataset_type_helpers() -> None:
    assert convert_dataset.should_generate_page_dataset("page") is True
    assert convert_dataset.should_generate_page_dataset("both") is True
    assert convert_dataset.should_generate_page_dataset("character") is False

    assert convert_dataset.should_generate_character_dataset("character") is True
    assert convert_dataset.should_generate_character_dataset("both") is True
    assert convert_dataset.should_generate_character_dataset("page") is False


def test_get_size_category_covers_large_character_datasets() -> None:
    assert convert_dataset.get_size_category(999) == "n<1K"
    assert convert_dataset.get_size_category(1_000) == "1K<n<10K"
    assert convert_dataset.get_size_category(100_000) == "100K<n<1M"
    assert convert_dataset.get_size_category(1_000_000) == "1M<n<10M"


def test_create_page_dataset_card_contains_detailed_standard_sections() -> None:
    card = convert_dataset.create_dataset_card(
        repo_id="example/kuzushiji-pages",
        bbox_format="coco",
        num_images=12_345,
        num_books=35,
        num_categories=4_321,
    )
    content = str(card)

    assert "pretty_name: Kuzushiji Page Dataset (COCO)" in content
    assert "# Dataset Card for Kuzushiji Page Dataset" in content
    assert "## Dataset Summary" in content
    assert "## Supported Tasks and Leaderboards" in content
    assert "### Data Instances" in content
    assert "### Data Fields" in content
    assert "### Data Splits" in content
    assert "| train | 12,345 |" in content
    assert "## Dataset Creation" in content
    assert "### Personal and Sensitive Information" in content
    assert "## Considerations for Using the Data" in content
    assert "### Discussion of Biases" in content
    assert "### Other Known Limitations" in content
    assert "## Additional Information" in content
    assert "### Licensing Information" in content
    assert "### Citation Information" in content


def test_create_character_dataset_card_contains_detailed_standard_sections() -> None:
    card = convert_dataset.create_character_dataset_card(
        repo_id="example/kuzushiji-characters",
        num_characters=123_456,
        num_books=35,
        num_categories=4_321,
    )
    content = str(card)

    assert "pretty_name: Kuzushiji Character Dataset" in content
    assert "size_categories:" in content
    assert "- 100K<n<1M" in content
    assert "# Dataset Card for Kuzushiji Character Dataset" in content
    assert "## Dataset Summary" in content
    assert "## Supported Tasks and Leaderboards" in content
    assert "### Data Instances" in content
    assert "### Data Fields" in content
    assert "### Data Splits" in content
    assert "| train | 123,456 |" in content
    assert "## Dataset Creation" in content
    assert "### Annotation Process" in content
    assert "### Personal and Sensitive Information" in content
    assert "## Considerations for Using the Data" in content
    assert "### Discussion of Biases" in content
    assert "### Other Known Limitations" in content
    assert "## Additional Information" in content
    assert "### Citation Information" in content


def test_main_rejects_character_dataset_type_for_roboflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_dataset.py",
            "--export-format",
            "roboflow",
            "--dataset-type",
            "character",
        ],
    )

    with pytest.raises(SystemExit):
        convert_dataset.main()
