from pathlib import Path
import sys

import pandas as pd

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
