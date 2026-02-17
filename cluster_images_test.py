import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from cluster_images import (
    cluster_images_in_directory,
    main,
    parse_timestamp,
    cluster_and_filter_by_class,
    export_cluster,
)


def test_parse_timestamp():
    # Valid timestamp
    ts = parse_timestamp("20230523_105352366_foo.jpg")
    assert ts == datetime(2023, 5, 23, 10, 53, 52, 366000)

    # Invalid timestamps
    with pytest.raises(ValueError):
        parse_timestamp("invalid_filename.jpg")

    with pytest.raises(IndexError):
        parse_timestamp("20230523.jpg")


def test_cluster_images_basic(tmp_path):
    # Mock files
    (tmp_path / "20230523_105352000_a.jpg").touch()
    (tmp_path / "20230523_105352500_b.jpg").touch()  # 0.5s gap
    (tmp_path / "20230523_105354000_c.jpg").touch()  # 1.5s gap

    # default threshold is 0.5s.
    # a and b are 0.5s apart, so they should be clustered if threshold >= 0.5
    # c is 1.5s after b, so it should be in a new cluster

    clusters = cluster_images_in_directory(tmp_path, gap_threshold_seconds=0.5)

    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert clusters[0][0][1] == tmp_path / "20230523_105352000_a.jpg"
    assert clusters[0][1][1] == tmp_path / "20230523_105352500_b.jpg"
    assert len(clusters[1]) == 1
    assert clusters[1][0][1] == tmp_path / "20230523_105354000_c.jpg"


def test_cluster_images_no_files(tmp_path):
    clusters = cluster_images_in_directory(tmp_path)
    assert clusters == []


def test_cluster_images_invalid_filenames(tmp_path):
    (tmp_path / "invalid_name.jpg").touch()
    (tmp_path / "20230523_105352000_a.jpg").touch()

    clusters = cluster_images_in_directory(tmp_path)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1
    assert clusters[0][0][1] == tmp_path / "20230523_105352000_a.jpg"


def test_main_with_arguments(tmp_path, capsys):
    class_dir = tmp_path / "3001_brick_2x4"
    class_dir.mkdir()
    src = class_dir / "20230523_105352000_l10_r20_t30_b40_w10_h10.jpg"
    src.touch()

    mock_image = MagicMock()
    mock_image.width = 480
    mock_image.height = 640

    with (
        patch("PIL.Image.open") as mock_image_open,
        patch("os.link") as mock_link,
        patch("cluster_images.MIN_CLUSTERS_PER_CLASS", 1),
    ):
        mock_image_open.return_value.__enter__.return_value = mock_image
        output_dir = tmp_path / "yolo_dataset"
        main(["cluster_images.py", str(tmp_path), str(output_dir)])

        # Verify os.link was called with the correct arguments
        dst = (
            tmp_path
            / "yolo_dataset/images/train2023/3001_brick_2x4"
            / "20230523_105352000_3001_brick_2x4.jpg"
        )
        mock_link.assert_called_once_with(src, dst)

    captured = capsys.readouterr()
    assert "total clusters: 1" in captured.out
    assert "total images: 1" in captured.out
    assert "train: 1 clusters (100.0%), 1 images" in captured.out


def test_main_no_images(tmp_path, capsys):
    # Test main on an empty directory
    output_dir = tmp_path / "yolo_dataset"
    main(["cluster_images.py", str(tmp_path), str(output_dir)])

    captured = capsys.readouterr()
    assert "No images found to cluster." in captured.out


def test_process_images_aggregates_classes(tmp_path):
    # Setup: 2 dirs for same class "broken" with 10 and 15 images (total 25 > 20)
    (d1 := tmp_path / "root1" / "broken").mkdir(parents=True)
    (d2 := tmp_path / "root2" / "broken").mkdir(parents=True)

    for i in range(10):
        (d1 / f"20230523_1000{i:02d}000_a.jpg").touch()
    for i in range(15):
        (d2 / f"20230523_1100{i:02d}000_a.jpg").touch()

    # Mock YoloExporter
    exporter = MagicMock()

    def side_effect(p):
        if str(p) == str(d1.parent):
            return [d1]
        if str(p) == str(d2.parent):
            return [d2]
        return []

    exporter.find_paths_with_jpg_files.side_effect = side_effect
    exporter.path_to_class.return_value = "broken"

    # Verify aggregation (25 clusters > MIN_CLUSTERS_PER_CLASS=20)
    result = cluster_and_filter_by_class(exporter, [str(d1.parent), str(d2.parent)])
    assert len(result["broken"]) == 25

    # Verify filtering (reduce to 10 total < 20)
    for f in list(d1.glob("*.jpg"))[5:] + list(d2.glob("*.jpg"))[5:]:
        f.unlink()

    result = cluster_and_filter_by_class(exporter, [str(d1.parent), str(d2.parent)])
    assert "broken" not in result


def test_export_cluster_multi_image():
    exporter = MagicMock()
    cluster = [
        (datetime(2023, 1, 1), Path("img1.jpg")),
        (datetime(2023, 1, 2), Path("img2.jpg")),
        (datetime(2023, 1, 3), Path("img3.jpg")),
    ]

    # Test train split (should only export median)
    export_cluster(exporter, cluster, "class1", "train2023")
    assert exporter.export_file.call_count == 1
    exporter.export_file.assert_called_with(
        child=Path("img2.jpg"), class_name="class1", split="train2023"
    )

    exporter.export_file.reset_mock()

    # Test val split (should export median, first, last)
    export_cluster(exporter, cluster, "class1", "val2023")
    exporter.export_file.assert_has_calls(
        [
            call(child=Path("img2.jpg"), class_name="class1", split="val2023"),
            call(child=Path("img1.jpg"), class_name="class1", split="val2023"),
            call(child=Path("img3.jpg"), class_name="class1", split="val2023"),
        ],
        any_order=False,
    )


def test_per_class_split(tmp_path):
    # Create valid class directory and images
    class_dir = tmp_path / "class_a"
    class_dir.mkdir()
    # create 10 clusters, each with 3 images
    for i in range(10):
        for j in range(3):
            # 1 ms gap between images in same cluster
            # 1 min gap between clusters
            ts = f"20230523_10{i:02d}00{j:03d}"
            (class_dir / f"{ts}_foo.jpg").touch()

    # Mock exporter
    exporter = MagicMock()
    exporter.find_paths_with_jpg_files.return_value = [class_dir]
    exporter.path_to_class.return_value = "class_a"

    # Mock MIN_CLUSTERS_PER_CLASS to allow small number of clusters
    with (
        patch("cluster_images.MIN_CLUSTERS_PER_CLASS", 1),
        patch("cluster_images.YoloExporter", return_value=exporter),
    ):
        output_dir = tmp_path / "yolo_dataset"
        main(["cluster_images.py", str(tmp_path), str(output_dir)])

    # 10 clusters:
    # val target = 10 * 0.25 = 2.5 -> 2
    # test target = 10 * 0.25 = 2.5 -> 2
    # train target = remaining = 6

    # Check that export_file was called with correct splits
    splits = [c.kwargs["split"] for c in exporter.export_file.call_args_list]
    assert (
        splits.count("val2023") >= 2
    )  # At least 2, could be more due to multi-image export
    assert splits.count("test2023") >= 2
    assert splits.count("train2023") == 6  # Train only exports 1 image per cluster

    # Verify basic split assignment logic (approximate check via string counts)
    # val: 2 clusters -> 6 export calls (median + first + last)
    # test: 2 clusters -> 6 export calls
    # train: 6 clusters -> 6 export calls
    # Total calls = 18
    assert exporter.export_file.call_count == 18
