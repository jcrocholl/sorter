import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from cluster_images import cluster_images, main, parse_timestamp


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

    clusters = cluster_images(tmp_path, gap_threshold_seconds=0.5)

    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert clusters[0][0][1] == tmp_path / "20230523_105352000_a.jpg"
    assert clusters[0][1][1] == tmp_path / "20230523_105352500_b.jpg"
    assert len(clusters[1]) == 1
    assert clusters[1][0][1] == tmp_path / "20230523_105354000_c.jpg"


def test_cluster_images_no_files(tmp_path):
    clusters = cluster_images(tmp_path)
    assert clusters == []


def test_cluster_images_invalid_filenames(tmp_path):
    (tmp_path / "invalid_name.jpg").touch()
    (tmp_path / "20230523_105352000_a.jpg").touch()

    clusters = cluster_images(tmp_path)
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
        patch("cluster_images.MIN_CLUSTERS_PER_DIR", 1),
    ):
        mock_image_open.return_value.__enter__.return_value = mock_image
        # Test main with the tmp_path as an argument
        main(["cluster_images.py", str(tmp_path)])

        # Verify os.link was called with the correct arguments
        dst = (
            Path("../yolo_dataset/images/train2023/3001_brick_2x4")
            / "20230523_105352000_3001_brick_2x4.jpg"
        )
        mock_link.assert_called_once_with(src, dst)

    captured = capsys.readouterr()
    assert "total clusters: 1" in captured.out
    assert "total images: 1" in captured.out
    assert "train: 1 clusters (100.0%), 1 images" in captured.out


def test_main_no_images(tmp_path, capsys):
    # Test main on an empty directory
    main(["cluster_images.py", str(tmp_path)])

    captured = capsys.readouterr()
    assert "No images found to cluster." in captured.out
