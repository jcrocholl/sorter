import pathlib
from unittest.mock import MagicMock, patch
import pytest
from yolo_exporter import YoloExporter


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "yolo_output"


@pytest.fixture
def exporter(output_dir):
    return YoloExporter(output_dir)


def test_train_test_split(exporter):
    assert (
        exporter.train_test_split(pathlib.Path("20231026_120008_abc.jpg")) == "val2023"
    )
    assert (
        exporter.train_test_split(pathlib.Path("20231026_120009_def.jpg")) == "test2023"
    )
    assert (
        exporter.train_test_split(pathlib.Path("20231026_120007_ghi.jpg"))
        == "train2023"
    )


def test_path_to_class(exporter):
    assert exporter.path_to_class(pathlib.Path("broken/98282_brick")) == "broken"
    assert (
        exporter.path_to_class(pathlib.Path("3702_technic_brick"))
        == "3702_technic_brick"
    )
    assert exporter.path_to_class(pathlib.Path("minifig/head")) == "minifig_head"


def test_find_paths_with_jpg_files(exporter, tmp_path):
    # Setup real file structure
    dir_with_jpg = tmp_path / "dir1"
    dir_with_jpg.mkdir()
    (dir_with_jpg / "test.jpg").touch()

    empty_dir = tmp_path / "dir2"
    empty_dir.mkdir()

    results = exporter.find_paths_with_jpg_files(tmp_path)
    assert dir_with_jpg in results
    assert empty_dir not in results


def test_write_yaml(exporter, output_dir):
    exporter.class_names = ["class1", "class2"]
    exporter.num_train_per_class = {"class1": 10, "class2": 5}
    exporter.num_val_per_class = {"class1": 2, "class2": 1}
    exporter.num_test_per_class = {"class1": 1, "class2": 1}

    exporter.write_yaml()

    yaml_path = output_dir / "data" / "bricks.yaml"
    assert yaml_path.exists()

    content = yaml_path.read_text(encoding="utf-8")
    assert "train: bricks/images/train2023\n" in content
    assert "nc: 2\n" in content
    assert "class1," in content
    assert "class2," in content


def test_export_file(exporter, tmp_path, output_dir):
    # Setup dummy source image
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    jpg_file = src_dir / "20231026_120000000_l10_r20_t30_b40_.jpg"
    jpg_file.touch()

    mock_image = MagicMock()
    mock_image.width = 640
    mock_image.height = 480

    with (
        patch("PIL.Image.open") as mock_image_open,
        patch("os.link") as mock_link,
    ):
        mock_image_open.return_value.__enter__.return_value = mock_image

        exporter.export_file(jpg_file, "class1", 0, "train2023")

        # Verify links and labels are created
        # We still mock os.link to avoid complexity of real hardlinking in tests
        # unless absolutely necessary, but we can check if label file was written.
        mock_link.assert_called_once()

        label_file = (
            output_dir
            / "bricks"
            / "labels"
            / "train2023"
            / "20231026_120000000_class1.txt"
        )
        assert label_file.exists()

        # 15/640=0.023, 35/480=0.073, 10/640=0.016, 10/480=0.021
        content = label_file.read_text()
        assert "0 0.023 0.073 0.016 0.021" in content
        assert exporter.num_train_per_class["class1"] == 1


def test_export_dir(exporter, tmp_path):
    class_dir = tmp_path / "1234_brick"
    class_dir.mkdir()
    jpg_file = class_dir / "20231026_120000000_l10_r20_t30_b40_.jpg"
    jpg_file.touch()

    with patch.object(exporter, "export_file") as mock_export_file:
        exporter.export_dir(class_dir)
        assert "1234_brick" in exporter.class_names
        mock_export_file.assert_called_once_with(jpg_file, "1234_brick", 0, "train2023")
