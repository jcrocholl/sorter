from unittest.mock import MagicMock, patch
import pytest
from yolo_exporter import YoloExporter


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "yolo_output"


@pytest.fixture
def input_dir(tmp_path):
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()
    return input_dir


@pytest.fixture
def exporter(input_dir, output_dir):
    return YoloExporter(input_dir, output_dir)


def test_path_to_class(exporter, input_dir):
    assert exporter.path_to_class(input_dir / "broken" / "98282_brick") == "broken"
    assert (
        exporter.path_to_class(input_dir / "technic" / "3702_technic_brick")
        == "3702_technic_brick"
    )
    assert (
        exporter.path_to_class(input_dir / "minifig" / "minecraft" / "head")
        == "minifig_minecraft_head"
    )


def test_find_paths_with_jpg_files(exporter, input_dir):
    # Setup real file structure
    dir_with_jpg = input_dir / "dir1"
    dir_with_jpg.mkdir()
    (dir_with_jpg / "test.jpg").touch()

    empty_dir = input_dir / "dir2"
    empty_dir.mkdir()

    dir_with_yaml = input_dir / "dir3"
    dir_with_yaml.mkdir()
    (dir_with_yaml / "bricks.yaml").touch()

    results = exporter.find_paths_with_jpg_files(input_dir)
    assert dir_with_jpg in results
    assert empty_dir not in results
    assert dir_with_yaml not in results


def test_write_yaml(exporter, output_dir):
    exporter.class_names = ["3001_brick_2x4", "3002_brick_2x3"]
    exporter.num_train_per_class = {"3001_brick_2x4": 10, "3002_brick_2x3": 5}
    exporter.num_val_per_class = {"3001_brick_2x4": 2, "3002_brick_2x3": 1}
    exporter.num_test_per_class = {"3001_brick_2x4": 1, "3002_brick_2x3": 1}

    exporter.write_yaml()

    yaml_path = output_dir / "bricks.yaml"
    assert yaml_path.exists()

    content = yaml_path.read_text(encoding="utf-8")
    assert "train: bricks/images/train2023\n" in content
    assert "nc: 2\n" in content
    assert "3001_brick_2x4," in content
    assert "3002_brick_2x3," in content


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

        exporter.export_file(jpg_file, "3001_brick_2x4", "train2023")

        # Verify links and labels are created
        # We still mock os.link to avoid complexity of real hardlinking in tests
        # unless absolutely necessary, but we can check if label file was written.
        mock_link.assert_called_once()

        label_file = (
            output_dir
            / "labels"
            / "train2023"
            / "3001_brick_2x4"
            / "20231026_120000000_3001_brick_2x4.txt"
        )
        assert label_file.exists()

        # 15/640=0.023, 35/480=0.073, 10/640=0.016, 10/480=0.021
        content = label_file.read_text()
        assert "0 0.023 0.073 0.016 0.021" in content
        assert exporter.num_train_per_class["3001_brick_2x4"] == 1
        assert "3001_brick_2x4" in exporter.class_names
