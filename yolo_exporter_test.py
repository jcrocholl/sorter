import pathlib
from unittest.mock import MagicMock, patch, mock_open
import pytest
from yolo_exporter import YoloExporter


@pytest.fixture
def exporter():
    return YoloExporter(pathlib.Path("/tmp/yolo_output"))


def test_train_test_split(exporter):
    mock_file = MagicMock(spec=pathlib.Path)
    mock_file.name = "20231026_120008"
    assert exporter.train_test_split(mock_file) == "val2023"

    mock_file.name = "20231026_120009"
    assert exporter.train_test_split(mock_file) == "test2023"

    mock_file.name = "20231026_120007"
    assert exporter.train_test_split(mock_file) == "train2023"


def test_path_to_class(exporter):
    assert exporter.path_to_class(pathlib.Path("broken/98282_brick")) == "broken"
    assert (
        exporter.path_to_class(pathlib.Path("3702_technic_brick"))
        == "3702_technic_brick"
    )
    assert exporter.path_to_class(pathlib.Path("minifig/head")) == "minifig_head"


def test_find_paths_with_jpg_files(exporter):
    mock_dir = MagicMock(spec=pathlib.Path)
    mock_dir.is_dir.return_value = True

    mock_file = MagicMock(spec=pathlib.Path)
    mock_file.is_dir.return_value = False
    mock_file.is_file.return_value = True
    mock_file.name = "test.jpg"

    mock_dir.iterdir.return_value = [mock_file]

    paths = exporter.find_paths_with_jpg_files(mock_dir)
    assert mock_dir in paths


def test_write_yaml(exporter):
    exporter.class_names = ["class1", "class2"]
    exporter.num_train_per_class = {"class1": 10, "class2": 5}
    exporter.num_val_per_class = {"class1": 2, "class2": 1}
    exporter.num_test_per_class = {"class1": 1, "class2": 1}

    m = mock_open()
    with patch("builtins.open", m):
        with patch("pathlib.Path.mkdir"):
            exporter.write_yaml()

    m.assert_called_once_with(
        exporter.output_dir / "data" / "bricks.yaml", "wt", encoding="utf-8"
    )
    handle = m()

    # Collect all written calls
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert "train: bricks/images/train2023\n" in written_content
    assert "nc: 2\n" in written_content
    assert "class1," in written_content
    assert "class2," in written_content


def test_export_file(exporter):
    mock_image = MagicMock()
    mock_image.width = 640
    mock_image.height = 480

    mock_jpg = MagicMock(spec=pathlib.Path)
    mock_jpg.name = "20231026_120000000_l10_r20_t30_b40_.jpg"

    m = mock_open()
    with (
        patch("PIL.Image.open") as mock_image_open,
        patch("os.link") as mock_link,
        patch("builtins.open", m),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.exists", return_value=False),
    ):
        mock_image_open.return_value.__enter__.return_value = mock_image
        exporter.export_file(mock_jpg, "class1", 0, "train2023")

        mock_link.assert_called_once()
        m.assert_called_once()
        handle = m()
        handle.write.assert_called_once()
        # 15/640=0.023, 35/480=0.073, 10/640=0.016, 10/480=0.021
        assert "0 0.023 0.073 0.016 0.021" in handle.write.call_args[0][0]
        assert exporter.num_train_per_class["class1"] == 1


def test_export_dir(exporter):
    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.parts = ["bricks"]
    mock_path.name = "1234_brick"

    mock_jpg = MagicMock(spec=pathlib.Path)
    mock_jpg.name = "20231026_120000000_l10_r20_t30_b40_.jpg"
    mock_jpg.is_file.return_value = True

    mock_path.iterdir.return_value = [mock_jpg]

    with patch.object(exporter, "export_file") as mock_export_file:
        exporter.export_dir(mock_path)
        assert "1234_brick" in exporter.class_names
        mock_export_file.assert_called_once_with(mock_jpg, "1234_brick", 0, "train2023")
