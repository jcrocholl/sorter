from unittest.mock import MagicMock
import numpy as np
import pytest
from brick_camera import BrickCamera, Hypothesis


def make_mock_model(detections: list[list[float]]) -> MagicMock:
    """Build a mock yolov7 model that returns the given detections.

    Each detection is a row: [x_center, y_center, width, height, conf, cls_idx].
    """
    import torch

    tensor = torch.tensor(detections, dtype=torch.float32)
    names = ["3001_brick_2x4", "3003_brick_2x2", "reject"]

    result = MagicMock()
    result.xywhn = [tensor]
    result.names = names

    model = MagicMock(return_value=result)
    return model


@pytest.fixture
def fake_img() -> np.ndarray:
    """A minimal 4x4 BGR image suitable for passing to recognize()."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_recognize_returns_hypotheses(fake_img):
    model = make_mock_model(
        [
            # x_center, y_center, width, height, conf, cls_idx
            [0.5, 0.5, 0.3, 0.4, 0.92, 0],
            [0.2, 0.7, 0.1, 0.1, 0.65, 2],
        ]
    )

    camera = BrickCamera(model)
    hypotheses = camera.recognize(fake_img, capture_timestamp=1.0)

    assert len(hypotheses) == 2

    h0 = hypotheses[0]
    assert isinstance(h0, Hypothesis)
    assert h0.confidence == pytest.approx(0.92)
    assert h0.x_center == pytest.approx(0.5)
    assert h0.y_center == pytest.approx(0.5)
    assert h0.width == pytest.approx(0.3)
    assert h0.height == pytest.approx(0.4)
    assert h0.class_id == 0
    assert h0.class_name == "3001_brick_2x4"

    h1 = hypotheses[1]
    assert h1.confidence == pytest.approx(0.65)
    assert h1.class_id == 2
    assert h1.class_name == "reject"


def test_recognize_empty(fake_img):
    import torch

    result = MagicMock()
    result.xywhn = [torch.zeros((0, 6))]
    result.names = ["3001_brick_2x4"]

    model = MagicMock(return_value=result)
    camera = BrickCamera(model)
    hypotheses = camera.recognize(fake_img, capture_timestamp=0.0)

    assert hypotheses == []
