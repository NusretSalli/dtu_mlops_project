import pytest
import torch

from src.exam_project.model import ResNet18


def test_model():
    dummy_input = torch.randn(1, 3, 224, 224)
    assert ResNet18()(dummy_input).shape == (1, 2)


def test_error_on_wrong_shape():
    model = ResNet18()
    with pytest.raises(ValueError, match="Expected input to have 4D tensor"):
        model(torch.randn(1, 3, 224))

#Test output shape of model
