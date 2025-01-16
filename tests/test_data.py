from torch.utils.data import Dataset
import pytest
from src.exam_project.data import melanoma_data
import os.path


@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data not processed")
def test_my_dataset():
    """Test the MyDataset class."""
    train, validation, test = melanoma_data()

    assert isinstance(train, Dataset)
    assert isinstance(validation, Dataset)
    assert isinstance(test, Dataset)

    for dataset in [train, validation, test]:
        for x,y in dataset:
            assert x.shape == (3, 224, 224)
            assert y == 0 or y == 1


# Test image format
# Test rgb image
