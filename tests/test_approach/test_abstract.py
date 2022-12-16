from unittest import mock

import pytest

from rul_adapt.approach.abstract import AdaptionApproach


class DummyApproach(AdaptionApproach):
    def __init__(self):
        super().__init__()


def test_set_model():
    mock_feature_extractor = mock.MagicMock()
    mock_regressor = mock.MagicMock()
    approach = DummyApproach()

    approach.set_model(mock_feature_extractor, mock_regressor)

    assert approach.feature_extractor is mock_feature_extractor
    assert approach.regressor is mock_regressor


def test_feature_extractor():
    approach = DummyApproach()

    with pytest.raises(RuntimeError):
        _ = approach.feature_extractor


def test_regressor():
    approach = DummyApproach()

    with pytest.raises(RuntimeError):
        _ = approach.regressor


@pytest.mark.integration
def test_dummy_integration():
    """This is a dummy test so that the integration workflow does not report missing
    test cases when no approach has integration tests yet. """
    pass
