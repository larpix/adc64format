import pytest
import os

test_dir = os.path.dirname(__file__)
_example_files = [
    os.path.join(test_dir, 'test_0a7a314c.data'),
    os.path.join(test_dir, 'test_0a7b54bd.data')
]


@pytest.fixture
def example_files():
    return _example_files


@pytest.fixture(params=_example_files)
def example_file(request):
    return request.param
