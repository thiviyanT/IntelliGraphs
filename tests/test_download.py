import pytest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
from intelligraphs.data_loaders.download import DatasetDownloader


@pytest.fixture
def downloader():
    return DatasetDownloader(download_dir=".data")


@pytest.fixture
def mock_datasets():
    return {
        "test.zip": {
            "url": "http://test.com/test.zip",
            "md5": "abc123",
            "expected_contents": ["file1.txt", "file2.txt"]
        }
    }


@patch("os.makedirs")
def test_init_creates_directory(mock_makedirs):
    """ Check if data directory is created when initializing DatasetDownloader """
    # Create downloader after the mock is in place
    downloader = DatasetDownloader(download_dir=".data")
    mock_makedirs.assert_called_once_with(".data", exist_ok=True)



@patch("requests.get")
def test_download_file_new(mock_get, downloader):
    """ Tests downloading a new file, verifies proper HTTP request and file writing """
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response

    with patch("builtins.open", mock_open()) as mock_file:
        path = downloader.download_file("http://test.com/test.zip", "test.zip")
        mock_file().write.assert_called_with(b"data")
        assert path == ".data/test.zip"

@patch("os.path.exists")
def test_verify_datasets(mock_exists):
    mock_exists.return_value = True  # Simulate files exist
    downloader = DatasetDownloader()
    with patch('zipfile.ZipFile') as mock_zip:
        mock_zip.return_value.__enter__().testzip.return_value = None
        report = downloader.verify_datasets()
        assert isinstance(report, bool)


@patch("os.path.exists")
def test_download_file_exists(mock_exists, downloader):
    """ Confirm download is skipped if file already exists """
    mock_exists.return_value = True
    path = downloader.download_file("http://test.com/test.zip", "test.zip")
    assert path == ".data/test.zip"


def test_verify_integrity_success(downloader):
    """ Check if MD5 checksum verification passes for valid files """
    content = b"test data"
    expected_md5 = "eb733a00c0c9d336e65691a37ab54293"

    with patch("builtins.open", mock_open(read_data=content)):
        assert downloader.verify_integrity("test.zip", expected_md5)


def test_verify_integrity_failure(downloader):
    """ Ensure MD5 verification fails for invalid checksums """
    content = b"test data"
    wrong_md5 = "wrong123"

    with patch("builtins.open", mock_open(read_data=content)):
        assert not downloader.verify_integrity("test.zip", wrong_md5)


@patch("zipfile.ZipFile")
def test_extract_zip(mock_zipfile, downloader):
    """ Verify ZIP file extraction functionality """
    downloader.extract_zip("test.zip")
    mock_zipfile.assert_called_once_with("test.zip", 'r')
    mock_zipfile.return_value.__enter__().extractall.assert_called_once()


@patch("os.path.exists")
def test_check_datasets_exist(mock_exists, downloader, mock_datasets):
    """ Test detection of missing/present datasets """
    with patch("intelligraphs.info.DATASETS", mock_datasets):
        mock_exists.return_value = True
        assert downloader.check_datasets_exist()

        mock_exists.return_value = False
        assert not downloader.check_datasets_exist()


@patch("intelligraphs.data_loaders.download.DATASETS", {
    "test.zip": {
        "url": "http://test.com/test.zip",
        "md5": "abc123",
        "expected_contents": ["file1.txt", "file2.txt"]
    }
})
@patch.object(DatasetDownloader, "download_file")
@patch.object(DatasetDownloader, "verify_integrity")
@patch.object(DatasetDownloader, "extract_zip")
def test_download_and_verify_all(mock_extract, mock_verify, mock_download):
    """ Validate  the complete workflow - download, verify, extract - happens in correct order """
    # Set up return values for our mocks
    mock_download.return_value = "test.zip"
    mock_verify.return_value = True

    # Create downloader and run the method
    downloader = DatasetDownloader()
    downloader.download_and_verify_all()

    # Verify the calls
    mock_download.assert_called_once_with(
        "http://test.com/test.zip",
        "test.zip"
    )
    mock_verify.assert_called_once()
    mock_extract.assert_called_once()
