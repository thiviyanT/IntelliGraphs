import os
import requests
import hashlib
import zipfile
from intelligraphs.info import DATASETS


class DatasetDownloader:
    def __init__(self, download_dir=".data"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_file(self, url, filename):
        """Download a file from Zenodo and save it locally."""
        filepath = os.path.join(self.download_dir, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping download.")
            return filepath

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filepath

    def verify_integrity(self, filepath, expected_md5):
        """Verify MD5 checksum of a downloaded file."""
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        computed_md5 = md5_hash.hexdigest()
        if computed_md5 == expected_md5:
            print(f"MD5 checksum verified for {os.path.basename(filepath)} ✓")
            return True
        else:
            print(f"MD5 mismatch for {os.path.basename(filepath)} ✗")
            print(f"Expected: {expected_md5}, Got: {computed_md5}")
            return False

    def extract_zip(self, filepath):
        """Extract a zip file."""
        print(f"Extracting {os.path.basename(filepath)}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.download_dir)

    def download_and_verify_all(self):
        """Download, verify, and extract all datasets."""
        for filename, data in DATASETS.items():
            file_path = self.download_file(data["url"], filename)
            if self.verify_integrity(file_path, data["md5"]):
                self.extract_zip(file_path)


# Example usage
if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_and_verify_all()
