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

    def check_datasets_exist(self):
        """Check if all expected dataset files exist in the download directory."""
        missing_files = []
        for filename in DATASETS:
            filepath = os.path.join(self.download_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
                print(f"Missing dataset: {filename}")

        if missing_files:
            print(f"\nMissing {len(missing_files)} of {len(DATASETS)} datasets")
            return False

        print(f"\nAll {len(DATASETS)} datasets present")
        return True

    def verify_datasets(self):
        verification_report = {
            "total_datasets": len(DATASETS),
            "verified_datasets": 0,
            "missing_files": [],
            "checksum_failures": [],
            "extraction_failures": [],
            "verified_files": []
        }

        for filename, data in DATASETS.items():
            filepath = os.path.join(self.download_dir, filename)

            if not os.path.exists(filepath):
                verification_report["missing_files"].append(filename)
                continue

            if not self.verify_integrity(filepath, data["md5"]):
                verification_report["checksum_failures"].append(filename)
                continue

            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    test_result = zip_ref.testzip()
                    if test_result is not None:
                        verification_report["extraction_failures"].append(
                            f"{filename} (corrupted file: {test_result})"
                        )
                        continue

                    if "expected_contents" in data:
                        zip_contents = set(zip_ref.namelist())
                        missing_contents = set(data["expected_contents"]) - zip_contents
                        if missing_contents:
                            verification_report["extraction_failures"].append(
                                f"{filename} (missing files: {', '.join(missing_contents)})"
                            )
                            continue

            except zipfile.BadZipFile:
                verification_report["extraction_failures"].append(
                    f"{filename} (invalid zip file)"
                )
                continue

            verification_report["verified_files"].append(filename)
            verification_report["verified_datasets"] += 1

        self._print_verification_summary(verification_report)
        return verification_report["verified_datasets"] == verification_report["total_datasets"]

    def _print_verification_summary(self, report):
        """ Print a formatted summary of the dataset verification results."""
        print("\n=== Dataset Verification Summary ===")
        if report['verified_files']:
            print("\nVerified files:")
            for file in report['verified_files']:
                print(f"  ✓ {file}")

        if report['missing_files']:
            print("\nMissing files:")
            for file in report['missing_files']:
                print(f"  ✗ {file}")

        if report['checksum_failures']:
            print("\nChecksum failures:")
            for file in report['checksum_failures']:
                print(f"  ✗ {file}")

        if report['extraction_failures']:
            print("\nExtraction failures:")
            for file in report['extraction_failures']:
                print(f"  ✗ {file}")


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_and_verify_all()
