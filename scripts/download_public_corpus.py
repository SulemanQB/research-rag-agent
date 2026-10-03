from pathlib import Path
from urllib.request import urlopen


MAX_DOCUMENT_BYTES = 50 * 1024 * 1024


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus" / "papers"
PUBLIC_DOCUMENTS = {
    "NIST.AI.100-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "NIST.AI.600-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
}


def download_documents() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in PUBLIC_DOCUMENTS.items():
        destination = CORPUS_DIR / filename
        print(f"Downloading {filename}...")
        with urlopen(url, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOCUMENT_BYTES:
                raise ValueError(f"{filename} exceeds the {MAX_DOCUMENT_BYTES} byte download limit")
            content = response.read(MAX_DOCUMENT_BYTES + 1)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"{filename} exceeds the {MAX_DOCUMENT_BYTES} byte download limit")
        if not content.startswith(b"%PDF-"):
            raise ValueError(f"{filename} did not contain a PDF response")
        destination.write_bytes(content)
        print(f"Saved {destination.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    download_documents()
