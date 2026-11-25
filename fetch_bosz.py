"""Download the BOSZ R500 spectra needed only to rebuild a cache."""
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from shutil import copyfileobj


def main():
    root = Path(__file__).resolve().parent
    for rel in (root / "bosz_spectra.txt").read_text().splitlines():
        dest = root / rel
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = "https://archive.stsci.edu/hlsps/bosz/bosz2024/" + quote(rel)
        with urlopen(url, timeout=120) as res, tmp.open("wb") as out:
            copyfileobj(res, out)
        tmp.replace(dest)


if __name__ == "__main__":
    main()
