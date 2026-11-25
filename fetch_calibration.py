"""Fetch the CALSPEC reference used by the paper."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

name = "alpha_lyr_stis_012.fits"
url = "https://ssb.stsci.edu/trds/calspec/" + name
root = Path(__file__).resolve().parent / "calibration"
root.mkdir(exist_ok=True)
with urlopen(url, timeout=90) as response:
    content = response.read()
if sha256(content).hexdigest() != "7201cd8566d53d384430c3d5fd2ae8d8122e9a7f765711748b49336f63237672":
    raise ValueError("Unexpected CALSPEC checksum")
(root / name).write_bytes(content)
