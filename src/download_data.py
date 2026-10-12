"""Download the agreed KOCOH release without silently replacing local data."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'c26d8e86f3d7a9a6dcd4e308cb747b6bc6427d6f'
URL = f'https://huggingface.co/datasets/E-Park/KOCOH/resolve/{REVISION}/KOCOH_v.2.csv'
DATA = ROOT / 'data' / 'KOCOH_v.2.csv'


def download():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        raw = DATA.read_bytes()
        print('Existing file retained:', DATA)
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw = response.read()
        # Decode/parse before saving so an HTML error is never accepted as CSV.
        import csv
        import io
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        if len(rows) != 3000 or 'Hate speech' not in rows[0]:
            raise ValueError('Unexpected download; expected KOCOH v2 with 3,000 rows.')
        DATA.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = ROOT / 'data' / 'provenance.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['sha256'] != digest:
            raise ValueError('Local CSV hash differs from provenance.json; original retained.')
    else:
        # Verify an existing unregistered file against the pinned upstream revision.
        with urllib.request.urlopen(URL, timeout=60) as response:
            upstream_hash = hashlib.sha256(response.read()).hexdigest()
        if digest != upstream_hash:
            raise ValueError('Local CSV is not the pinned release; original retained.')
        manifest.write_text(json.dumps({'dataset': 'KOCOH v2', 'revision': REVISION,
            'url': URL, 'sha256': digest, 'bytes': len(raw)}, indent=2) + '\n', encoding='utf-8')
    print('SHA256:', digest)
    return DATA


if __name__ == '__main__':
    download()
