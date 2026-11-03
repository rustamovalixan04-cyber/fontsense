"""Build-only verified independent-font assets; no archive paths are extracted."""
from functools import lru_cache
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError
import time
import zipfile

from scripts.extended_font_assets import make_subset

ALLOWED_HOSTS = {'raw.githubusercontent.com', 'get.fontspace.co'}
MAX_DOWNLOAD = 32 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024


@lru_cache(maxsize=64)
def payload(url, digest):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError('Unapproved independent-font source')
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'FontSense-verified-font-build'}),timeout=60) as response:
                data = response.read(MAX_DOWNLOAD + 1)
            break
        except (URLError, TimeoutError):
            if attempt == 2: raise
            time.sleep(1)
    if len(data) > MAX_DOWNLOAD or (digest is not None and hashlib.sha256(data).hexdigest() != digest):
        raise ValueError('Independent-font download checksum or size mismatch: ' + url)
    if digest is None and not zipfile.is_zipfile(BytesIO(data)):
        raise ValueError('Independent-font source returned an invalid ZIP: ' + url)
    return data


def member(archive, name):
    with zipfile.ZipFile(BytesIO(archive)) as package:
        if len(package.infolist()) > 1000 or sum(info.file_size for info in package.infolist()) > 64 * 1024 * 1024:
            raise ValueError('Oversized independent-font archive')
        info = package.getinfo(name)
        if info.is_dir() or info.file_size > MAX_MEMBER:
            raise ValueError('Invalid independent-font archive member')
        return package.read(info)


def prepare(source, root):
    relative = Path(source['file'])
    license_relative = Path(source['license_file'])
    if any(path.is_absolute() or '..' in path.parts or path.parts[0] != 'independent-fonts' for path in (relative,license_relative)):
        raise ValueError('Invalid independent-font asset path')
    target, license_target = root / relative, root / license_relative
    if (target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == source['sha256']
            and license_target.exists() and hashlib.sha256(license_target.read_bytes()).hexdigest() == source['license_sha256']):
        return
    if source['kind'] == 'archive':
        # FontSpace can rebuild the ZIP wrapper (timestamps/extra documents).
        # Pin the exact selected font and license bytes below, independently of
        # that wrapper. The observed archive hash remains in the source audit.
        archive = payload(source['download_url'],None)
        raw, license_data = member(archive,source['member']),member(archive,source['license_member'])
    else:
        raw = payload(source['download_url'],source['source_sha256'])
        license_data = payload(source['license_url'],source['license_sha256'])
    if hashlib.sha256(raw).hexdigest() != source['source_sha256'] or hashlib.sha256(license_data).hexdigest() != source['license_sha256']:
        raise ValueError('Independent-font source or license checksum mismatch: ' + source['id'])
    converted = make_subset(raw,source['id'])
    if hashlib.sha256(converted).hexdigest() != source['sha256']:
        raise ValueError('Independent-font subset checksum mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(converted)
    license_target.write_bytes(license_data)
    if source['kind'] == 'archive' and hashlib.sha256(archive).hexdigest() != source['archive_sha256']:
        print('Verified repackaged font and license components:', source['id'], flush=True)
