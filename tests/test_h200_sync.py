import hashlib
import sys
from types import SimpleNamespace
from h200 import sync


class FakeS3:
    def __init__(self, body, metadata):
        self.head = {"ContentLength": len(body), "Metadata": metadata}
        self.uploads = []

    def head_object(self, **kwargs):
        return self.head

    def upload_file(self, path, bucket, key, ExtraArgs):
        from pathlib import Path
        self.uploads.append(key)
        self.head = {"ContentLength": Path(path).stat().st_size,
                     "Metadata": ExtraArgs["Metadata"]}


def run_sync(monkeypatch, tmp_path, client):
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *a, **kw: client))
    monkeypatch.setattr(sys, "argv", ["sync", "--root", str(tmp_path), "--prefix", "test"])
    sync.main()


def test_changed_same_length_summary_is_uploaded_then_unchanged_file_is_skipped(tmp_path, monkeypatch):
    old, new = b'{"n":183}', b'{"n":198}'
    (tmp_path / "summary.json").write_bytes(new)
    client = FakeS3(old, {"sha256": hashlib.sha256(old).hexdigest()})
    run_sync(monkeypatch, tmp_path, client)
    assert client.uploads == ["test/summary.json"]
    assert client.head["Metadata"]["sha256"] == hashlib.sha256(new).hexdigest()
    run_sync(monkeypatch, tmp_path, client)
    assert len(client.uploads) == 1


def test_legacy_object_without_checksum_is_refreshed(tmp_path, monkeypatch):
    body = b'unchanged'
    (tmp_path / "config.json").write_bytes(body)
    client = FakeS3(body, {})
    run_sync(monkeypatch, tmp_path, client)
    assert client.uploads == ["test/config.json"]
