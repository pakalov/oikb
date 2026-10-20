import json
from unittest.mock import Mock

import httpx
import pytest
import respx
from click.testing import CliRunner

from oikb.client import OikbClient
from oikb.connectors import BaseConnector, ManifestEntry
from oikb.kb_sync import group_entries_by_kb, run_entries_sync
from oikb.sync import SyncResult

BASE = "https://webui.example/api/v1"


class Source(BaseConnector):
    def __init__(self, content):
        self.content = content

    def build_manifest(self):
        return [ManifestEntry(name, "", str(len(data)), len(data)) for name, data in self.content.items()]

    def read_file(self, path, filename):
        return self.content[filename]


def _client():
    return OikbClient("https://webui.example", "token")


@respx.mock
def test_default_upload_is_a_single_processed_request():
    upload = respx.post(f"{BASE}/files/").mock(return_value=httpx.Response(200, json={"id": "f"}))

    _client().upload_file(b"data", "a.txt", "kb", "hash")

    assert upload.call_count == 1
    assert "process" not in upload.calls.last.request.url.params
    assert len(respx.calls) == 1


@respx.mock
def test_unprocessed_upload_stores_stubs_and_links_the_file():
    upload = respx.post(f"{BASE}/files/").mock(return_value=httpx.Response(200, json={"id": "f1"}))
    stub = respx.post(f"{BASE}/files/f1/data/content/update").mock(return_value=httpx.Response(200, json={}))
    link = respx.post(f"{BASE}/knowledge/kb/file/add").mock(return_value=httpx.Response(200, json={}))

    result = _client().upload_file(b"raw bytes", "big.xlsx", "kb", "hash", directory_id="dir", process=False)

    assert result["id"] == "f1"
    assert upload.calls.last.request.url.params["process"] == "false"
    assert b'"knowledge_id": "kb"' in upload.calls.last.request.content
    assert b'"file_hash": "hash"' in upload.calls.last.request.content
    content = json.loads(stub.calls.last.request.content)["content"]
    assert "big.xlsx" in content and "f1" in content
    assert json.loads(link.calls.last.request.content) == {"file_id": "f1", "directory_id": "dir"}
    assert [call.request.url.path for call in respx.calls] == [
        "/api/v1/files/",
        "/api/v1/files/f1/data/content/update",
        "/api/v1/knowledge/kb/file/add",
    ]


@respx.mock
def test_unprocessed_upload_omits_directory_at_kb_root():
    respx.post(f"{BASE}/files/").mock(return_value=httpx.Response(200, json={"id": "f1"}))
    respx.post(f"{BASE}/files/f1/data/content/update").mock(return_value=httpx.Response(200, json={}))
    link = respx.post(f"{BASE}/knowledge/kb/file/add").mock(return_value=httpx.Response(200, json={}))

    _client().upload_file(b"raw", "a.csv", "kb", "hash", process=False)

    assert json.loads(link.calls.last.request.content) == {"file_id": "f1"}


@pytest.mark.parametrize("failing_step", ["stub", "link"])
@respx.mock
def test_unprocessed_upload_deletes_the_file_when_linking_fails(failing_step):
    respx.post(f"{BASE}/files/").mock(return_value=httpx.Response(200, json={"id": "f1"}))
    respx.post(f"{BASE}/files/f1/data/content/update").mock(
        return_value=httpx.Response(500 if failing_step == "stub" else 200, json={"detail": "boom"})
    )
    link = respx.post(f"{BASE}/knowledge/kb/file/add").mock(
        return_value=httpx.Response(400 if failing_step == "link" else 200, json={"detail": "boom"})
    )
    delete = respx.delete(f"{BASE}/files/f1").mock(return_value=httpx.Response(200, json=True))

    with pytest.raises(httpx.HTTPStatusError):
        _client().upload_file(b"raw", "a.xlsx", "kb", "hash", process=False)

    assert delete.called
    assert link.called is (failing_step == "link")


@respx.mock
def test_unprocessed_upload_keeps_the_original_error_when_cleanup_fails():
    respx.post(f"{BASE}/files/").mock(return_value=httpx.Response(200, json={"id": "f1"}))
    respx.post(f"{BASE}/files/f1/data/content/update").mock(return_value=httpx.Response(200, json={}))
    respx.post(f"{BASE}/knowledge/kb/file/add").mock(return_value=httpx.Response(400, json={"detail": "link failed"}))
    respx.delete(f"{BASE}/files/f1").mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _client().upload_file(b"raw", "a.xlsx", "kb", "hash", process=False)

    assert excinfo.value.response.status_code == 400


@pytest.mark.parametrize(("entry", "expected"), [({}, True), ({"process": True}, True), ({"process": False}, False)])
def test_entry_process_flag_reaches_every_upload(entry, expected):
    client = Mock()
    client.sync_diff.side_effect = lambda kb, manifest: {"added": manifest}

    result = run_entries_sync(
        client,
        [{"source": "one", "kb-id": "kb", **entry}],
        resolve_connector=lambda *a, **kw: Source({"a.xlsx": b"a", "b.csv": b"b"}),
        quiet=True,
    )

    assert result.added == 2
    assert [call.kwargs["process"] for call in client.upload_file.call_args_list] == [expected, expected]


def test_kb_group_must_agree_on_process():
    with pytest.raises(ValueError, match="process"):
        group_entries_by_kb([
            {"source": "one", "kb-id": "kb", "process": False},
            {"source": "two", "kb-id": "kb"},
        ])
    groups = group_entries_by_kb([
        {"source": "one", "kb-id": "kb", "process": True},
        {"source": "two", "kb-id": "kb"},
    ])
    assert len(groups) == 1


@pytest.mark.parametrize("value", ["false", 0, None])
def test_process_must_be_a_boolean(value):
    with pytest.raises(ValueError, match="process must be true or false"):
        group_entries_by_kb([{"source": "one", "kb-id": "kb", "process": value}])


@pytest.mark.parametrize(("args", "expected"), [([], True), (["--no-process"], False)])
def test_cli_no_process_flag_for_single_source(monkeypatch, args, expected):
    import oikb.cli as cli
    import oikb.sync as sync_module

    monkeypatch.setattr(cli, "_resolve_connector", lambda *a, **kw: Source({}))
    monkeypatch.setattr(cli, "_make_client", lambda *a: Mock())
    run_sync = Mock(return_value=SyncResult())
    monkeypatch.setattr(sync_module, "run_sync", run_sync)

    result = CliRunner().invoke(cli.cli, ["sync", "./docs", "--kb-id", "kb", *args])

    assert result.exit_code == 0, result.output
    assert run_sync.call_args.kwargs["process"] is expected
