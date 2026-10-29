import io
import json

import pytest

from scripts.shared_queue import SharedQueue


class Conflict(Exception):
    def __init__(self): self.response = {'Error': {'Code': 'PreconditionFailed'}}


class FakeS3:
    def __init__(self): self.rows = {}; self.version = 0
    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, Metadata=None):
        old = self.rows.get((Bucket, Key))
        if IfNoneMatch == '*' and old or IfMatch is not None and (not old or old[0] != IfMatch):
            raise Conflict()
        self.version += 1; etag = str(self.version); self.rows[(Bucket, Key)] = (etag, Body, Metadata or {})
        return {'ETag': etag}
    def get_object(self, Bucket, Key):
        tag, body, metadata = self.rows[(Bucket, Key)]
        return {'ETag': tag, 'Body': io.BytesIO(body), 'Metadata': metadata}
    def head_object(self, Bucket, Key):
        tag, _, metadata = self.rows[(Bucket, Key)]
        return {'ETag': tag, 'Metadata': metadata}


def test_two_ends_cannot_claim_same_example_and_done_is_durable():
    s = FakeS3(); a = SharedQueue(s, 'b', 'p', 'vv', 'sha'); b = SharedQueue(s, 'b', 'p', 'fpmi', 'sha')
    tag = a.reserve('math', 'id', 'prompt')
    assert tag and b.reserve('math', 'id', 'prompt') is None
    assert a.reserve('math', 'id', 'prompt') == tag
    a.finish('math', 'id', 'prompt', tag, {'answer': 5})
    assert b.reserve('math', 'id', 'prompt') is None
    assert json.loads(s.rows[('b', a.key('math', 'id'))][1])['result']['answer'] == 5


def test_mismatching_prompt_is_never_silently_reused():
    s = FakeS3(); a = SharedQueue(s, 'b', 'p', 'a', 'sha'); b = SharedQueue(s, 'b', 'p', 'b', 'sha')
    a.reserve('gpqa', 'id', 'one')
    with pytest.raises(ValueError, match='identity mismatch'):
        b.reserve('gpqa', 'id', 'different')
