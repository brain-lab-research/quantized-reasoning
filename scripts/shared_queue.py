"""Durable per-example conditional claims for a shared S3 inference queue."""
from __future__ import annotations

import hashlib
import json
from typing import Any


class SharedQueue:
    def __init__(self, client: Any, bucket: str, prefix: str, owner: str, fingerprint: str):
        if not owner or not fingerprint:
            raise ValueError('Queue owner and fingerprint are required')
        self.client, self.bucket, self.prefix = client, bucket, prefix.rstrip('/')
        self.owner, self.fingerprint = owner, fingerprint

    def key(self, benchmark: str, example_id: str) -> str:
        digest = hashlib.sha256((benchmark + '\0' + example_id).encode()).hexdigest()
        return f'{self.prefix}/items/{digest}.json'

    @staticmethod
    def _body(record):
        return json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode()

    def reserve(self, benchmark: str, example_id: str, prompt: str):
        key = self.key(benchmark, example_id)
        identity = {'benchmark': benchmark, 'example_id': example_id,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'fingerprint': self.fingerprint}
        record = {**identity, 'owner': self.owner, 'state': 'claimed'}
        metadata = {'state': 'claimed', 'owner': self.owner, 'fingerprint': self.fingerprint,
                    'prompt-sha256': identity['prompt_sha256']}
        try:
            result = self.client.put_object(Bucket=self.bucket, Key=key, Body=self._body(record),
                                            Metadata=metadata, IfNoneMatch='*')
            return result['ETag']
        except Exception as exc:
            error = getattr(exc, 'response', {}).get('Error', {}).get('Code')
            if error not in ('PreconditionFailed', '412', 'ConditionalRequestConflict', '409'):
                raise
        current = self.client.head_object(Bucket=self.bucket, Key=key)
        meta = current.get('Metadata', {})
        if meta.get('fingerprint') and meta.get('prompt-sha256'):
            saved = {**identity, 'fingerprint': meta['fingerprint'], 'prompt_sha256': meta['prompt-sha256'],
                     'owner': meta.get('owner'), 'state': meta.get('state')}
        else:
            # Older pre-seeded items do not have identity metadata.
            current = self.client.get_object(Bucket=self.bucket, Key=key)
            saved = json.loads(current['Body'].read())
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Queue identity mismatch for {benchmark}/{example_id}')
        if saved.get('state') == 'claimed' and saved.get('owner') == self.owner:
            return current['ETag']
        return None

    def finish(self, benchmark: str, example_id: str, prompt: str, etag: str, result: dict):
        record = {'benchmark': benchmark, 'example_id': example_id,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'fingerprint': self.fingerprint, 'owner': self.owner,
                  'state': 'done', 'result': result}
        self.client.put_object(Bucket=self.bucket, Key=self.key(benchmark, example_id),
                               Body=self._body(record), Metadata={'state': 'done', 'owner': self.owner,
                                                                  'fingerprint': self.fingerprint,
                                                                  'prompt-sha256': record['prompt_sha256']}, IfMatch=etag)
