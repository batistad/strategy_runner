from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


def _hmac_signature(secret: str, payload: bytes) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def post_json(
    url: str, obj: Any, secret_env: str | None, timeout: int = 10, retries: int = 3
) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}

    if secret_env:
        secret = os.getenv(secret_env)
        if secret:
            headers["X-Signature"] = _hmac_signature(secret, payload)

    for i in range(retries):
        try:
            r = requests.post(url, data=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            log.info("Webhook delivered: %s", r.status_code)
            return
        except Exception as e:
            log.warning("Webhook attempt %d failed: %s", i + 1, e)
            time.sleep(2**i)
    raise RuntimeError("Webhook failed after retries")
