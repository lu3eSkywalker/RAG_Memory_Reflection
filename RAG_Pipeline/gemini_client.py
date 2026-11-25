"""Shared Gemini HTTP client.

Networking note: urllib3/requests iterate over every address returned by
getaddrinfo, so whichever address family the machine cannot route burns the full
connect timeout before the next one is tried - a "10s" request could appear to
hang. Different hosts route differently (one reachable over IPv6 only, another
over IPv4 only), so this client never hard-codes a family: it tries the
resolver's order first, then retries over the other family if the connection
fails.

The API key is read from the GEMINI_API_KEY environment variable only - it is
never stored in source.
"""

import os
import socket
from typing import List, Optional, Tuple

DEFAULT_MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 90


class AddressFamily:
    """Context manager restricting getaddrinfo to one address family.

    family="ipv4" -> AF_INET only, "ipv6" -> AF_INET6 only, None -> resolver order.
    """

    _FAMILY = {"ipv4": socket.AF_INET, "ipv6": socket.AF_INET6}

    def __init__(self, family: Optional[str] = None) -> None:
        self.family = family
        self._original = socket.getaddrinfo

    def __enter__(self) -> "AddressFamily":
        if self.family is None:
            return self
        wanted = self._FAMILY[self.family]
        original = self._original

        def restricted(host, port, *args, **kwargs):
            try:
                infos = original(host, port, *args, **kwargs)
            except socket.gaierror:
                return original(host, port, wanted, socket.SOCK_STREAM)
            return [info for info in infos if info[0] == wanted] or original(
                host, port, wanted, socket.SOCK_STREAM
            )

        socket.getaddrinfo = restricted
        return self

    def __exit__(self, *exc_info) -> None:
        socket.getaddrinfo = self._original


# Backwards-compatible alias
IPv4First = AddressFamily


def api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


def available() -> bool:
    return bool(api_key())


def _post(url: str, key: str, payload: dict, timeout: Tuple[float, float]):
    """POST trying the resolver's order, then the other address family.

    Which family a host can actually route differs per machine (and can even
    change between networks), so a connection failure triggers a retry with the
    family pinned rather than a hard-coded preference.
    """
    import requests

    last_error: Optional[Exception] = None
    for family in (None, "ipv4", "ipv6"):
        try:
            with AddressFamily(family):
                return requests.post(url, params={"key": key}, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as error:
            last_error = error
            continue
    raise last_error  # type: ignore[misc]


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    timeout: Tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT),
    json_mode: bool = False,
) -> str:
    """Call Gemini and return the response text.

    Set ``json_mode=True`` to request a JSON object response. Falls back to a
    plain request if the deployed model rejects the MIME type.

    Raises RuntimeError when the key or `requests` is missing, or when the API
    call fails - callers decide how to handle it.
    """
    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")
    if not prompt.strip():
        raise RuntimeError("prompt is empty")

    try:
        import requests
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("requests is not installed") from error

    url = ENDPOINT.format(model=model)
    config: dict = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": max_tokens,
    }
    if json_mode:
        config["response_mime_type"] = "application/json"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": config,
    }

    response = _post(url, key, payload, timeout)

    if json_mode and response.status_code != 200:
        # Model may not support response_mime_type — retry without it.
        payload["generationConfig"].pop("response_mime_type", None)
        response = _post(url, key, payload, timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text[:300]}"
        )

    candidates: List[dict] = response.json().get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return parts[0].get("text", "") if parts else ""