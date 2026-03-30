from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


COMMUNITY_MAGIC_STRING = (
    "anthropic_magic_string_trigger_refusal_"
    "1faefb6177b4672dee07f9d3afc62588ccd2631edcf22e8ccc1fb35b501c9c86"
)

PROXY_HEADER_MARKERS = {
    "x-oneapi-request-id": "oneapi request id header",
    "x-newapi-request-id": "new-api request id header",
    "x-request-id": "generic request id header",
}

VERTEX_MODEL_RE = re.compile(r"^claude-[a-z0-9][a-z0-9\-]*@\d{8}$")
ANTHROPIC_MODEL_RE = re.compile(r"^claude-[a-z0-9][a-z0-9\-]*-\d{8}$")
BEDROCK_MODEL_RE = re.compile(r"^anthropic\.claude-[a-z0-9.\-]+-v1(?::\d+)?$")


class RootHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.generator = ""
        self.title = ""
        self.descriptions: list[str] = []
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            name = attr_map.get("name", "").lower()
            if name == "generator":
                self.generator = attr_map.get("content", "")
            if name == "description" and attr_map.get("content"):
                self.descriptions.append(attr_map["content"])
        if tag.lower() == "title":
            self._capture_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._capture_title = False

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self.title += data


@dataclass
class HTTPResult:
    url: str
    method: str
    status: int | None
    headers: dict[str, str]
    text: str
    json_body: Any | None
    error: str | None = None


@dataclass
class ProbeFinding:
    signal: str
    detail: str
    weight: int = 1


@dataclass
class ProbeReport:
    base_url: str
    resolved_paths: dict[str, str]
    requested_model: str
    actual_model: str | None
    classification: str
    confidence: str
    summary: str
    findings: dict[str, list[ProbeFinding]] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["findings"] = {
            key: [asdict(item) for item in items]
            for key, items in self.findings.items()
        }
        return raw


def mask_secret(value: str) -> str:
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _normalize_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("base_url is required")
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value


def _api_root(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return base_url.rstrip("/")
    if not path:
        return base_url.rstrip("/") + "/v1"
    return base_url.rstrip("/")


def _path_map(base_url: str) -> dict[str, str]:
    root = _normalize_base_url(base_url)
    api_root = _api_root(root)
    return {
        "root": root,
        "models": api_root + "/models",
        "messages": api_root + "/messages",
        "chat_completions": api_root + "/chat/completions",
    }


def _json_loads(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> HTTPResult:
    body: bytes | None = None
    request_headers = {
        "User-Agent": "curl/8.5.0",
        "Accept": "*/*",
        **dict(headers or {}),
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            text = raw_body.decode("utf-8", errors="replace")
            return HTTPResult(
                url=url,
                method=method,
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                text=text,
                json_body=_json_loads(text),
            )
    except HTTPError as exc:
        raw_body = exc.read()
        text = raw_body.decode("utf-8", errors="replace")
        return HTTPResult(
            url=url,
            method=method,
            status=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            text=text,
            json_body=_json_loads(text),
            error=str(exc),
        )
    except URLError as exc:
        return HTTPResult(
            url=url,
            method=method,
            status=None,
            headers={},
            text="",
            json_body=None,
            error=str(exc),
        )
    except TimeoutError as exc:
        return HTTPResult(
            url=url,
            method=method,
            status=None,
            headers={},
            text="",
            json_body=None,
            error=str(exc),
        )
    except Exception as exc:
        return HTTPResult(
            url=url,
            method=method,
            status=None,
            headers={},
            text="",
            json_body=None,
            error=str(exc),
        )


def _extract_models(models_result: HTTPResult) -> list[dict[str, Any]]:
    payload = models_result.json_body
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("data") or payload.get("models") or []
    return [item for item in raw_items if isinstance(item, dict)]


def _pick_candidate_model(model: str | None, models: list[dict[str, Any]]) -> str:
    if model:
        return model
    model_ids = [str(item.get("id")) for item in models if item.get("id")]
    preferred_prefixes = [
        "claude-opus",
        "claude-sonnet",
        "claude-haiku",
    ]
    for prefix in preferred_prefixes:
        for model_id in model_ids:
            if model_id.startswith(prefix):
                return model_id
    if model_ids:
        return model_ids[0]
    return "claude-opus-4-1-20250805"


def _root_hints(root_result: HTTPResult) -> dict[str, Any]:
    hints: dict[str, Any] = {"generator": "", "title": "", "descriptions": []}
    if not root_result.text:
        return hints
    parser = RootHTMLParser()
    parser.feed(root_result.text)
    hints["generator"] = parser.generator.strip()
    hints["title"] = parser.title.strip()
    hints["descriptions"] = [text.strip() for text in parser.descriptions if text.strip()]
    return hints


def _response_content_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    if isinstance(payload.get("choices"), list):
        choices = payload["choices"]
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
    return ""


def _response_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if model:
        return str(model)
    return None


def _response_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("id"):
        return str(payload["id"])
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _build_findings() -> dict[str, list[ProbeFinding]]:
    return {"proxy": [], "vertex": [], "anthropic": [], "behavior": []}


def _add_finding(findings: dict[str, list[ProbeFinding]], category: str, signal: str, detail: str, weight: int = 1) -> None:
    findings.setdefault(category, []).append(ProbeFinding(signal=signal, detail=detail, weight=weight))


def _classify(
    findings: dict[str, list[ProbeFinding]],
    *,
    requested_model: str,
    actual_model: str | None,
    response_id: str | None,
) -> tuple[str, str, str]:
    proxy_score = sum(item.weight for item in findings.get("proxy", []))
    vertex_score = sum(item.weight for item in findings.get("vertex", []))
    anthropic_score = sum(item.weight for item in findings.get("anthropic", []))
    behavior_score = sum(item.weight for item in findings.get("behavior", []))

    if proxy_score >= 3:
        if vertex_score >= 2:
            confidence = "high" if response_id and response_id.startswith("msg_vrtx_") else "medium"
            summary = (
                "This endpoint behaves like a third-party compatibility gateway and also exposes "
                "multiple Vertex AI signals."
            )
            return "proxy_to_vertex_ai_possible", confidence, summary
        summary = (
            "This endpoint behaves like a third-party compatibility gateway rather than a direct "
            "Anthropic or Vertex AI endpoint."
        )
        return "third_party_proxy_likely", "high", summary

    if vertex_score >= 3 and proxy_score == 0:
        summary = "The endpoint looks closer to a direct Vertex AI Anthropic surface than to direct Anthropic."
        return "vertex_ai_anthropic_likely", "medium", summary

    if anthropic_score >= 2 and proxy_score == 0 and vertex_score == 0:
        summary = "The endpoint matches direct Anthropic naming and response conventions more closely."
        return "direct_anthropic_likely", "medium", summary

    if behavior_score >= 2 and actual_model and actual_model != requested_model:
        summary = "The endpoint rewrites or downgrades models, which points to an intermediate routing layer."
        return "routed_backend_unknown", "medium", summary

    return "unknown", "low", "The collected signals are mixed and do not support a confident attribution."


def probe_origin(
    base_url: str,
    api_key: str,
    *,
    model: str | None = None,
    anthropic_version: str = "2023-06-01",
    timeout: int = 30,
) -> ProbeReport:
    paths = _path_map(base_url)
    findings = _build_findings()

    root_result = http_request(paths["root"], timeout=timeout)
    root_hints = _root_hints(root_result)
    generator = _normalize_text(root_hints.get("generator", ""))
    title = _normalize_text(root_hints.get("title", ""))
    descriptions = [_normalize_text(item) for item in root_hints.get("descriptions", [])]

    if generator:
        if "new-api" in generator:
            _add_finding(findings, "proxy", "html_generator", f"root page generator={generator}", 3)
    if "new api" in title or any("gateway" in item or "compatible" in item for item in descriptions):
        _add_finding(findings, "proxy", "marketing_copy", "root page advertises compatibility gateway behavior", 2)

    for header_name, detail in PROXY_HEADER_MARKERS.items():
        if header_name in root_result.headers:
            _add_finding(findings, "proxy", "root_header", detail, 2)

    models_result = http_request(
        paths["models"],
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    models = _extract_models(models_result)
    candidate_model = _pick_candidate_model(model, models)

    for item in models:
        model_id = str(item.get("id", ""))
        owner = str(item.get("owned_by", "") or item.get("owner", ""))
        if owner.lower() == "vertex-ai":
            _add_finding(findings, "vertex", "model_owner", f"{model_id} reports owned_by=vertex-ai", 2)
        if VERTEX_MODEL_RE.match(model_id):
            _add_finding(findings, "vertex", "vertex_model_name", f"{model_id} matches Vertex naming", 2)
        elif ANTHROPIC_MODEL_RE.match(model_id):
            _add_finding(findings, "anthropic", "anthropic_model_name", f"{model_id} matches Anthropic naming", 1)
        elif BEDROCK_MODEL_RE.match(model_id):
            _add_finding(findings, "behavior", "bedrock_model_name", f"{model_id} matches Bedrock naming", 1)
        elif model_id.startswith("claude-") and "@" not in model_id and not ANTHROPIC_MODEL_RE.match(model_id):
            _add_finding(findings, "proxy", "alias_model_name", f"{model_id} looks like a custom alias", 1)
        endpoint_types = item.get("supported_endpoint_types")
        if isinstance(endpoint_types, list) and len(endpoint_types) > 1:
            _add_finding(findings, "proxy", "endpoint_emulation", f"{model_id} advertises multiple endpoint types", 2)

    request_headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }

    basic_payload = {
        "model": candidate_model,
        "max_tokens": 16,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly TEST."}],
    }
    basic_result = http_request(
        paths["messages"],
        method="POST",
        headers=request_headers,
        payload=basic_payload,
        timeout=timeout,
    )
    basic_model = _response_model(basic_result.json_body)
    basic_id = _response_id(basic_result.json_body)
    if isinstance(basic_result.json_body, dict):
        error = basic_result.json_body.get("error")
        if isinstance(error, dict):
            error_type = str(error.get("type", ""))
            if error_type == "new_api_error":
                _add_finding(findings, "proxy", "error_type", "basic message returned new_api_error", 2)

    if basic_model and basic_model != candidate_model:
        _add_finding(
            findings,
            "proxy",
            "model_rewrite",
            f"requested {candidate_model} but response reported {basic_model}",
            3,
        )
    if basic_model and VERTEX_MODEL_RE.match(basic_model):
        _add_finding(findings, "vertex", "response_model_name", f"response model {basic_model} matches Vertex naming", 3)
    elif basic_model and ANTHROPIC_MODEL_RE.match(basic_model):
        _add_finding(findings, "anthropic", "response_model_name", f"response model {basic_model} matches Anthropic naming", 1)
    elif basic_model and basic_model.startswith("claude-") and not ANTHROPIC_MODEL_RE.match(basic_model):
        _add_finding(findings, "proxy", "response_alias_model", f"response model {basic_model} looks like a custom alias", 1)

    if basic_id:
        if basic_id.startswith("msg_vrtx_"):
            _add_finding(findings, "vertex", "response_id", f"response id {basic_id} matches Vertex prefix", 3)
        elif basic_id.startswith("msg_"):
            _add_finding(findings, "anthropic", "response_id", f"response id {basic_id} matches Anthropic-style prefix", 1)

    for header_name, detail in PROXY_HEADER_MARKERS.items():
        if header_name in basic_result.headers:
            _add_finding(findings, "proxy", "response_header", detail, 2)

    magic_payload = {
        "model": candidate_model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{"role": "user", "content": COMMUNITY_MAGIC_STRING}],
    }
    magic_result = http_request(
        paths["messages"],
        method="POST",
        headers=request_headers,
        payload=magic_payload,
        timeout=timeout,
    )
    magic_text = _response_content_text(magic_result.json_body)
    if isinstance(magic_result.json_body, dict):
        error = magic_result.json_body.get("error")
        if isinstance(error, dict) and str(error.get("type", "")) == "new_api_error":
            _add_finding(findings, "proxy", "error_type", "magic-string probe returned new_api_error", 2)
    if magic_result.status == 200 and magic_text:
        _add_finding(
            findings,
            "behavior",
            "magic_string_not_blocked",
            "community magic-string probe returned normal content instead of a refusal/error",
            2,
        )

    max_payload = {
        "model": candidate_model,
        "max_tokens": 1,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly this sequence and nothing else: 1 2 3 4 5 6 7 8 9 10",
            }
        ],
    }
    max_result = http_request(
        paths["messages"],
        method="POST",
        headers=request_headers,
        payload=max_payload,
        timeout=timeout,
    )
    max_text = _response_content_text(max_result.json_body)
    if isinstance(max_result.json_body, dict):
        error = max_result.json_body.get("error")
        if isinstance(error, dict) and str(error.get("type", "")) == "new_api_error":
            _add_finding(findings, "proxy", "error_type", "max_tokens probe returned new_api_error", 2)
    output_tokens = None
    if isinstance(max_result.json_body, dict):
        usage = max_result.json_body.get("usage")
        if isinstance(usage, dict):
            value = usage.get("output_tokens")
            if isinstance(value, int):
                output_tokens = value
    if output_tokens and output_tokens > 1:
        _add_finding(
            findings,
            "behavior",
            "max_tokens_violation",
            f"max_tokens=1 returned usage.output_tokens={output_tokens}",
            2,
        )
    elif max_text.count(" ") >= 2:
        _add_finding(
            findings,
            "behavior",
            "max_tokens_violation",
            "max_tokens=1 still returned a long multi-token looking string",
            2,
        )

    classification, confidence, summary = _classify(
        findings,
        requested_model=candidate_model,
        actual_model=basic_model,
        response_id=basic_id,
    )

    probes = {
        "root": {
            "status": root_result.status,
            "headers": root_result.headers,
            "html_hints": root_hints,
        },
        "models": {
            "status": models_result.status,
            "headers": models_result.headers,
            "body": models_result.json_body,
        },
        "basic_message": {
            "status": basic_result.status,
            "headers": basic_result.headers,
            "body": basic_result.json_body,
        },
        "magic_string": {
            "status": magic_result.status,
            "headers": magic_result.headers,
            "body": magic_result.json_body,
        },
        "max_tokens_boundary": {
            "status": max_result.status,
            "headers": max_result.headers,
            "body": max_result.json_body,
        },
    }

    return ProbeReport(
        base_url=_normalize_base_url(base_url),
        resolved_paths=paths,
        requested_model=candidate_model,
        actual_model=basic_model,
        classification=classification,
        confidence=confidence,
        summary=summary,
        findings=findings,
        probes=probes,
    )
