from __future__ import annotations

import unittest

from claude_openai_bridge.origin_probe import (
    ProbeFinding,
    _classify,
    _path_map,
)


class PathMapTests(unittest.TestCase):
    def test_root_base_url_gets_v1_paths(self) -> None:
        paths = _path_map("https://example.com")
        self.assertEqual(paths["models"], "https://example.com/v1/models")
        self.assertEqual(paths["messages"], "https://example.com/v1/messages")

    def test_existing_v1_base_url_is_preserved(self) -> None:
        paths = _path_map("https://example.com/v1/")
        self.assertEqual(paths["models"], "https://example.com/v1/models")
        self.assertEqual(paths["chat_completions"], "https://example.com/v1/chat/completions")


class ClassificationTests(unittest.TestCase):
    def test_proxy_to_vertex_is_separated_from_direct_vertex(self) -> None:
        findings = {
            "proxy": [ProbeFinding("response_header", "x-oneapi-request-id", 2), ProbeFinding("model_rewrite", "rewrote model", 3)],
            "vertex": [ProbeFinding("model_owner", "owned_by=vertex-ai", 2)],
            "anthropic": [],
            "behavior": [],
        }
        classification, confidence, _summary = _classify(
            findings,
            requested_model="claude-opus-4-6",
            actual_model="claude-sonnet-4-6",
            response_id="msg_123",
        )
        self.assertEqual(classification, "proxy_to_vertex_ai_possible")
        self.assertEqual(confidence, "medium")

    def test_direct_anthropic_requires_clean_signals(self) -> None:
        findings = {
            "proxy": [],
            "vertex": [],
            "anthropic": [
                ProbeFinding("response_model_name", "claude-opus-4-1-20250805", 1),
                ProbeFinding("response_id", "msg_x", 1),
            ],
            "behavior": [],
        }
        classification, confidence, _summary = _classify(
            findings,
            requested_model="claude-opus-4-1-20250805",
            actual_model="claude-opus-4-1-20250805",
            response_id="msg_abc",
        )
        self.assertEqual(classification, "direct_anthropic_likely")
        self.assertEqual(confidence, "medium")

    def test_plain_proxy_without_vertex_signals_stays_proxy(self) -> None:
        findings = {
            "proxy": [
                ProbeFinding("html_generator", "generator=new-api", 3),
                ProbeFinding("endpoint_emulation", "multiple endpoint types", 2),
            ],
            "vertex": [],
            "anthropic": [],
            "behavior": [],
        }
        classification, confidence, _summary = _classify(
            findings,
            requested_model="claude-opus-4-6",
            actual_model="claude-opus-4-6",
            response_id=None,
        )
        self.assertEqual(classification, "third_party_proxy_likely")
        self.assertEqual(confidence, "high")


if __name__ == "__main__":
    unittest.main()
