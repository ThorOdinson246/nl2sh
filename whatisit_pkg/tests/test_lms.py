"""LM Studio CLI backend: discovery, load/chat, and engine selection."""
from __future__ import annotations

import json

import pytest

from whatisit import config as cfg_mod
from whatisit import engine, lms_backend


class TestResolveModelKey:
    def test_pin_wins(self, tmp_path):
        cfg = {"lms_path": str(tmp_path / "lms"), "lms_model": "nl2sh-3b"}
        assert lms_backend.resolve_model_key(cfg) == "nl2sh-3b"

    def test_unique_nl2sh_prefix(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms.exe"
        lms.write_text("")
        cfg = {"lms_path": str(lms)}

        def fake_list(p):
            return [
                {"type": "llm", "modelKey": "other-model"},
                {"type": "llm", "modelKey": "nl2sh-1.5b"},
            ]

        monkeypatch.setattr(lms_backend, "list_models", fake_list)
        assert lms_backend.resolve_model_key(cfg, lms) == "nl2sh-1.5b"

    def test_multiple_nl2sh_requires_pin(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms)}
        monkeypatch.setattr(
            lms_backend, "list_models",
            lambda p: [{"type": "llm", "modelKey": "nl2sh-1.5b"},
                       {"type": "llm", "modelKey": "nl2sh-3b"}],
        )
        with pytest.raises(lms_backend.LmsError, match="multiple nl2sh"):
            lms_backend.resolve_model_key(cfg, lms)

    def test_none_is_unavailable(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms)}
        monkeypatch.setattr(lms_backend, "list_models", lambda p: [])
        with pytest.raises(lms_backend.LmsUnavailable, match="no nl2sh"):
            lms_backend.resolve_model_key(cfg, lms)


class TestEnsureLoaded:
    def test_skips_load_when_already_in_ps(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms), "lms_model": "nl2sh-1.5b"}
        monkeypatch.setattr(
            lms_backend, "list_models",
            lambda p: [{"type": "llm", "modelKey": "nl2sh-1.5b"}],
        )
        monkeypatch.setattr(lms_backend, "is_loaded", lambda p, k: True)
        calls = []

        def boom(*a, **k):
            calls.append(a)
            raise AssertionError("load should not run")

        monkeypatch.setattr(lms_backend, "_run", boom)
        assert lms_backend.ensure_loaded(cfg) == "nl2sh-1.5b"
        assert calls == []

    def test_load_passes_ttl_when_configured(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms), "lms_model": "nl2sh-1.5b", "lms_ttl": 600}
        monkeypatch.setattr(
            lms_backend, "list_models",
            lambda p: [{"type": "llm", "modelKey": "nl2sh-1.5b"}],
        )
        monkeypatch.setattr(lms_backend, "is_loaded", lambda p, k: False)
        seen = []

        def fake_run(path, args, timeout=120.0):
            seen.append(args)
            class R:
                returncode = 0
                stdout = ""
                stderr = ""
            return R()

        monkeypatch.setattr(lms_backend, "_run", fake_run)
        lms_backend.ensure_loaded(cfg)
        assert seen == [["load", "nl2sh-1.5b", "-y", "--ttl", "600"]]

    def test_load_omits_ttl_when_unset(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms), "lms_model": "nl2sh-1.5b"}
        monkeypatch.setattr(
            lms_backend, "list_models",
            lambda p: [{"type": "llm", "modelKey": "nl2sh-1.5b"}],
        )
        monkeypatch.setattr(lms_backend, "is_loaded", lambda p, k: False)
        seen = []

        def fake_run(path, args, timeout=120.0):
            seen.append(args)
            class R:
                returncode = 0
                stdout = ""
                stderr = ""
            return R()

        monkeypatch.setattr(lms_backend, "_run", fake_run)
        lms_backend.ensure_loaded(cfg)
        assert seen == [["load", "nl2sh-1.5b", "-y"]]


class TestChat:
    def test_returns_stdout(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        cfg = {"lms_path": str(lms), "lms_model": "nl2sh-1.5b"}
        monkeypatch.setattr(lms_backend, "ensure_loaded", lambda c, key=None: "nl2sh-1.5b")

        def fake_run(path, args, timeout=120.0):
            assert args[0] == "chat"
            assert "--dont-fetch-catalog" in args
            assert "-p" in args
            class R:
                returncode = 0
                stdout = "ps aux\n"
                stderr = ""
            return R()

        monkeypatch.setattr(lms_backend, "_run", fake_run)
        assert lms_backend.chat(cfg, "sys", "list processes") == "ps aux"


class TestEngineSelection:
    def _host(self, monkeypatch):
        class H:
            @staticmethod
            def build(prompt, enabled=True, cwd=None):
                return "SYS", prompt
        monkeypatch.setattr(engine, "hostctx", H)

    def test_legacy_uses_server_not_lms(self, monkeypatch, tmp_path):
        self._host(monkeypatch)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"x")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))
        monkeypatch.setattr(engine, "start_server", lambda *a, **k: 9)
        monkeypatch.setattr(
            engine, "_query_server",
            lambda *a, **k: [("ls -la", "stop")],
        )
        lms_calls = []

        def no_lms(*a, **k):
            lms_calls.append(1)
            raise AssertionError("lms must not run in legacy mode")

        monkeypatch.setattr(engine, "_query_lms", no_lms)
        cmds, _, mode = engine.generate("list files", {})
        assert cmds == ["ls -la"] and mode == "server" and lms_calls == []

    def test_primary_lms(self, monkeypatch, tmp_path):
        self._host(monkeypatch)
        monkeypatch.setattr(
            engine, "_query_lms",
            lambda cfg, user, n, system: [("ps aux", "stop")],
        )
        # llama missing should not matter when lms is configured
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        cfg = {"backend_primary": "lms", "lms_path": str(tmp_path / "lms")}
        cmds, _, mode = engine.generate("list processes", cfg)
        assert cmds == ["ps aux"] and mode == "lms"

    def test_lms_unavailable_does_not_silently_use_llama(self, monkeypatch, tmp_path):
        self._host(monkeypatch)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)

        def fail_lms(cfg, user, n, system):
            raise lms_backend.LmsUnavailable("no lms")

        monkeypatch.setattr(engine, "_query_lms", fail_lms)
        cfg = {"backend_primary": "lms", "lms_path": str(tmp_path / "missing")}
        with pytest.raises(lms_backend.LmsUnavailable, match="no lms"):
            engine.generate("noop", cfg)


class TestJsonHelpers:
    def test_list_models_parses_array(self, monkeypatch, tmp_path):
        lms = tmp_path / "lms"
        lms.write_text("")
        payload = [{"type": "llm", "modelKey": "nl2sh-1.5b"}]

        def fake_run(path, args, timeout=120.0):
            class R:
                returncode = 0
                stdout = json.dumps(payload)
                stderr = ""
            return R()

        monkeypatch.setattr(lms_backend, "_run", fake_run)
        assert lms_backend.list_models(lms) == payload


class TestSubprocessEncoding:
    def test_run_forces_utf8_not_locale_codepage(self, monkeypatch, tmp_path):
        """Windows defaults text=True to cp1252; lms emits UTF-8 spinners/glyphs."""
        lms = tmp_path / "lms"
        lms.write_text("")
        seen = {}

        def fake_run(*a, **kw):
            seen.update(kw)
            class R:
                returncode = 0
                stdout = "ps aux\n"
                stderr = ""
            return R()

        monkeypatch.setattr(lms_backend.subprocess, "run", fake_run)
        r = lms_backend._run(lms, ["chat", "m", "-p", "x"])
        assert r.stdout == "ps aux\n"
        assert seen.get("encoding") == "utf-8"
        assert seen.get("errors") == "replace"
        assert seen.get("text") is True
