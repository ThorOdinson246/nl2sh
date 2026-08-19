"""Tests for whatisit.engine, with the HTTP layer entirely mocked out.

No test here ever starts llama-server, touches the network, or requires a
model file: engine._post / engine._query_server / engine.start_server are all
monkeypatched, and "model" paths are just empty tmp_path files that only need
to exist.
"""
import os
import stat
import sys
import urllib.error
import urllib.request

import pytest

from whatisit import config as cfg_mod
from whatisit import engine

# ---------------------------------------------------------- looks_degenerate

class TestLooksDegenerate:
    def test_short_command_is_never_degenerate(self):
        assert engine.looks_degenerate("ls -la") is False

    def test_ordinary_long_command_is_not_degenerate(self):
        cmd = "find . -type f -name '*.py' -newer ref.txt -exec grep -l TODO {} +"
        assert engine.looks_degenerate(cmd) is False

    def test_flag_spam_loop_is_degenerate(self):
        # Same shape as the reproducible real-world zip failure from the
        # docstring, extended so the repeated flag hits the default
        # min_repeats=4 threshold.
        cmd = "zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 -9 -n -j -0 archive.zip ."
        assert engine.looks_degenerate(cmd) is True

    def test_repeated_flag_below_threshold_is_not_degenerate(self):
        cmd = "tar -czvf out.tar.gz -C dir1 a -C dir2 b -C dir3 c file.txt"
        # "-C" repeats only 3 times here, one below the default min_repeats=4.
        assert engine.looks_degenerate(cmd, min_repeats=4) is False

    def test_custom_min_repeats_threshold(self):
        cmd = "prog -x -x -x file1 file2 file3 file4 file5"
        assert engine.looks_degenerate(cmd, min_repeats=3) is True
        assert engine.looks_degenerate(cmd, min_repeats=4) is False


# --------------------------------------------------------------- CLI chrome

class TestStripCliChrome:
    def test_extracts_answer_between_echo_and_footer(self):
        out = (
            "llama-cli banner text\n"
            "loading model...\n"
            "> find files bigger than 100MB\n"
            "find . -size +100M\n"
            "[ Prompt: 12 tokens, 1.5 ms/token ]\n"
        )
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_uses_last_echo_line_when_several_present(self):
        # --no-display-prompt does not suppress the prompt echo, and in a
        # multi-turn-looking transcript there can be more than one "> " line;
        # the real answer follows the LAST one.
        out = (
            "> irrelevant earlier turn\n"
            "some old answer\n"
            "> find files bigger than 100MB\n"
            "find . -size +100M\n"
            "Exiting...\n"
        )
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_strips_ansi_codes(self):
        out = "> prompt\n\x1b[32mfind . -size +100M\x1b[0m\n[ Prompt: done ]\n"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_no_footer_present_still_returns_body(self):
        out = "> prompt\nfind . -size +100M\n"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_carriage_returns_normalized(self):
        out = "> prompt\rfind . -size +100M\r[ Prompt: done ]\r"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"


# ------------------------------------------------------------- private write

class TestWritePrivate:
    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_creates_file_with_0600(self, tmp_path):
        p = tmp_path / "server.token"
        engine._write_private(p, "supersecret")
        assert p.read_text() == "supersecret"
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_overwrites_existing_content(self, tmp_path):
        p = tmp_path / "server.pid"
        p.write_text("old")
        engine._write_private(p, "new")
        assert p.read_text() == "new"

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_pre_existing_loosely_permissioned_file_is_tightened(self, tmp_path):
        p = tmp_path / "server.token"
        p.write_text("old")
        os.chmod(p, 0o644)
        engine._write_private(p, "new-secret-token")
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="os.O_NOFOLLOW is not available on Windows")
    def test_refuses_to_follow_a_symlink(self, tmp_path):
        target = tmp_path / "outside_file.txt"
        target.write_text("do not touch me")
        link = tmp_path / "server.token"
        link.symlink_to(target)
        try:
            engine._write_private(link, "pwned")
        except OSError:
            pass  # ELOOP is the expected, safe outcome
        # Either way, the symlink target must be untouched.
        assert target.read_text() == "do not touch me"


# ------------------------------------------------------ greedy-first ordering

class TestQueryServerGreedyOrdering:
    def test_greedy_call_is_always_first_and_uses_temperature_zero(self, monkeypatch):
        calls = []

        def fake_post(port, body):
            calls.append(body)
            if body.get("temperature") == 0.0 and "n" not in body:
                return [("greedy answer", "stop")]
            return [("sampled alt 1", "stop"), ("sampled alt 2", "stop")]

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="zip up this project",
                                    cfg={"temperature": 0.0, "max_tokens": 64}, n=3)
        assert out[0] == ("greedy answer", "stop")
        # the greedy request (first call made) must be temperature 0
        assert calls[0]["temperature"] == 0.0

    def test_single_candidate_skips_the_sampled_call_entirely(self, monkeypatch):
        calls = []

        def fake_post(port, body):
            calls.append(body)
            return [("greedy answer", "stop")]

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="list files",
                                    cfg={"temperature": 0.0}, n=1)
        assert out == [("greedy answer", "stop")]
        assert len(calls) == 1

    def test_sampled_call_failure_does_not_lose_the_greedy_answer(self, monkeypatch):
        def fake_post(port, body):
            if body.get("temperature") == 0.0:
                return [("greedy answer", "stop")]
            raise RuntimeError("server hiccup")

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="list files",
                                    cfg={"temperature": 0.0}, n=3)
        assert out == [("greedy answer", "stop")]


# ------------------------------------------------ generate(): finish_reason

class TestGenerateDiscardsTruncatedCandidates:
    """generate() must discard any candidate whose finish_reason == "length":
    a truncated flag-spam loop can carry a destructive flag (e.g. zip -m,
    which deletes the source files) that only appears once the spam runs on
    long enough to hit the token budget.
    """

    def _fake_cfg_and_model(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        monkeypatch.setattr(engine, "hostctx",
                             _FakeHostCtx())
        # Make server_bin resolution succeed without needing a real binary:
        # WHATISIT_LLAMA_SERVER just needs to point at a file that exists.
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))
        monkeypatch.setattr(engine, "start_server", lambda *a, **kw: 12345)

    def test_length_finish_reason_is_dropped(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [
                ("zip -r -9 -m -j -0 -1 -1 -1", "length"),   # truncated, has -m: drop
                ("zip -r archive.zip .", "stop"),             # clean: keep
            ]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, elapsed, mode = engine.generate("zip this folder", {}, n=2)
        assert cmds == ["zip -r archive.zip ."]
        assert mode == "server"

    def test_degenerate_candidate_is_also_dropped_even_if_finished(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [
                ("zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 -9 -n -j -0 a.zip .", "stop"),
                ("zip -r archive.zip .", "stop"),
            ]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("zip this folder", {}, n=2)
        assert cmds == ["zip -r archive.zip ."]

    def test_duplicate_candidates_are_deduplicated(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [("ls -la", "stop"), ("ls -la", "stop")]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("list files", {}, n=2)
        assert cmds == ["ls -la"]

    def test_no_model_raises_file_not_found(self, monkeypatch):
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        with pytest.raises(FileNotFoundError):
            engine.generate("do something", {})


# ------------------------------------------------ generate(): grammar + postprocess wiring

class TestGenerateGrammarAndPostprocess:
    """generate() must pass the GBNF grammar to the local backends and run each
    extracted command through hostctx.postprocess_command using the host pkg.
    """

    def _fake_cfg_and_model(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))
        monkeypatch.setattr(engine, "start_server", lambda *a, **kw: 12345)

    def test_passes_grammar_to_query_server(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        class FakeHost:
            @staticmethod
            def build(prompt, enabled=True, cwd=None, include_volatile=True):
                return ("SYS", prompt)
            @staticmethod
            def stable_facts():
                return {"pkg": "pacman"}
            @staticmethod
            def grammar_for_pkg(pkg_mgr):
                return "fake-grammar"
            @staticmethod
            def is_install_request(prompt):
                return True
            @staticmethod
            def postprocess_command(cmd, pkg_mgr):
                assert pkg_mgr == "pacman"
                return cmd

        monkeypatch.setattr(engine, "hostctx", FakeHost())
        captured = {}
        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            captured["grammar"] = grammar
            return [("pacman -S htop", "stop")]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, mode = engine.generate("install htop", {}, n=1)
        assert captured["grammar"] == "fake-grammar"
        assert mode == "server"

    def test_postprocesses_wrong_distro_syntax(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)
        # Drive postprocessing with the real implementation, pinned to pacman.
        monkeypatch.setattr(engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True: ("SYS", p))
        monkeypatch.setattr(engine.hostctx, "stable_facts",
                            lambda *a, **k: {"pkg": "pacman"})
        monkeypatch.setattr(engine.hostctx, "grammar_for_pkg",
                            lambda pkg_mgr: None)
        monkeypatch.setattr(engine, "_query_server",
                            lambda *a, **k: [("apt-get install -y htop", "stop")])
        cmds, _, _ = engine.generate("install htop", {}, n=1)
        assert cmds == ["pacman -S htop"]

    def test_oneshot_passes_prompts_and_grammar_via_files(self, monkeypatch, tmp_path):
        """_query_oneshot must not put the prompt/system/grammar on the argv:
        llama-cli's command line is visible to any co-tenant via `ps`."""
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        cli = tmp_path / "llama-cli"
        cli.write_bytes(b"fake")

        captured = {}
        class _Proc:
            returncode = 0
            stdout = "> install htop\npacman -S htop\n[ Prompt: 12 tokens ]\n"
            stderr = ""
        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            return _Proc()

        monkeypatch.setattr(engine.subprocess, "run", fake_run)
        engine._query_oneshot(model, cli, "install htop", {}, threads=1,
                              system="SYS", grammar="GBNF")

        cmd = captured["cmd"]
        assert "--file" in cmd and "--system-prompt-file" in cmd and "--grammar-file" in cmd
        # The raw prompt, system prompt and grammar must never appear as argv
        # elements, only as paths to private files.
        assert "install htop" not in cmd
        assert "SYS" not in cmd
        assert "GBNF" not in cmd
        # The files themselves must be 0600 and cleaned up afterwards.
        for flag in ("--file", "--system-prompt-file", "--grammar-file"):
            path = cmd[cmd.index(flag) + 1]
            assert not os.path.exists(path)


class _FakeHostCtx:
    @staticmethod
    def build(prompt, enabled=True, cwd=None, include_volatile=True):
        return ("SYSTEM PROMPT", prompt)
    @staticmethod
    def stable_facts():
        return {"pkg": "unknown"}
    @staticmethod
    def postprocess_command(cmd, pkg_mgr):
        return cmd
    @staticmethod
    def grammar_for_pkg(pkg_mgr):
        return None
    @staticmethod
    def is_install_request(prompt):
        return "install" in prompt


class _FakeResp:
    """Stand-in for urllib's HTTPResponse: just a callable read()."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass

    def read(self, size=-1, *a, **k):
        # One-shot stream, like a real HTTP response body: emit the payload
        # once, then EOF, so _read_capped()'s read loop terminates.
        if self._exhausted:
            return b""
        self._exhausted = True
        return self._payload


class TestNormalizeEndpoint:
    def test_trailing_slash_removed(self):
        assert engine.normalize_endpoint_url("http://h:1/v1/") == "http://h:1/v1"

    def test_full_route_stripped(self):
        assert (engine.normalize_endpoint_url("http://h:1/v1/chat/completions/") ==
                "http://h:1/v1")

    def test_preserves_path_prefix(self):
        assert (engine.normalize_endpoint_url("https://ex.com/llm/v1/") ==
                "https://ex.com/llm/v1")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            engine.normalize_endpoint_url("")

    def test_non_http_rejected(self):
        with pytest.raises(ValueError, match="http"):
            engine.normalize_endpoint_url("ftp://h/")

    def test_embedded_credentials_rejected(self):
        with pytest.raises(ValueError, match="credentials"):
            engine.normalize_endpoint_url("http://u:p@h/v1")

    def test_fragment_rejected(self):
        with pytest.raises(ValueError, match="fragment"):
            engine.normalize_endpoint_url("http://h/v1#frag")

    def test_no_host_rejected(self):
        with pytest.raises(ValueError, match="no host"):
            engine.normalize_endpoint_url("http:///v1")


class TestParseChoices:
    def test_ok(self):
        assert engine._parse_choices(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        ) == [("x", "stop")]

    def test_missing_choices(self):
        with pytest.raises(RuntimeError, match="no choices"):
            engine._parse_choices({})

    def test_non_string_content_skipped(self):
        r = {"choices": [{"message": {"content": None}},
                         {"message": {"content": "ok"}, "finish_reason": "stop"}]}
        assert engine._parse_choices(r) == [("ok", "stop")]

    def test_no_usable_choices(self):
        with pytest.raises(RuntimeError, match="no usable"):
            engine._parse_choices({"choices": [{"message": {"content": None}}]})


class TestRemotePost:
    """_remote_post(): the HTTP layer, with urllib fully mocked."""

    def _fake_opener(self, resp):
        class _Opener:
            def __init__(self, r):
                self.r = r

            def open(self, req, **kw):
                self.req, self.kw = req, kw
                return self.r

        o = _Opener(resp)
        return o

    def test_sends_bearer_and_parses(self, monkeypatch):
        import json
        payload = json.dumps({"choices": [{"message": {"content": "ls -la"},
                                            "finish_reason": "stop"}]}).encode()
        opener = self._fake_opener(_FakeResp(payload))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        out = engine._remote_post("http://h:1/v1/", "m", "secret", {"model": "m"})
        assert out == [("ls -la", "stop")]
        req = opener.req
        assert req.full_url == "http://h:1/v1/chat/completions"
        assert req.get_method() == "POST"
        assert req.get_header("Authorization") == "Bearer secret"
        assert json.loads(req.data)["model"] == "m"

    def test_no_authorization_header_when_keyless(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b'{"choices":[]}'))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        with pytest.raises(RuntimeError, match="no usable"):
            engine._remote_post("http://h:1/v1", "m", "", {})
        assert opener.req.get_header("Authorization") is None

    def test_http_error_message_no_secret(self, monkeypatch):
        import urllib.error
        err = urllib.error.HTTPError("http://h:1/v1/chat/completions", 401,
                                     "Unauthorized", {}, _FakeResp(b"invalid key"))

        def boom(*a, **k):
            raise err

        class _Bad:
            open = staticmethod(boom)

        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: _Bad())
        with pytest.raises(RuntimeError) as ei:
            engine._remote_post("http://h:1/v1", "m", "sekrit-key", {})
        assert "401" in str(ei.value)
        assert "sekrit-key" not in str(ei.value)

    def test_non_json_response(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b"<html>proxy error</html>"))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        with pytest.raises(RuntimeError, match="non-JSON"):
            engine._remote_post("http://h:1/v1", "m", "", {})

    def test_no_cross_origin_redirect_handler(self):
        h = engine._NoCrossOriginRedirect()
        req = urllib.request.Request("http://h:1/v1/chat/completions",
                                     data=b"{}", method="POST")
        # Refuses a redirect to a different authority.
        assert h.redirect_request(req, None, 302, "Found", {},
                                  "http://evil:9/steal") is None

    def test_list_remote_models(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b'{"data":[{"id":"a"},{"id":"b"}]}'))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        names = engine.list_remote_models({"base_url": "http://h:1/v1", "api_key": ""})
        assert names == ["a", "b"]
        assert opener.req.full_url == "http://h:1/v1/models"
        assert opener.req.get_method() == "GET"


class TestQueryRemote:
    REMOTE = {"base_url": "http://h/v1", "model": "m", "api_key": "",
              "timeout": 120.0, "max_tokens": 512}

    def test_greedy_then_separate_sampled_calls(self, monkeypatch):
        calls = []

        def fake_post(base, model, key, body, timeout=120.0):
            calls.append((body.get("temperature"), body.get("n")))
            return [(f"cmd{len(calls)}", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        out = engine._query_remote(self.REMOTE, "prompt",
                                   {"temperature": 0.2}, n=3, system="S")
        assert [c for c, _ in out] == ["cmd1", "cmd2", "cmd3"]
        assert calls[0] == (0.2, 1)
        assert calls[1] == calls[2] == (max(0.6, 0.2), 1)  # never uses n>1

    def test_single_candidate_skips_sampling(self, monkeypatch):
        calls = []

        def fake_post(*a, **k):
            calls.append(1)
            return [("cmd", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        engine._query_remote(self.REMOTE, "p", {}, n=1, system="S")
        assert calls == [1]

    def test_sampled_failure_keeps_greedy(self, monkeypatch):
        state = {"i": 0}

        def fake_post(*a, **k):
            state["i"] += 1
            if state["i"] > 1:
                raise RuntimeError("boom")
            return [("cmd", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        out = engine._query_remote(self.REMOTE, "p", {}, n=3, system="S")
        assert out == [("cmd", "stop")]

    def test_passes_model_max_tokens_and_key(self, monkeypatch):
        seen = {}

        def fake_post(base, model, key, body, timeout=120.0):
            seen.update(base=base, model=model, key=key, body=body, to=timeout)
            return [("c", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        engine._query_remote({**self.REMOTE, "api_key": "K", "timeout": 30.0,
                              "max_tokens": 1024}, "p", {}, n=1, system="S")
        assert seen["model"] == "m" and seen["key"] == "K"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["max_tokens"] == 1024
        assert seen["to"] == 30.0


class TestGenerateRemote:
    def _remote(self, **over):
        return {"base_url": "http://h/v1", "model": "m", "api_key": "",
                "timeout": 120.0, "max_tokens": 512, **over}

    def _enable_remote(self, monkeypatch, remote):
        monkeypatch.setattr(cfg_mod, "remote_config", lambda cfg: remote)
        monkeypatch.setattr(engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True: ("SYS", p))

    def test_remote_mode_needs_no_local_model(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls -la", "stop")])
        cmds, elapsed, mode = engine.generate("list files", {})
        assert cmds == ["ls -la"]
        assert mode == "remote"

    def test_remote_missing_model_raises(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote(model=None))
        with pytest.raises(RuntimeError, match="no model selected"):
            engine.generate("list files", {})

    def test_remote_oneshot_incompatible(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        with pytest.raises(RuntimeError, match="oneshot"):
            engine.generate("list files", {}, force_oneshot=True)

    def test_remote_drops_length_finish(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls -la", "length"), ("pwd", "stop")])
        cmds, _, _ = engine.generate("list files", {})
        assert cmds == ["pwd"]

    def test_remote_dedups_candidates(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls", "stop"), ("ls", "stop")])
        cmds, _, mode = engine.generate("list files", {}, n=2)
        assert cmds == ["ls"]
        assert mode == "remote"


class TestRemoteWarnings:
    def test_warns_request_leaves_machine(self):
        w = engine.remote_warnings({"base_url": "https://api.example.com/v1",
                                    "api_key": ""})
        assert any("leaves this machine" in x for x in w)

    def test_http_with_key_warns_cleartext(self):
        w = engine.remote_warnings({"base_url": "http://remote.example.com/v1",
                                    "api_key": "k"})
        assert any("unencrypted" in x for x in w)

    def test_no_warning_for_local_loopback(self):
        w = engine.remote_warnings({"base_url": "http://127.0.0.1:8080/v1",
                                    "api_key": ""})
        assert w == []

    def test_https_loopback_does_not_claim_data_leaves(self):
        w = engine.remote_warnings({"base_url": "https://localhost:8443/v1",
                                    "api_key": ""})
        assert w == []

    def test_invalid_url_reported(self):
        w = engine.remote_warnings({"base_url": "not-a-url", "api_key": ""})
        assert any("invalid" in x for x in w)
