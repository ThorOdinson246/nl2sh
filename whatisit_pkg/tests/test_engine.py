"""Tests for whatisit.engine, with the HTTP layer entirely mocked out.

No test here ever starts llama-server, touches the network, or requires a
model file: engine._post / engine._query_server / engine.start_server are all
monkeypatched, and "model" paths are just empty tmp_path files that only need
to exist.
"""
import os
import stat
import sys

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

        def fake_query_server(port, prompt, cfg, n, system=None):
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

        def fake_query_server(port, prompt, cfg, n, system=None):
            return [
                ("zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 -9 -n -j -0 a.zip .", "stop"),
                ("zip -r archive.zip .", "stop"),
            ]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("zip this folder", {}, n=2)
        assert cmds == ["zip -r archive.zip ."]

    def test_duplicate_candidates_are_deduplicated(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None):
            return [("ls -la", "stop"), ("ls -la", "stop")]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("list files", {}, n=2)
        assert cmds == ["ls -la"]

    def test_no_model_raises_file_not_found(self, monkeypatch):
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        with pytest.raises(FileNotFoundError):
            engine.generate("do something", {})


class _FakeHostCtx:
    @staticmethod
    def build(prompt, enabled=True, cwd=None):
        return ("SYSTEM PROMPT", prompt)
