"""Tests for whatisit.cli argument parsing and dispatch.

These cover cases that were REAL bugs found by typing realistic requests:
  - unquoted natural language must not be swallowed by argparse subparsers
    ("look at queued tasks in slurm" once died with "invalid choice: slurm")
  - request text containing flag-like tokens ("find files -name test") must
    survive untouched
  - a subcommand only counts as one when it is the FIRST token
  - --quiet must still run the safety check and refuse a DANGER command

None of this starts a server or touches the network: engine.generate is
monkeypatched wherever cmd_query would otherwise call into it.
"""
import pytest

from whatisit import cli

# ------------------------------------------------------------------ QueryArgs

class TestQueryArgsHandParsing:
    def test_unflagged_natural_language_is_not_eaten(self):
        # This exact sentence used to die on argparse subparsers with
        # "invalid choice: 'slurm'" because "slurm" is not a subcommand but
        # looked like trailing positional noise to argparse.
        args = cli.QueryArgs(["look", "at", "queued", "tasks", "in", "slurm"])
        assert args.words == ["look", "at", "queued", "tasks", "in", "slurm"]
        assert args.num == 1
        assert args.execute is False

    def test_request_with_flag_like_text_survives(self):
        args = cli.QueryArgs(["find", "files", "-name", "test"])
        assert args.words == ["find", "files", "-name", "test"]

    def test_double_dash_separator_ends_flag_parsing(self):
        args = cli.QueryArgs(["--", "-n", "3", "do", "the", "thing"])
        assert args.words == ["-n", "3", "do", "the", "thing"]

    def test_dash_n_space_3_form(self):
        args = cli.QueryArgs(["-n", "3", "compress", "this", "folder"])
        assert args.num == 3
        assert args.words == ["compress", "this", "folder"]

    def test_dash_n3_glued_form(self):
        args = cli.QueryArgs(["-n3", "compress", "this", "folder"])
        assert args.num == 3
        assert args.words == ["compress", "this", "folder"]

    def test_leading_flags_before_request_are_all_consumed(self):
        args = cli.QueryArgs(["-e", "-q", "-t", "count", "lines"])
        assert args.execute is True
        assert args.quiet is True
        assert args.timing is True
        assert args.words == ["count", "lines"]

    def test_oneshot_flag(self):
        args = cli.QueryArgs(["--oneshot", "do", "a", "thing"])
        assert args.oneshot is True
        assert args.words == ["do", "a", "thing"]

    def test_missing_n_argument_raises(self):
        with pytest.raises(ValueError):
            cli.QueryArgs(["-n"])

    def test_no_flags_at_all(self):
        args = cli.QueryArgs(["show", "disk", "usage"])
        assert args.words == ["show", "disk", "usage"]
        assert args.num == 1

    def test_empty_argv(self):
        args = cli.QueryArgs([])
        assert args.words == []


# ------------------------------------------------------------------- routing

class TestSubcommandRoutingIsFirstTokenOnly:
    def test_subcommand_as_first_token_routes_to_subparser(self, monkeypatch):
        called = {}

        def fake_cmd_config(args, cfg):
            called["hit"] = True
            return 0

        monkeypatch.setattr(cli, "cmd_config", fake_cmd_config)
        # rebuild parser mapping so "config" dispatches to our stub
        parser = cli.build_parser()
        monkeypatch.setattr(cli, "build_parser", lambda: parser)
        rc = cli.main(["config"])
        assert rc == 0
        assert called.get("hit") is True

    def test_subcommand_word_buried_in_sentence_is_not_routed(self, monkeypatch):
        # "whatisit show me the git config" must stay a plain-English question,
        # not be treated as the `config` subcommand, because "config" is not
        # the first token.
        captured = {}

        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False):
            captured["prompt"] = prompt
            return (["git config --list"], 0.01, "server")

        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["show", "me", "the", "git", "config"])
        assert rc == 0
        assert captured["prompt"] == "show me the git config"

    def test_setup_doctor_stop_are_recognized_subcommands(self):
        assert cli.SUBCOMMANDS == {"setup", "doctor", "stop", "config"}


# --------------------------------------------------------------- cmd_query

class TestCmdQueryQuietDangerRefusal:
    def test_quiet_refuses_danger_command_exit_6(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False:
                (["rm -rf /"], 0.01, "server"))
        rc = cli.main(["-q", "delete", "everything"])
        assert rc == 6
        out = capsys.readouterr()
        # stdout must stay bare even when refusing -- nothing should be
        # emitted there that a $(...) capture could pick up and run.
        assert out.out == ""

    def test_quiet_prints_bare_command_on_success(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False:
                (["ls -la"], 0.01, "server"))
        rc = cli.main(["-q", "list", "files"])
        assert rc == 0
        out = capsys.readouterr()
        assert out.out.strip() == "ls -la"

    def test_no_model_found_reports_and_exits_3(self, monkeypatch):
        def raise_not_found(prompt, cfg, n=1, force_oneshot=False, quiet=False):
            raise FileNotFoundError("no model found -- run `whatisit setup`")
        monkeypatch.setattr(cli.engine, "generate", raise_not_found)
        rc = cli.main(["do", "something"])
        assert rc == 3

    def test_empty_request_after_flags_only(self):
        rc = cli.main(["-q"])
        assert rc == 0  # falls through to help, per main()'s "no words" branch

    def test_refuse_execute_in_windows(self, monkeypatch, capsys):
        # Patch the helper, not os.name: setting os.name="nt" on Linux makes
        # Path.home() (via load_config in main) raise RuntimeError.
        monkeypatch.setattr(cli, "_is_windows", lambda: True)
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False:
                (["ls -la"], 0.01, "server"))
        rc = cli.main(["-e", "list", "files"])
        assert rc == 7
        assert "disabled" in capsys.readouterr().out

# ------------------------------------------------------------------ parser

class TestBuildParser:
    def test_help_flag_does_not_crash(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["-h"])
        assert exc.value.code == 0

    def test_config_subparser_accepts_set(self):
        parser = cli.build_parser()
        args = parser.parse_args(["config", "--set", "threads=2"])
        assert args.sub == "config"
        assert args.set == ["threads=2"]

    def test_setup_subparser_flags(self):
        parser = cli.build_parser()
        args = parser.parse_args(["setup", "--model", "/tmp/m.gguf", "--copy"])
        assert args.model == "/tmp/m.gguf"
        assert args.copy is True


def test_trailing_flag_is_part_of_the_request_but_is_flagged():
    """`whatisit list files -e` sends "-e" to the model. That is deliberate
    (see QueryArgs) but silent, so it must at least be reported."""
    a = cli.QueryArgs(["list", "files", "-e"])
    assert a.execute is False
    assert a.words == ["list", "files", "-e"]
    assert a.stray_flags == ["-e"]


def test_leading_flag_still_acts_as_a_flag():
    a = cli.QueryArgs(["-e", "list", "files"])
    assert a.execute is True
    assert a.words == ["list", "files"]
    assert a.stray_flags == []


def test_double_dash_ends_flags_and_silences_the_note():
    """`--` means the user meant it literally, so no note."""
    a = cli.QueryArgs(["list", "files", "--", "-e"])
    assert a.words == ["list", "files", "-e"]
    assert a.stray_flags == []


def test_flag_shaped_words_that_are_not_our_flags_are_left_alone():
    """`find files -name test` must not warn: -name is not one of our flags."""
    a = cli.QueryArgs(["find", "files", "-name", "test"])
    assert a.words == ["find", "files", "-name", "test"]
    assert a.stray_flags == []


def test_subcommand_word_inside_a_question_stays_a_question():
    a = cli.QueryArgs(["how", "do", "I", "stop", "a", "stuck", "process"])
    assert a.words[0] == "how"
    assert a.stray_flags == []

