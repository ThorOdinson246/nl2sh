"""Tests for whatisit.extract.extract().

extract() turns raw model output into one runnable command line. Getting this
wrong is a real security concern, not a cosmetic one: escape sequences in
model output are stripped here specifically because they can conceal or
repaint the command a user is about to approve (see extract.py's module
docstring), so the control-byte-stripping path is exercised explicitly below
alongside the ordinary parsing cases (fences, prompts, preambles, <think>).
"""
from whatisit.extract import extract


class TestMarkdownFences:
    def test_fenced_with_language_tag(self):
        assert extract("```bash\nls -la\n```") == "ls -la"

    def test_fenced_without_language_tag(self):
        assert extract("```\nls -la\n```") == "ls -la"

    def test_unterminated_fence(self):
        # Small models sometimes never emit the closing ```.
        assert extract("```bash\nls -la") == "ls -la"

    def test_fence_with_comment_line_inside(self):
        assert extract("```bash\n# comment\nls -la\n```") == "ls -la"

    def test_prose_preamble_before_fence(self):
        assert extract("here's how:\n```bash\nls -la\n```") == "ls -la"


class TestThinkBlocks:
    def test_paired_think_block_is_removed(self):
        assert extract("<think>reasoning here</think>ls -la") == "ls -la"

    def test_unterminated_think_with_nothing_before_yields_empty(self):
        # An unterminated <think> means the model never stopped reasoning;
        # there is no answer to hand back.
        assert extract("<think>still thinking, never closes") == ""

    def test_unterminated_think_with_prose_before_still_empty(self):
        # The prose before the dangling <think> is itself just a preamble,
        # so nothing usable remains once it is stripped.
        assert extract("sure, here you go:\n<think>oops unterminated") == ""

    def test_multiline_reasoning_inside_paired_think(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nls -la"
        assert extract(raw) == "ls -la"


class TestPrompts:
    def test_dollar_prompt_stripped(self):
        assert extract("$ ls -la") == "ls -la"

    def test_angle_prompt_stripped(self):
        assert extract("> ls -la") == "ls -la"

    def test_hash_comment_line_is_skipped_not_returned(self):
        # Lines starting with "#" are treated as full-line comments and
        # skipped outright (see the loop in extract()); the following real
        # command line is what gets returned.
        assert extract("# just a comment\nls -la") == "ls -la"

    def test_hash_only_input_yields_empty(self):
        # A single line that looks like a "# " root-shell prompt is
        # indistinguishable from a comment line and is skipped entirely,
        # leaving no candidate line at all.
        assert extract("# ls -la") == ""

    def test_inline_backticks_stripped(self):
        assert extract("`ls -la`") == "ls -la"


class TestProsePreambles:
    def test_sure_heres_the_command(self):
        assert extract("Sure, here's the command:\nls -la") == "ls -la"

    def test_the_command_colon_prefix(self):
        assert extract("The command: ls -la") == "ls -la"

    def test_certainly_prefix(self):
        assert extract("Certainly, here you go:\nls -la") == "ls -la"


class TestControlAndAnsiStripping:
    """A real vulnerability: escape sequences can hide the true command."""

    def test_sgr_conceal_sequence_stripped(self):
        # \x1b[8m is "conceal" -- text following it can be invisible in a
        # real terminal while still being the command that gets executed.
        assert extract("\x1b[8mrm -rf /\x1b[0m") == "rm -rf /"

    def test_ordinary_color_codes_stripped(self):
        assert extract("\x1b[31mls -la\x1b[0m") == "ls -la"

    def test_osc_sequence_stripped(self):
        # OSC (\x1b]...BEL or \x1b]...ST) can be used to set a terminal
        # title or otherwise repaint output; it must not survive either.
        assert extract("\x1b]0;evil title\x07ls -la") == "ls -la"

    def test_bare_control_bytes_stripped(self):
        assert extract("ls\x00 -la") == "ls -la"


class TestEmptyAndGarbage:
    def test_empty_string(self):
        assert extract("") == ""

    def test_none_like_falsy_input(self):
        assert extract(None) == ""

    def test_whitespace_only(self):
        assert extract("   \n\n  ") == ""

    def test_garbage_no_newline(self):
        assert extract("asdkjfh q9w8e7r") == "asdkjfh q9w8e7r"
