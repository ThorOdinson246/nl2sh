"""Locks the identifiers that are published and cannot be changed freely.

The GGUF filenames and Hugging Face repo IDs are names that already exist on a
model hub. They look like they belong to this package, so a find-and-replace
over the tool's own name will rewrite them and silently 404 every download in
the install instructions. These tests make that a failing build instead.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whatisit import config as cfg_mod  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent


def _pyproject_field(pattern):
    text = (PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"no match for {pattern!r} in pyproject.toml"
    return m.group(1)


class TestPublishedModelNames:
    def test_model_filename_is_the_one_on_the_hub(self):
        assert cfg_mod.MODEL_NAME == "nl2sh-1.5b-Q4_K_M.gguf"

    def test_readmes_reference_the_real_gguf_files(self):
        for readme in (REPO_ROOT / "README.md", PKG_ROOT / "README.md"):
            if not readme.exists():
                continue
            text = readme.read_text(encoding="utf-8")
            assert "nl2sh-1.5b-Q4_K_M.gguf" in text, f"{readme.name} lost the model filename"
            assert "ThorOdinson246/nl2sh-1.5b-Q4_K_M" in text, f"{readme.name} lost the repo ID"


class TestApplicationNames:
    def test_app_and_legacy_names(self):
        assert cfg_mod.APP_NAME == "whatisit"
        assert cfg_mod.LEGACY_NAME == "nl2sh"

    def test_xdg_dirs_use_the_app_name(self, monkeypatch, tmp_path):
        for var in ("WHATISIT_CONFIG_DIR", "WHATISIT_DATA_DIR",
                    "NL2SH_CONFIG_DIR", "NL2SH_DATA_DIR",
                    "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)
        assert cfg_mod.config_dir().name == "whatisit"
        assert cfg_mod.data_dir().name == "whatisit"


class TestPackagingMetadata:
    def test_distribution_name(self):
        assert _pyproject_field(r'^name = "([^"]+)"') == "whatisit"

    def test_only_the_whatisit_console_script_is_shipped(self):
        """A second `nl2sh` entry point would collide with the unrelated PyPI
        project of that name: whichever installed last would win."""
        text = (PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
        scripts = dict(re.findall(r'^\s*([\w.-]+)\s*=\s*"([^"]+)"', block, re.MULTILINE))
        assert scripts == {"whatisit": "whatisit.cli:main"}

    def test_version_matches_dunder_version(self):
        """publish.yml refuses to release when these disagree; fail here first."""
        from whatisit import __version__
        assert _pyproject_field(r'^version = "([^"]+)"') == __version__
