"""Tests for nl2sh.config: XDG path resolution, env overrides, thread policy,
and the permission-hardened config write.

Every test isolates itself via monkeypatch env vars / tmp_path -- none of this
may read or write the real ~/.config or ~/.local/share.
"""
import json
import os
import stat

import pytest

from nl2sh import config as cfg_mod


def _clear_xdg(monkeypatch):
    for var in ("NL2SH_CONFIG_DIR", "NL2SH_DATA_DIR", "NL2SH_MODEL",
                "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)


class TestConfigDirResolution:
    def test_default_is_dot_config_nl2sh(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)
        assert cfg_mod.config_dir() == tmp_path / ".config" / "nl2sh"

    def test_xdg_config_home_override(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgcfg"))
        assert cfg_mod.config_dir() == tmp_path / "xdgcfg" / "nl2sh"

    def test_nl2sh_config_dir_wins_over_xdg(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgcfg"))
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "explicit"))
        assert cfg_mod.config_dir() == tmp_path / "explicit"

    def test_config_path_is_config_json_under_config_dir(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "c"))
        assert cfg_mod.config_path() == tmp_path / "c" / "config.json"


class TestDataDirResolution:
    def test_default_is_local_share_nl2sh(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)
        assert cfg_mod.data_dir() == tmp_path / ".local/share" / "nl2sh"

    def test_xdg_data_home_override(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
        assert cfg_mod.data_dir() == tmp_path / "xdgdata" / "nl2sh"

    def test_nl2sh_data_dir_wins_over_xdg(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
        monkeypatch.setenv("NL2SH_DATA_DIR", str(tmp_path / "explicit-data"))
        assert cfg_mod.data_dir() == tmp_path / "explicit-data"


class TestModelResolution:
    def test_nl2sh_model_env_used_when_it_exists(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        model = tmp_path / "custom.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setenv("NL2SH_MODEL", str(model))
        assert cfg_mod.find_model() == model

    def test_nl2sh_model_env_pointing_nowhere_returns_none(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_MODEL", str(tmp_path / "does-not-exist.gguf"))
        assert cfg_mod.find_model() is None

    def test_model_found_in_data_dir_models(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_DATA_DIR", str(tmp_path / "data"))
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        model = models_dir / cfg_mod.MODEL_NAME
        model.write_bytes(b"fake")
        assert cfg_mod.find_model() == model

    def test_no_model_anywhere_returns_none(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.chdir(tmp_path)
        assert cfg_mod.find_model() is None


class TestThreadPolicy:
    """Half the cores, capped at 4 -- decode is memory-bandwidth bound."""

    def test_explicit_threads_setting_wins(self):
        assert cfg_mod.resolve_threads({"threads": 7}) == 7

    def test_zero_or_missing_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        assert cfg_mod.resolve_threads({"threads": 0}) == 4
        assert cfg_mod.resolve_threads({}) == 4

    @pytest.mark.parametrize("cores,expected", [
        (1, 1),   # max(1, min(4, 0)) -> at least 1 thread
        (2, 1),
        (4, 2),
        (8, 4),
        (16, 4),  # capped at 4 even with plenty of cores
        (64, 4),
    ])
    def test_auto_is_half_cores_capped_at_4(self, monkeypatch, cores, expected):
        monkeypatch.setattr(os, "cpu_count", lambda: cores)
        assert cfg_mod.resolve_threads({"threads": 0}) == expected

    def test_cpu_count_none_defaults_safely(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        assert cfg_mod.resolve_threads({"threads": 0}) == 2


class TestSaveConfigPermissions:
    def test_config_dir_created_0700(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(cdir))
        cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        mode = stat.S_IMODE(cdir.stat().st_mode)
        assert mode == 0o700

    def test_config_file_created_0600(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(cdir))
        p = cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600

    def test_pre_existing_loosely_permissioned_file_gets_fixed(self, monkeypatch, tmp_path):
        # fchmod on the open fd must re-tighten a file that pre-dates this
        # fix (or was written by an older nl2sh), not just newly-created ones.
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        cdir.mkdir()
        p = cdir / "config.json"
        p.write_text("{}")
        os.chmod(p, 0o644)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(cdir))
        cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_written_config_round_trips_as_json(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "cfgdir"))
        cfg = dict(cfg_mod.DEFAULTS)
        cfg["threads"] = 3
        p = cfg_mod.save_config(cfg)
        assert json.loads(p.read_text())["threads"] == 3


class TestLoadConfig:
    def test_load_config_returns_defaults_when_absent(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "nope"))
        assert cfg_mod.load_config() == cfg_mod.DEFAULTS

    def test_load_config_merges_saved_values(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "cfgdir"))
        cfg_mod.save_config({**cfg_mod.DEFAULTS, "threads": 2})
        loaded = cfg_mod.load_config()
        assert loaded["threads"] == 2
        assert loaded["max_tokens"] == cfg_mod.DEFAULTS["max_tokens"]

    def test_corrupt_config_does_not_crash(self, monkeypatch, tmp_path):
        # A corrupt config must not brick the tool -- fall back to defaults.
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        cdir.mkdir()
        (cdir / "config.json").write_text("{not valid json")
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(cdir))
        assert cfg_mod.load_config() == cfg_mod.DEFAULTS
