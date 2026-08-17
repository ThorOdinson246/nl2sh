"""Tests for whatisit.config: XDG path resolution, env overrides, thread policy,
and the permission-hardened config write.

Every test isolates itself via monkeypatch env vars / tmp_path -- none of this
may read or write the real ~/.config or ~/.local/share.
"""
import json
import os
import stat
import sys

import pytest

from whatisit import config as cfg_mod


def _clear_xdg(monkeypatch):
    # Both families: the NL2SH_* names are still honoured as a fallback, so
    # leaving one set in the ambient environment would silently steer a test.
    for suffix in ("CONFIG_DIR", "DATA_DIR", "MODEL"):
        for prefix in ("WHATISIT_", "NL2SH_"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)


class TestConfigDirResolution:
    def test_default_is_dot_config_whatisit(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)
        assert cfg_mod.config_dir() == tmp_path / ".config" / "whatisit"

    def test_xdg_config_home_override(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgcfg"))
        assert cfg_mod.config_dir() == tmp_path / "xdgcfg" / "whatisit"

    def test_whatisit_config_dir_wins_over_xdg(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgcfg"))
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "explicit"))
        assert cfg_mod.config_dir() == tmp_path / "explicit"

    def test_config_path_is_config_json_under_config_dir(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "c"))
        assert cfg_mod.config_path() == tmp_path / "c" / "config.json"


class TestDataDirResolution:
    def test_default_is_local_share_whatisit(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)
        assert cfg_mod.data_dir() == tmp_path / ".local/share" / "whatisit"

    def test_xdg_data_home_override(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
        assert cfg_mod.data_dir() == tmp_path / "xdgdata" / "whatisit"

    def test_whatisit_data_dir_wins_over_xdg(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "explicit-data"))
        assert cfg_mod.data_dir() == tmp_path / "explicit-data"


class TestModelResolution:
    def test_whatisit_model_env_used_when_it_exists(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        model = tmp_path / "custom.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_MODEL", str(model))
        assert cfg_mod.find_model() == model

    def test_whatisit_model_env_pointing_nowhere_returns_none(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_MODEL", str(tmp_path / "does-not-exist.gguf"))
        assert cfg_mod.find_model() is None

    def test_model_found_in_data_dir_models(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        model = models_dir / cfg_mod.MODEL_NAME
        model.write_bytes(b"fake")
        assert cfg_mod.find_model() == model

    def test_no_model_anywhere_returns_none(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
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
    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_config_dir_created_0700(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(cdir))
        cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        mode = stat.S_IMODE(cdir.stat().st_mode)
        assert mode == 0o700

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_config_file_created_0600(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(cdir))
        p = cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_pre_existing_loosely_permissioned_file_gets_fixed(self, monkeypatch, tmp_path):
        # fchmod on the open fd must re-tighten a pre-existing file, not just
        # newly-created ones.
        _clear_xdg(monkeypatch)
        cdir = tmp_path / "cfgdir"
        cdir.mkdir()
        p = cdir / "config.json"
        p.write_text("{}")
        os.chmod(p, 0o644)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(cdir))
        cfg_mod.save_config(dict(cfg_mod.DEFAULTS))
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_written_config_round_trips_as_json(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfgdir"))
        cfg = dict(cfg_mod.DEFAULTS)
        cfg["threads"] = 3
        p = cfg_mod.save_config(cfg)
        assert json.loads(p.read_text())["threads"] == 3


class TestLoadConfig:
    def test_load_config_returns_defaults_when_absent(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "nope"))
        assert cfg_mod.load_config() == cfg_mod.DEFAULTS

    def test_load_config_merges_saved_values(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfgdir"))
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
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(cdir))
        assert cfg_mod.load_config() == cfg_mod.DEFAULTS


class TestLegacyEnvFallback:
    """The pre-rename NL2SH_* names stay honoured, permanently."""

    def test_old_name_used_when_new_is_unset(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "legacy"))
        assert cfg_mod.config_dir() == tmp_path / "legacy"

    def test_new_name_wins_when_both_set(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_CONFIG_DIR", str(tmp_path / "legacy"))
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "current"))
        assert cfg_mod.config_dir() == tmp_path / "current"

    def test_old_model_var_still_resolves(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setenv("NL2SH_MODEL", str(model))
        assert cfg_mod.find_model() == model

    def test_old_data_dir_still_resolves(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setenv("NL2SH_DATA_DIR", str(tmp_path / "legacy-data"))
        assert cfg_mod.data_dir() == tmp_path / "legacy-data"


class TestLegacyDirMigration:
    def _home(self, monkeypatch, tmp_path):
        _clear_xdg(monkeypatch)
        monkeypatch.setattr(cfg_mod.Path, "home", lambda: tmp_path)

    def test_moves_config_and_reports_once(self, monkeypatch, tmp_path):
        self._home(monkeypatch, tmp_path)
        old = tmp_path / ".config" / "nl2sh"
        old.mkdir(parents=True)
        (old / "config.json").write_text('{"threads": 3}')

        msgs = cfg_mod.migrate_legacy_dirs(echo=lambda m: None)
        assert any("moved" in m for m in msgs)
        assert not old.exists()
        assert cfg_mod.load_config()["threads"] == 3

        # Idempotent: a second run has nothing to say.
        assert cfg_mod.migrate_legacy_dirs(echo=lambda m: None) == []

    def test_never_overwrites_an_existing_new_dir(self, monkeypatch, tmp_path):
        self._home(monkeypatch, tmp_path)
        old = tmp_path / ".config" / "nl2sh"
        old.mkdir(parents=True)
        (old / "config.json").write_text('{"threads": 3}')
        new = tmp_path / ".config" / "whatisit"
        new.mkdir(parents=True)
        (new / "config.json").write_text('{"threads": 9}')

        assert cfg_mod.migrate_legacy_dirs(echo=lambda m: None) == []
        assert old.exists()
        assert cfg_mod.load_config()["threads"] == 9

    def test_skipped_when_dir_set_explicitly(self, monkeypatch, tmp_path):
        self._home(monkeypatch, tmp_path)
        (tmp_path / ".config" / "nl2sh").mkdir(parents=True)
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "elsewhere"))
        assert not any("config" in m for m in
                       cfg_mod.migrate_legacy_dirs(echo=lambda m: None))

    def test_noop_when_nothing_to_migrate(self, monkeypatch, tmp_path):
        self._home(monkeypatch, tmp_path)
        assert cfg_mod.migrate_legacy_dirs(echo=lambda m: None) == []

    def test_absolute_symlinks_survive_the_move(self, monkeypatch, tmp_path):
        """setup registers the model as a symlink; it must still resolve."""
        self._home(monkeypatch, tmp_path)
        real = tmp_path / "store" / "nl2sh-1.5b-Q4_K_M.gguf"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"gguf")
        old = tmp_path / ".local/share" / "nl2sh" / "models"
        old.mkdir(parents=True)
        (old / cfg_mod.MODEL_NAME).symlink_to(real)

        msgs = cfg_mod.migrate_legacy_dirs(echo=lambda m: None)
        assert not any("no longer resolves" in m for m in msgs)
        moved = tmp_path / ".local/share" / "whatisit" / "models" / cfg_mod.MODEL_NAME
        assert moved.is_symlink() and moved.exists()
        assert cfg_mod.find_model() == moved

    def test_broken_relative_symlink_is_reported(self, monkeypatch, tmp_path):
        self._home(monkeypatch, tmp_path)
        old = tmp_path / ".local/share" / "nl2sh" / "bin"
        old.mkdir(parents=True)
        (old / "llama-server").symlink_to("../../nl2sh/bin/real")

        msgs = cfg_mod.migrate_legacy_dirs(echo=lambda m: None)
        assert any("no longer resolves" in m for m in msgs)


class TestRemoteConfig:
    """remote_config() resolution: when it activates, env-over-config, secrets."""

    def _clear(self, monkeypatch):
        _clear_xdg(monkeypatch)
        for suff in ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY",
                     "OPENAI_TIMEOUT", "OPENAI_MAX_TOKENS"):
            for pref in ("WHATISIT_", "NL2SH_"):
                monkeypatch.delenv(pref + suff, raising=False)

    def test_none_when_no_base_url(self, monkeypatch):
        self._clear(monkeypatch)
        assert cfg_mod.remote_config({}) is None

    def test_empty_base_url_means_local(self, monkeypatch):
        self._clear(monkeypatch)
        assert cfg_mod.remote_config({"openai_base_url": ""}) is None

    def test_resolves_from_config(self, monkeypatch):
        self._clear(monkeypatch)
        c = cfg_mod.remote_config({"openai_base_url": "http://h:1/v1",
                                   "openai_model": "m",
                                   "openai_api_key": "k"})
        assert c["base_url"] == "http://h:1/v1"
        assert c["model"] == "m"
        assert c["api_key"] == "k"
        assert c["max_tokens"] == 512
        assert c["timeout"] == 120.0

    def test_env_overrides_config(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("WHATISIT_OPENAI_BASE_URL", "http://env:2/v1")
        monkeypatch.setenv("WHATISIT_OPENAI_MODEL", "env-model")
        monkeypatch.setenv("WHATISIT_OPENAI_API_KEY", "env-key")
        c = cfg_mod.remote_config({"openai_base_url": "http://cfg:1/",
                                   "openai_model": "cfg-model",
                                   "openai_api_key": "cfg-key"})
        assert c["base_url"] == "http://env:2/v1"
        assert c["model"] == "env-model"
        assert c["api_key"] == "env-key"

    def test_keyless_by_default(self, monkeypatch):
        self._clear(monkeypatch)
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1"})
        assert c["api_key"] == ""
        assert c["model"] is None

    def test_timeout_and_max_tokens_overridable(self, monkeypatch):
        self._clear(monkeypatch)
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                   "openai_timeout": 30, "openai_max_tokens": 1000})
        assert c["timeout"] == 30
        assert c["max_tokens"] == 1000

    def test_env_timeout_precedence(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("WHATISIT_OPENAI_MAX_TOKENS", "256")
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                   "openai_max_tokens": 999})
        assert c["max_tokens"] == 256

    def test_garbage_numeric_values_fall_back(self, monkeypatch):
        self._clear(monkeypatch)
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                   "openai_timeout": "abc", "openai_max_tokens": []})
        assert c["timeout"] == 120.0
        assert c["max_tokens"] == 512

    def test_whitespace_only_base_url_means_local(self, monkeypatch):
        self._clear(monkeypatch)
        assert cfg_mod.remote_config({"openai_base_url": "   "}) is None

    def test_whitespace_only_model_is_unset(self, monkeypatch):
        self._clear(monkeypatch)
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                   "openai_model": "  "})
        assert c["model"] is None

    def test_empty_env_overrides_config_and_goes_local(self, monkeypatch):
        """WHATISIT_OPENAI_BASE_URL= must not fall through to config.json."""
        self._clear(monkeypatch)
        monkeypatch.setenv("WHATISIT_OPENAI_BASE_URL", "")
        assert cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                      "openai_model": "m"}) is None

    def test_empty_env_key_clears_stored_key(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("WHATISIT_OPENAI_API_KEY", "")
        c = cfg_mod.remote_config({"openai_base_url": "http://h/v1",
                                   "openai_api_key": "stored"})
        assert c["api_key"] == ""
