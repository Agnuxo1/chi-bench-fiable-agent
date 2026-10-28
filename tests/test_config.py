import pytest

from fiable_agent.config import (
    CONFIG_DIR,
    ConfigError,
    api_model_id,
    has_api_key,
    load_config,
    resolve_model,
)


def test_aliases_resolve_to_price_table_ids():
    assert resolve_model("sonnet") == "anthropic/claude-sonnet-5"
    assert resolve_model("opus") == "anthropic/claude-opus-5"
    assert resolve_model("haiku") == "anthropic/claude-haiku-4-5"
    assert resolve_model("claude-opus-4-7") == "anthropic/claude-opus-4-7"
    assert api_model_id("anthropic/claude-sonnet-5") == "claude-sonnet-5"


def test_default_profile_loads_full_pipeline():
    cfg = load_config()
    assert cfg.profile == "default"
    assert cfg.agent_model == "anthropic/claude-sonnet-5"
    assert cfg.critic_model == cfg.agent_model  # cost consistent with aggregator.py
    assert cfg.rag.enabled and cfg.checklist.enabled and cfg.critic.enabled
    assert cfg.critic.judges == ["contract", "grounding", "policy"]


@pytest.mark.parametrize("profile,model", [("opus", "anthropic/claude-opus-5"),
                                           ("haiku", "anthropic/claude-haiku-4-5"),
                                           ("sonnet", "anthropic/claude-sonnet-5")])
def test_model_profiles(profile, model):
    assert load_config(profile).agent_model == model


def test_experiment_profiles_toggle_components():
    e1 = load_config("e1_baseline")
    assert not (e1.rag.enabled or e1.checklist.enabled or e1.critic.enabled)
    e2 = load_config("e2_rag")
    assert e2.rag.enabled and not e2.checklist.enabled and not e2.critic.enabled
    e3 = load_config("e3_checklist")
    assert e3.checklist.enabled and not e3.critic.enabled
    assert load_config("e4_critic_heuristic").critic.mode == "heuristic"


def test_env_model_override():
    cfg = load_config(env={"FIABLE_AGENT_MODEL": "anthropic/claude-opus-5"})
    assert cfg.agent_model == "anthropic/claude-opus-5"
    assert cfg.critic_model == "anthropic/claude-opus-5"


def test_secret_like_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  agent: sonnet\n  api_key: whatever\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="credential"):
        load_config(p)


def test_secret_like_value_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    fake = "sk-ant-" + "x" * 30  # synthetic value, not a real key
    p.write_text(f"model:\n  agent: sonnet\n  extra_request_params:\n    note: {fake}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="secret"):
        load_config(p)


def test_unknown_keys_and_bad_enums(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text(f"extends: {(CONFIG_DIR / 'default.yaml').as_posix()}\nrag:\n  backend: elastic\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_has_api_key_never_returns_value():
    assert has_api_key({}) is False
    assert has_api_key({"ANTHROPIC_API_KEY": "   "}) is False
    assert has_api_key({"ANTHROPIC_API_KEY": "dummy-not-a-key"}) is True


def test_packaged_configs_have_no_secrets():
    for path in CONFIG_DIR.glob("*.yaml"):
        load_config(path)  # raises ConfigError if anything shaped like a secret is present

