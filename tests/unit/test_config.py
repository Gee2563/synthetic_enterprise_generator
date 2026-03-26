from synthetic_enterprise.app.settings import AppConfig, load_config


def test_config_loads_with_defaults() -> None:
    config = load_config()

    assert isinstance(config, AppConfig)
    assert config.package_name == "synthetic_enterprise"
    assert config.default_seed == 0
    assert config.default_locale == "en_US"
