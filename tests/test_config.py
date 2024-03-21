from data_filters import config


def test_get_configs_from_file_succeed():
    path = "yaml_files/config.yaml"
    try:
        _ = config.get_config_from_file(path)
    except Exception:
        assert False
