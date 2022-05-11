from data_filters import config


def test_get_configs_from_file_succeed():
    path = "config.yaml"
    try:
        _ = config.get_configs_from_file(path)
    except Exception:
        assert False
