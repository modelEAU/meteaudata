from filters import config


def test_get_configs_from_file_succeed():
    path = "tests/test_config.yaml"
    try:
        _ = config.get_configs_from_file(path)
        return True
    except Exception:
        return False
