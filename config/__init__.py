import yaml
import json
import types

def load(path_to_config_yaml="./config.yaml"):

    with open(path_to_config_yaml) as f:
        dct = yaml.safe_load(f)
    def load_object(dct):
        return types.SimpleNamespace(**dct)

    cfg = json.loads(json.dumps(dct), object_hook=load_object)

    return cfg