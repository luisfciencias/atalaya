# tools to load the config params
import yaml

def load_config(config_file):
    """
    Load json config file into a dictionary.
    """
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg

# testing
params_config = load_config('config_params.yaml')
print('Config parameters:')
print(params_config)
