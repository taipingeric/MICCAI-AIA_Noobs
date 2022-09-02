import yaml
import os


def load_config(args):
    config_path = args.config
    config_workspace_path = args.config_workspace
    print(f'Config path \n config: {config_path} \n config_workspace: {config_workspace_path}')
    config = load_yaml(config_path)
    config_workspace = load_yaml(config_workspace_path)
    # merge
    config = dict(config, **config_workspace)
    return config


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def check_save_dir(config):
    # init save dir
    config_dir = config['config_dir']
    ckpt_dir = config['model_dir']
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f'Create config dir: {config_dir}')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f'Create ckpt dir: {ckpt_dir}')


if __name__ == '__main__':
    print('init/configs')

