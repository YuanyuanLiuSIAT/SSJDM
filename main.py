# python main.py --config=config.yaml

import sys
import yaml
from utils.utils import *
from runners import Runner

def parse_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    new_config = dict2namespace(config)

    # set random seed
    init_seeds(new_config.seed)

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = new_config.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_config.device = device

    return args, new_config


def main():

    args, config = parse_args_and_config()

    runner = Runner(args, config)
    if config.mode.pre_training:
        print("mode:pre_training")
        runner.pre_train()
    if config.mode.training:
        print("mode:training")
        runner.train()
    if config.mode.testing:
        print("mode:SSJDM_testing")
        runner.test()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())