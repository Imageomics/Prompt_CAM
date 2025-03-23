import argparse
from experiment.visualize_run import basic_vis
from utils.setup_logging import get_logger
from utils.misc import set_seed,load_yaml
import time

logger = get_logger('Prompt_CAM')


def main():
    args = setup_parser().parse_args()

    if args.config:
        yaml_config = load_yaml(args.config)
        for key, value in yaml_config.items():
            setattr(args, key, value)

    set_seed(args.random_seed)
    start = time.time()
    basic_vis(args)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')

def setup_parser():
    parser = argparse.ArgumentParser(description='Prompt_CAM')
    ######################## YAML Config #########################
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    ####################### Model #########################
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to the model checkpoint')         

    ####################### Visualization Configuration #########################
    parser.add_argument('--vis_attn', default=True, type=bool, help='visualize the attention map')
    parser.add_argument('--vis_cls', default=23, type=int, help='Class in the current Dataset to visualize')
    parser.add_argument('--nmbr_samples', default=10, type=int, help='Number of samples to visualize')
    parser.add_argument('--top_traits', default=4, type=int, help='Number of top traits per sample to visualize')

    parser.add_argument('--vis_outdir', default="./visualization", type=str, help='Output directory for visualization')
    ########################Misc#########################
    parser.add_argument('--gpu_num', default=1,
                        type=int,
                        help='Number of GPU (default: %(default)s)')
    parser.add_argument('--random_seed', default=42,
                        type=int,
                        help='Random seed (default: %(default)s)')

    return parser


if __name__ == '__main__':
    main()
