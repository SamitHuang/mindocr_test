import argparse
#from .mindocr import build_dataset, build_model

def create_parser():
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments (default='')')

    # The main parser. It inherits the --config argument for better help information.
    parser = argparse.ArgumentParser(description='OCR Training', parents=[parser_config])

    # System parameters
    group = parser.add_argument_group('System parameters')
    group.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    
    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', type=str, default='mobilenet_v2_035_224',
                       help='Name of model')

    return parser_config, parser

def parse_args():
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def main(args):
    
    # create dataset and loader, return mindspore loader 
    #loader_train = build_dataloader(name=args.dataset.name, config=args.dataset.config)
    
    
    #net = build_mode()
    pass


if __name__ == '__main__':

    main()
