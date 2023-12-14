# NOTE: do not run this file directly, use the run.sh script instead.

import os
import argparse
from sys import exit
import logging
import configparser
logging.basicConfig(level=logging.NOTSET)
handle = "floorplan-main"
main_logger = logging.getLogger(handle)



def main(debug: bool = False):
    """Main function."""
    env_file_path = os.environ['RMBASE_FILE_PYTHON']
    parser = argparse.ArgumentParser(
                prog='PYML Organizer',
                description='All in one organizer for training and predicting models for RM.',
                epilog='See /doc for more details or contact the author.')
    parser.add_argument('-v', '--debug', default=True)
    parser.add_argument('-f', '--config', 
                        default=env_file_path,
                        help='config file path')
    parser.add_argument('-nl', '--no-labels',
                        default=False, action='store_true')
    parser.add_argument('-w', '--watch', default=False, action='store_true')
    args = parser.parse_args()
    
    main_logger.info('args => '+str(args))
    main_logger.info('env path => '+env_file_path)

    # prefer args from command line over env file
    config = configparser.ConfigParser()
    if args.config:
      config.read(args.config)
    else:
      config.read(os.environ['RMBASE_FILE_PYTHON'])
    
    main_logger.info('config => '+str(config.sections()))
    main_logger.info(config.get('DEFAULT', 'model_path'))
    exit(0)

if __name__ == '__main__':
    main(True)
