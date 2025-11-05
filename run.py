import logging

from configs.path_config import set_path

def main():
    #setup logging config(需要调整参数为info或debug)
    logging.basicConfig(level=logging.INFO)

    #setup path config
    logging.info('Starting set paths...........')
    set_path()
    logging.info('Setting paths is finished!')


if __name__=='__main__':
    main()