import logging
import sys

from configs.path_config import set_path

def main():
    #setup logging config(需要调整参数为info或debug)
    logging.basicConfig(level=logging.INFO)

    #setup path config
    logging.info('Starting set paths...........')
    set_path()
    logging.info('Setting paths is finished!')

    #run the critic
    from trainer.train import run_critic
    ##要输入处理好的路径
    run_critic(
        port=6000,
        save_index=0,
        new_game=False,
        image_save_path="../env/screen_shot_buffer",
        output_video=False,
        task_name="farming_lite",
        task_id=0,
        checkpoint_interval=5,
        )


if __name__=='__main__':
    main()