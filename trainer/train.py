import logging
import time
import os

from benchmark.stardojo_main.env.llm_env import StarDojoLLM
from benchmark.stardojo_main.env.llm_env import SkillExecutor
from benchmark.stardojo_main.env.tasks.utils import load_task
from benchmark.stardojo_main.env.stardew_env import *
from benchmark.stardojo_main.agent.stardojo.stardojo_react_agent import *
from benchmark.stardojo_main.env.tasks.base import *

def run_sampling(
    port: int,
    save_index: int,
    new_game: bool,
    image_save_path: str,
    output_video: bool,
    task_name: str,
    task_id: int,
    checkpoint_interval: int = 5,
    env_config_path: str = "./conf/env_config_stardew.json",
    llm_config_path: str = "./conf/openai_config.json",
    embed_config_path: str = "./conf/openai_config.json"
):

    logging.basicConfig(
        filename='app.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    config.checkpoint_interval = checkpoint_interval

    config.load_env_config(env_config_path)

    task = load_task.load_task(task_name, task_id)
    
    if task.difficulty == "easy":
        config.max_turn_count = 30
    elif task.difficulty == "medium":
        config.max_turn_count = 50
    else:
        config.max_turn_count = 200

    react_agent = PipelineRunner(
        llm_provider_config_path=llm_config_path,
        embed_provider_config_path=embed_config_path,
        task_description=task.llm_description,
        use_self_reflection=False,
        use_task_inference=False
    )
    atexit.register(exit_cleanup, react_agent)

    stardojo_env = StarDojoLLM(
        port=port,
        save_index=save_index,
        new_game=new_game,
        image_save_path=image_save_path,
        agent=react_agent,
        needs_pausing=True,
        image_obs=True,
        task=task,
        output_video=output_video
    )

    time.sleep(1)

    terminated = truncated = False
    step = 0
    while not terminated and not truncated:
        try:
            logging.info(f"Running Task: {task.llm_description}, Step {step}")
            obs, reward, terminated, truncated, info = stardojo_env.step()
            step += 1

            if step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(react_agent.checkpoint_path, f'checkpoint_{step:06d}.json')
                # react_agent.memory.save(checkpoint_path)

            if step > config.max_turn_count:
                print('Max steps reached, exiting.')
                break

            if terminated:
                stardojo_env.action_proxy.exit_to_title()
                print('Task completed, exiting.')
                break

        except KeyboardInterrupt:
            print('Interrupted by user.')
            react_agent.pipeline_shutdown()
            stardojo_env.exit()
            break

    react_agent.pipeline_shutdown()
    stardojo_env.exit()
