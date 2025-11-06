import os
import json
from pathlib import Path

def set_model_config(model_type='qwen'):
    #get config file root
    project_root=Path(__file__).parent
    model_config_root=str(os.path.join(project_root,'agent\\provider\\configs'))

    #get config file
    if model_type=='qwen':
        model_config_path=str(os.path.join(model_config_root,'qwen_conf.json'))
    
    #get benchmark config file
    bench_config_path=os.path.join(project_root,'\\benchmark\\stardojo_main\\agent\\conf\\opensrc_config.json')

    #get model config
    with open(model_config_path,'r',encoding='utf-8') as file:
        conf=json.load(file)

        critic_api_url=conf['critic'].get('api_url','')
        critic_api_key=conf['critic'].get('api_key','')

        
    with open(str(bench_config_path),'w'):
        pass

