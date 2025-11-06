import os
from pathlib import Path
import sys

def set_path():
    #root
    project_root=Path(__file__).parent.parent
    #root parent
    root_parent=project_root.parent
    #agent 
    agent_path=str(os.path.join(project_root,'agent'))
    #other path
    project_dir={
        'root':project_root,
        'root_parent':root_parent,
        'trainer':os.path.join(project_root,'trainer'),
        'test':os.path.join(project_root,'test'),
        'configs':os.path.join(project_root,'configs'),

        'benchmark':os.path.join(project_root,'benchmark'),
        'stardojo_main':os.path.join(project_root,'benchmark\\stardojo_main'),
        'stardojo_agent':os.path.join(project_root,'benchmark\\stardojo_main\\agent'),
        'stardojo_env':os.path.join(project_root,'benchmark\\stardojo_main\\env'),
        'stardojo_':os.path.join(project_root,'benchmark\\stardojo_main\\agent\\stardojo'),

        'agent':agent_path,
        'memory':os.path.join(agent_path,'memory'),
        'planner':os.path.join(agent_path,'planner'),
        'provider':os.path.join(agent_path,'provider'),
        'conf':os.path.join(agent_path,'conf'),
        'model':os.path.join(agent_path,'provider\\model'),

    }

    #add to environment variable
    for path in project_dir.values():
        if path not in sys.path:
            sys.path.insert(0,str(path))
    