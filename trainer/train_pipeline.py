from typing import Optional
import json

from agent.provider.qwen import QwenProvider

class TrainPipline:
    def __init__(self):
        self.actor: Optional[QwenProvider]=None
        self.config={}
    
    def set_config(self,path):
        with open(path,'r',encoding='utf-8') as file:
            raw=json.load(file)
            self.config=raw['actor']
    
    def train_step(self):
        pass


