import json
import re
from typing import Optional

class DataProcessor:
    
    def __init__(self,template_path,json_path):
        self.template_path=template_path
        self.json_path=json_path
        self.reasoning_pattern=r'<reasoning>.*?</reasoning>'
        self.action_pattern=r'<actions>.*?<actions>'
        self.exist_information_holder="<$exist_information$>"
        self.goal_holder="<$goal$>"

    def load_data(self):
        with open(self.template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data=data["training_data"]
        
        return template,data
    
    def process_data(self):
          
        template,data=self.load_data()
        feature_data=[]
        target_data=[]

        for i in range(len(data)):
            sub_template=template
            #construct feature data
            sub_template.replace(self.exist_information_holder,data[i]["exist_information"])
            sub_template.replace(self.goal_holder,data[i]["goal"])
            feature_data.append(sub_template)

            #construct target data
            sub_template=re.sub(self.reasoning_pattern,data[i]["reasoning"][0],sub_template)
            sub_template=re.sub(self.action_pattern,data[i]["actions"],sub_template)
            target_data.append(sub_template)
        
        return feature_data,target_data
    
    @staticmethod
    def gather_data(feature_data: Optional[list[list]],
                    target_data: Optional[list[list]],
                    save_path)->None:
        
        feature_results=[]
        target_results=[]
        results=[]

        for feature,target in zip(feature_data,target_data):
             feature_results+=feature
             target_results+=target
            
        for instruction,output in zip(feature_results,target_results):
            results.append({
                "instruction": instruction,
                "output": output
            })
        
        with open(save_path+r"\sft_model.jsonl", 'w', encoding='utf-8') as file:
            for item in results:
                json_line = json.dumps(item,indent=4,ensure_ascii=False)
                file.write(json_line + '\n')
        
def test():
    data_processor=DataProcessor(template_path=r"F:\project\StardewAgent\trainer\sft\action_planning_template.prompt",
                                  json_path=r"F:\project\StardewAgent\trainer\sft\action_planning_data.json")
    feature_data,target_data=data_processor.process_data()
    data_processor.gather_data([feature_data],[target_data],save_path=r"F:\project\StardewAgent\trainer\sft")

if __name__ == "__main__":
     test()



         
        



                
                

          