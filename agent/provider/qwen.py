import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from PIL import Image
from torch.distributions import Categorical

class QwenProvider:
    def __init__(self, model_name="Qwen/Qwen2.5-vl-3b", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.is_loaded = False

    def download_model(self):
        if not self.is_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
            self.is_loaded = True

    def generate_response(self, text_input, image_path=None, max_length=512):
        if not self.is_loaded:
            self.download_model()
        
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        
        # 处理图像输入
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.image_processor(
                images=image, 
                return_tensors="pt"
            ).to(self.device, torch.float16)
            inputs.update(image_inputs)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

    def sample_action(self, state_description, action_space, temperature=1.0):
        if not self.is_loaded:
            self.download_model()
        
        # 构建动作选择提示
        prompt = f"""当前环境状态: {state_description}
        可选动作: {", ".join(action_space)}
        请选择最合适的动作: <action>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 提取动作token位置的logits
        action_token = self.tokenizer.encode("<action>", add_special_tokens=False)[0]
        logits = outputs.logits[0, -1, [action_token]]
        
        # 创建概率分布并采样
        probs = torch.softmax(logits / temperature, dim=-1)
        dist = Categorical(probs)
        action_idx = dist.sample().item()
        
        # 确保索引在有效范围内
        action_idx = min(action_idx, len(action_space) - 1)
        return action_idx, action_space[action_idx]

    def get_action_probabilities(self, state_description, action_space):
        if not self.is_loaded:
            self.download_model()
        
        # 构建动作选择提示
        prompt = f"""当前环境状态: {state_description}
        可选动作: {", ".join(action_space)}
        请评估动作概率: <action>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 提取动作token位置的logits
        action_token = self.tokenizer.encode("<action>", add_special_tokens=False)[0]
        logits = outputs.logits[0, -1, [action_token]]
        
        return torch.softmax(logits, dim=-1)