import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    train_test_split
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from optuna import trial, study
import optuna
from functools import partial

#还需要分割数据集、配置config//还需要更改templates，希望可以生成3个动作，并给出每个动作的probability//更改tempalte为，给出几个候选动作，从这几个候选动作中给出回答

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return self.args.custom_compute_loss(model, inputs, return_outputs)

class SFT_Trainer:
    def __init__(self,model_name="Qwen/Qwen2.5-VL-3B-Instruct",data_path=None):
        self.model_name = model_name
        self.data_path=data_path
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        self.special_tokens = ["<key>", "</key>", "<reasoning>","</reasoning>","<actions>", "</actions>"]
    
    def load_model_and_tokenizer(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
    def setup_lora(self,trial=None):

        if trial is not None:
            lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
            lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
            lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        else:
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.1
            learning_rate = 2e-4
            
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        return learning_rate
        
    def load_data(self,mode='train'):
        
        data = []
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data.append(item)
        
        return data
    
    def preprocess_data(self, examples,weight_factor=2.0):

        texts = []
        for i in range(len(examples)):
            instruction = examples[i].get("instruction", "")
            output = examples[i].get("output", "")
            
            if instruction and output:
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                texts.append(text)
            else:
                raise ValueError("Each example must contain 'instruction' and 'output' fields.")
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=2048,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()

        if "input_ids" in tokenized:
            tokenized["loss_weights"] = self.create_special_token_weights(
                torch.tensor(tokenized["input_ids"]), weight_factor
            )
        
        return tokenized
        
    
    def create_special_token_weights(self, input_ids, weight_factor=2.0):

        weights = torch.ones_like(input_ids, dtype=torch.float32)
        
        special_token_ids = []
        for token in self.special_tokens:
            if token in self.tokenizer.get_vocab():
                special_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
        
        if special_token_ids:
            for special_id in special_token_ids:
                special_positions = (input_ids == special_id)
                weights[special_positions] = weight_factor
        
        return weights
    
    def compute_weighted_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        
        labels = inputs.get("labels")
        loss_weights = inputs.get("loss_weights", torch.ones_like(labels, dtype=torch.float32))
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        
        weighted_loss = (loss * shift_weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    def objective(self, trial):

        self.load_model_and_tokenizer()
        
        learning_rate = self.setup_lora(trial)
        
        raw_data = self.load_data() #需要修改成load validation data
        
        dataset = Dataset.from_list(raw_data)
        tokenized_dataset = dataset.map(
            partial(self.preprocess_function, weight_factor=2.0),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        training_args = TrainingArguments(
            output_dir=f"./qwen_sft_output_trial_{trial.number}",
            per_device_train_batch_size=trial.suggest_categorical("batch_size", [1, 2, 4]),
            gradient_accumulation_steps=trial.suggest_categorical("grad_accum", [2, 4, 8]),
            learning_rate=learning_rate,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=20,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            warmup_steps=trial.suggest_int("warmup_steps", 50, 200),
            fp16=False,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        training_args.custom_compute_loss = self.compute_weighted_loss
        
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        train_result = trainer.train()

        return train_result.metrics["train_loss"]
    
    def hyperparameter_tuning(self, n_trials=10):

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler()
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_trial.params
    
    def train_for_output_template(self):

        #load model
        self.load_model_and_tokenizer()

        #setup lora
        self.setup_lora()
        
        #load data
        raw_data = self.load_data()
        
        #prepare dataset
        dataset = Dataset.from_list(raw_data)
        tokenized_dataset = dataset.map(
            partial(self.preprocess_function, weight_factor=2.0),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        #configurate training args
        training_args = TrainingArguments(
            output_dir="./qwen_sft_output",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            warmup_steps=100,
            fp16=False,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None  # 禁用wandb等记录器
        )
        
        #data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        training_args.custom_compute_loss = self.compute_weighted_loss

        #configurate trainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        #train
        trainer.train()
        
        #save
        trainer.save_model()
        self.tokenizer.save_pretrained("./qwen_sft_output")
    
    def train_for_output_with_best_params(self,best_params):

        self.load_model_and_tokenizer()
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=best_params.get("lora_r", 16),
            lora_alpha=best_params.get("lora_alpha", 32),
            lora_dropout=best_params.get("lora_dropout", 0.1),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        
        raw_data = self.load_jsonl_data()

        dataset = Dataset.from_list(raw_data)
        tokenized_dataset = dataset.map(
            partial(self.preprocess_function, weight_factor=2.0),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        training_args = TrainingArguments(
            output_dir="./qwen_sft_final",
            per_device_train_batch_size=best_params.get("batch_size", 2),
            gradient_accumulation_steps=best_params.get("grad_accum", 4),
            learning_rate=best_params.get("learning_rate", 2e-4),
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            warmup_steps=best_params.get("warmup_steps", 100),
            fp16=False,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        training_args.custom_compute_loss = self.compute_weighted_loss
        
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained("./qwen_sft_final")

    
    
