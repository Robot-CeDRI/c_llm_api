import os
from settings import auto_config as cfg
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from datasets import Dataset

def serialize_dataset(messages: list) -> Dataset:
    message_list = []
    for m in messages:
        message_list.append("<|" + m.role + "|>" + m.content)
    print(message_list)
    return Dataset.from_list()

class LLM:
    def __init__(self):
        self.model_hf_name = cfg.HF_LLM_MODEL
        self.cache_dir = cfg.MODELS_DIR
        self.hf_token = cfg.HF_TOKEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_name, cache_dir=self.cache_dir)
        base_model = AutoModelForCausalLM.from_pretrained(self.model_hf_name,
                                                          torch_dtype=torch.float16,
                                                          device_map="auto",
                                                          trust_remote_code=True,
                                                          cache_dir=self.cache_dir)
        if cfg.FINE_TUNED_MODEL_PATH:
            peft_model = PeftModel.from_pretrained(base_model, cfg.FINE_TUNED_MODEL_PATH, from_transformers=True, device_map="auto")
            self.model = peft_model.merge_and_unload()
        else:
            self.model = base_model
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             device_map="auto",
                             torch_dtype=torch.float16)

    async def exec_inference(self, messages: list, response_num_tokens=50, do_sample=True, response_temperature=.7, top_k=50, top_p=.95):
        prompt_pipe = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt_pipe,
                       batch_size=4,
                       max_new_tokens=response_num_tokens,
                       do_sample=do_sample,
                       temperature=response_temperature,
                       top_k=top_k,
                       top_p=top_p)
        return outputs[0]["generated_text"]

if not os.path.exists(cfg.MODELS_DIR):
    os.makedirs(cfg.MODELS_DIR)

LLM_MODEL = LLM()