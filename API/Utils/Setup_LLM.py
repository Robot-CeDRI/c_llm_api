import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from settings import auto_config as cfg


class LLM:
    def __init__(self):
        self.model_hf_name = cfg.HF_LLM_MODEL
        self.cache_dir = cfg.MODELS_DIR

        self.has_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.has_cuda else "cpu"

        # Perf flags (Dev only typically)
        if self.has_cuda and getattr(cfg, "LLM_USE_TF32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_hf_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )

        # Choose dtype: float16 on GPU, float32 on CPU (safer)
        torch_dtype = torch.float16 if self.has_cuda else torch.float32

        # Optional attention implementation (only makes sense on GPU + supported cards)
        attn_impl = getattr(cfg, "LLM_ATTN_IMPL", None)
        model_kwargs = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if self.has_cuda:
            model_kwargs["device_map"] = "auto"
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
        else:
            # CPU: don't use device_map auto/offload; keep it simple
            model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(self.model_hf_name, **model_kwargs)

        # Build pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.has_cuda else None,
            torch_dtype=torch_dtype,
        )

        # API default batch size (keep 1)
        self.default_batch_size = int(getattr(cfg, "LLM_BATCH_SIZE", 1))

    async def exec_inference(
        self,
        messages: list,
        response_num_tokens: int = 80,
        do_sample: bool = True,
        response_temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        # Safety: temperature must be > 0 when sampling
        if do_sample:
            if response_temperature is None or response_temperature <= 0:
                response_temperature = 0.7
        else:
            # Greedy decoding -> ignore sampling params
            response_temperature = None

        prompt_pipe = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        bs = int(getattr(cfg, "LLM_BATCH_SIZE", 1))

        if not do_sample:
            outputs = self.pipe(
                prompt_pipe,
                batch_size=bs,
                max_new_tokens=int(response_num_tokens),
                do_sample=False,
            )
            return outputs[0]["generated_text"]

        # Sampling: temperature tem de ser > 0
        if response_temperature is None or float(response_temperature) <= 0:
            response_temperature = 0.7

        outputs = self.pipe(
            prompt_pipe,
            batch_size=bs,
            max_new_tokens=int(response_num_tokens),
            do_sample=True,
            temperature=float(response_temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
        return outputs[0]["generated_text"]


if not os.path.exists(cfg.MODELS_DIR):
    os.makedirs(cfg.MODELS_DIR)

LLM_MODEL = LLM()
