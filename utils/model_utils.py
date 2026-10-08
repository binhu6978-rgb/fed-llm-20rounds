import os
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType

def load_model(args):
    # Honour the configured GPU instead of unconditionally selecting GPU 3.
    # This must happen before the first CUDA model allocation.
    requested_device = str(args.device)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
    elif visible_devices != requested_device:
        print(
            f"CUDA_VISIBLE_DEVICES={visible_devices} is already set; "
            f"leaving it in place instead of overriding --device {requested_device}."
        )
    model_name = getattr(args, 'model_path', '') or args.model
    if getattr(args, 'mcq', False):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='sdpa',
        )
        model.config.use_cache = False
        model.config.pad_token_id = model.config.eos_token_id
        return model
    
    if args.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16
        )
    elif args.task_type == 'CAUSAL_LM':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # auto / cuda:0 / cpu ...
            torch_dtype=torch.float16
        )
    return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(args, 'model_path', '') or args.model,
        local_files_only=bool(getattr(args, 'mcq', False)),
    )
    if getattr(args, 'mcq', False) and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_lora_config(args):
    if args.task_type == 'SEQ_CLS':
        lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],  # BERT applies "query", "value" instead of "q_proj"/"v_proj"
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    elif args.task_type == 'CAUSAL_LM':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
    return lora_config
