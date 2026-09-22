import os

# Must set cache paths BEFORE transformers/huggingface_hub is imported,
# because huggingface_hub reads HF_HOME once at import time.
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_HOME'] = os.getenv('HF_HOME', 'D:/hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', 'D:/hf_cache/hub')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model():
    model_name = os.getenv('MODEL_NAME', 'Salesforce/codegen-350M-mono')
    hf_token   = os.getenv('HF_TOKEN') or None
    use_gpu    = torch.cuda.is_available()

    print(f"Model : {model_name}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"Cache : {os.environ['HF_HOME']}")
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # codegen tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")

    if use_gpu:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto',
                token=hf_token,
            )
            print("Loaded in 4-bit quantized mode (GPU)")
        except Exception as e:
            print(f"4-bit load failed ({e}), trying float16...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                token=hf_token,
            )
            print("Loaded in float16 (GPU)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            token=hf_token,
        )
        model = model.to('cpu')
        print("Loaded in float32 (CPU)")

    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model()

    prompt = "// fix buffer overflow\nvoid foo() {"
    print(f"\nSmoke test prompt: {repr(prompt)}")

    inputs = tokenizer(prompt, return_tensors='pt').to('cpu')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated output:\n{generated}")
    print("\nSmoke test passed - model is ready.")
