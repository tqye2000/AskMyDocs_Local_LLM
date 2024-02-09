#============================================================
#
#============================================================
#from torch import bfloat16
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
from transformers import pipeline


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    #torch_dtype=bfloat16,
    torch_dtype=torch.float32,
    device_map='auto'
)
#model.eval()

generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.2
generation_config.do_sample = True
generation_config.top_p = 0.15
generation_config.top_k = 0

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generate_text = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer,
)

# generate_text = pipeline(
    # model=model, tokenizer=tokenizer,
    # return_full_text=False,  # Set this to True if using langchain
    # task="text-generation",
    # temperature=0.1,  # Controls the randomness of outputs
    # top_p=0.15,  # Probability threshold for selecting tokens
    # top_k=0,  # Number of top tokens to consider (0 relies on top_p)
    # max_new_tokens=512,  # Limits the number of generated tokens
    # repetition_penalty=1.1  # Discourages repetitive outputs
# )

test_prompt = "[INST]The future of AI is [/INST]"
result = generate_text(test_prompt)
print(result[0]['generated_text'])
