import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

gpu = torch.cuda.is_available()

print('\t\tLoading T0pp Tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")

print('\t\tLoading T0pp Model...')
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
if gpu:
    device_map = {0: [0,  1],
                  1: [2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12],
                  2: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
    try:
        model.parallelize(device_map)
        print('\t\tModel moved to GPU(s)')
    except:
        print('\t\tError parallelizing T0pp. Model is loaded on CPU')


def inference(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if gpu:
        inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=20)
    inf_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    del inputs
    del outputs
    torch.cuda.empty_cache()
    return inf_result.strip()
