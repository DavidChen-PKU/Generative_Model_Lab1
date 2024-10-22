# -*- coding: utf-8 -*-
import json
from transformers import AutoModelForCausalLM, AutoConfig, BertTokenizer  # Version: 4.40.0
from datasets import Dataset # Version: 3.0.1
from trl import SFTConfig, SFTTrainer # Version: 0.11.4
import torch # Version: 2.1.1
# Python Version: 3.10.11

# load training set
def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)
dataset = load_custom_dataset("train.json")

# load test set
def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['text'] for item in data] 
texts = load_test_dataset("test.json")
print(f"Loaded {len(texts)} texts for perplexity calculation")

# Perplexity calculation function
def calculate_perplexity(texts, model_test):
    model_test.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}  # 移到 GPU
            labels = inputs['input_ids']


            outputs = model_test(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += labels.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# load tokenizer
config = AutoConfig.from_pretrained("/home/chenlingjun/model/download/models--vietgpt--bert-30M-cased/snapshots/7e6394243a272af23f7934498965ed7737e1b70f")
tokenizer = BertTokenizer(vocab_file='vocab.txt')
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'cls_token': '[BOS]',
    'sep_token': '[EOS]',
    'mask_token': '[MASK]',
    'unk_token': '[UNK]',
    'bos_token': '[BOS]',
    'eos_token': '[EOS]'
})

# load model
model = AutoModelForCausalLM.from_config(config)

print(torch.cuda.is_available())
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

model.to(device)

# test before training
perplexity_pre = calculate_perplexity(texts, model)
print(f"Perplexity before training: {perplexity_pre}")

# set training config
sft_config = SFTConfig(
    dataset_text_field="text", 
    max_seq_length=512,
    output_dir="/tmp",
)

# load SFTTrainer
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
    tokenizer=tokenizer  
)

# training
trainer.train()

# test after training
perplexity_post = calculate_perplexity(texts, model)
print(f"Perplexity after training: {perplexity_post}")