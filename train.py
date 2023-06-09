import os
from itertools import chain
from transformers import BertTokenizerFast
from datasets import load_dataset
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer

print('[INFO] Loading dataset...')

files = [
    "dataset/train1.txt",
    "dataset/train2.txt",
    "dataset/train3.txt",
    "dataset/train4.txt",
    "dataset/train5.txt",
    "dataset/train6.txt",
    "dataset/train7.txt",
    "dataset/train8.txt",
    "dataset/train9.txt",
    "dataset/train10.txt",
    "dataset/train11.txt",
    "dataset/train12.txt"
]

d = load_dataset("text", data_files={'train': files, 'test': ['dataset/test.txt']})

print(d['train'])

special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

vocab_size = 87
max_length = 140
truncate_longer_samples = False

model_path = "smiles-bert-char-tokenizer"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

print('[INFO] Loading tokenizer from pretrained...')
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_without_truncation(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)


print('[INFO] Encoding train dataset...')
train_dataset = d["train"].map(encode_without_truncation, batched=True)
print('[INFO] Encoding test dataset...')
test_dataset = d["test"].map(encode_without_truncation, batched=True)

test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


print(f'[INFO] Grouping train dataset in chunks of {max_length}...')
train_dataset = train_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping texts in chunks of {max_length}")
print(f'[INFO] Grouping test dataset in chunks of {max_length}...')
test_dataset = test_dataset.map(group_texts, batched=True,
                                desc=f"Grouping texts in chunks of {max_length}")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

print(f'[INFO] Init model and trainer configs...')

model_config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length,
    num_hidden_layers=6,
    num_attention_heads=4,
    hidden_size=512,
    intermediate_size=1024
)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,
    logging_steps=1000,
    save_steps=1000,
    # load_best_model_at_end=True,
    # save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print(f'[INFO] Starting fit process...')
trainer.train()
