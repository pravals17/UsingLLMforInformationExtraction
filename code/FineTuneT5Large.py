import torch
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

#parameters
model = 't5-large'
batch_size = 2
num_procs = 16
epochs = 20
out_dir = './results_t5large20epochs/'
max_length = 512 # Maximum context length to consider while preparing dataset.


#loading the datasets
dataset_train = load_dataset(
    'csv',
    data_files='./train.csv',
    split='train'
)
dataset_valid = load_dataset(
    'csv',
    data_files='./validation.csv',
    split='train'
)

tokenizer = T5Tokenizer.from_pretrained(model)

# Function to convert text data into model inputs and targets
class Preprocess:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def preprocess_function(self, examples):
        #tokenizer = T5Tokenizer.from_pretrained(model)
        inputs = [f"assign tags: {title} {content}" for (title, content) in zip(examples['Title'], examples['Content'])]
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        # Set up the tokenizer for targets
        #cleaned_tag = [' '.join(''.join(tag.split('<')).split('>')[:-1]) for what in examples['What']]
        event5w_tags = [f"Where: {where}; When:{when}; What:{what}; Who:{who}; Why:{why} " for (where,when,what,who,why) in 
                       zip(examples['Where'],examples['When'],examples['What'],examples['Who'],examples['Why'])]
        #print(cleaned_tag)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                event5w_tags,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# Apply the function to the whole dataset
#tokenizer = T5Tokenizer.from_pretrained(model)
preprocess = Preprocess(tokenizer)
tokenized_train = dataset_train.map(
    preprocess.preprocess_function,
    batched=True,
    num_proc=num_procs
)
tokenized_valid = dataset_valid.map(
    preprocess.preprocess_function,
    batched=True,
    num_proc=num_procs
)

model = T5ForConditionalGeneration.from_pretrained(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('The device used for training')
print(device)
model.to(device)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir=out_dir,
    logging_steps=10,
    evaluation_strategy='steps',
    save_steps=250,
    eval_steps=250,
    load_best_model_at_end=True,
    save_total_limit=5,
    report_to='tensorboard',
    learning_rate=0.0001,
    fp16=True,
    dataloader_num_workers=1
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

#Train the model
history = trainer.train()

#save the tokenizer
tokenizer.save_pretrained(out_dir)