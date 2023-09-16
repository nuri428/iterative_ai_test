import dvc.api 

from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

params = dvc.api.params_show()

def preprocess_function(examples):
    return tokenizer(examples["RawText"], truncation=True)

model = AutoModelForSequenceClassification.from_pretrained("", num_labels=2),

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
params = dvc.api.params_show()
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = load_dataset("json", data_file= "./data/sentiment_train_data/train.json")
test_dataset = load_dataset("json", data_file= "./data/sentiment_train_data/test.json")

tf_train_dataset = train_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_dataset = test_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()