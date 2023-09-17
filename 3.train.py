import dvc.api 

from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType 
# from huggingface_hub import login
# login()

params = dvc.api.params_show()
print(params)

exit()
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"
model_checkpoint = params["MODEL"]
batch_size = params["TrainConfig"]["BATCH_SIZE"]


def preprocess_function(examples):
    return tokenizer(examples["RawText"], truncation=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2),

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = load_dataset("json", data_file= "./data/sentiment_train_data/datasets.json")


tf_dataset = dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'label'],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=params["TrainConfig"]["LEARNING_RATE"],
    per_device_train_batch_size=params["TrainConfig"]["PER_DEVICE_TRAIN_BATCH_SIZE"],
    per_device_eval_batch_size=params["TrainConfig"]["PER_DEVICE_EVAL_BATCH_SIZE"],
    num_train_epochs=params["TrainConfig"]["EPOCHS"],
    weight_decay=params["TrainConfig"]["WEIGHT_DECAY"],
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
model.save_pretrained(params["Train"]["MODEL"])