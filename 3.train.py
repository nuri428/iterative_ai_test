import torch
import dvc.api 
# from mlem.api import save

from dvclive.huggingface import DVCLiveCallback
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, Features, Value, ClassLabel
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType 
import numpy as np 


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

params = dvc.api.params_show()
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

# id2label = {-1: "NEGATIVE", 0:"NEUTRAL", 1: "POSITIVE"}
# label2id = {"NEGATIVE": -1, "NEUTRAL":0, "POSITIVE": 1}
id2label = {0:'neg',1:'neu',2:'pos'}
label2id = {'neg':0,'neu':1,'pos':2}

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3) #id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

ClassLabels = ClassLabel(num_classes=3, names=['neg','neu','pos'])

emotion_features = Features({'id':Value(dtype='int32'),
                             'sentence': Value(dtype='string'), 
                             'label': ClassLabels})

dataset = load_dataset("json", 
                       features=emotion_features,
                       data_files= "./data/sentiment_train_data/datasets.json", )
dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.cast_column('label', ClassLabels)

# Mapping Labels to IDs
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset.set_format(type="torch", columns=['attention_mask', 'input_ids', 'label'])

# print(len(dataset["train"].features["label"]))
print(dataset)
print(dataset["train"].features["label"])
 
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=params["TrainConfig"]["LEARNING_RATE"],
    per_device_train_batch_size=params["TrainConfig"]["PER_DEVICE_TRAIN_BATCH_SIZE"],
    num_train_epochs=params["TrainConfig"]["EPOCHS"],
    weight_decay=params["TrainConfig"]["WEIGHT_DECAY"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    # eval_dataset=dataset['eval'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.add_callback(DVCLiveCallback(save_dvc_exp=True))
trainer.train()
model.save_pretrained(params["TRAINED_MODEL"])
# model.save()
# save(model, f"models/{params['TRAINED_MODEL']}")