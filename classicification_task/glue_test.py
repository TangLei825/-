# --------------------------装载数据集---------------------------------
# https://github.com/huggingface/datasets
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")    # 判断两个句子意思是否相近（数据集是较小的了）

# ----------------------------词化-----------------------------
from transformers import AutoTokenizer
# 不是所有模型返回结果都一样的，得看你选择模型训练的时候人家咋设置的
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 使用官方提供的数据集的时候可以这样处理
# 官方推荐的预处理方法（记住格式即可），batched=True 会自动批量处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# ---------------------------------创建数据收集器（训练器的输入参数）--------------------------------
from transformers import DataCollatorWithPadding    # 用于把dataset处理成dataloader（这里叫collator，收集器）

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------------定义评估方法--------------------------------
from datasets import load_metric
import numpy as np

def compute_metrics(eval_preds):    # 计算指标的方法（可以记住格式）
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ---------------------------创建模型、配置模型参数、训练器---------------------------
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification(checkpoint)

training_args = TrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()















