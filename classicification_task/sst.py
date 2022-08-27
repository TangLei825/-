# -------------------------------load dataset-------------------------------
from datasets import load_dataset
imdb = load_dataset('imdb')     # imdb数据集

# -------------------------------tokenize-------------------------------
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)   # 这里没有进行padding


# 官方推荐的预处理方法（记住格式即可），batched=True 会自动批量处理
tokenized_imdb = imdb.map(tokenize_function, batched=True)


from transformers import DataCollatorWithPadding    # 用于把dataset处理成dataloader（这里叫collator，收集器）

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------创建指标计算函数-------------------------
import numpy as np
from datasets import load_metric

def compute_metrics(eval_preds):
    """
    :param eval_preds:   包含两个元素 logits : [batch_size, class_num], labels : [class_num]
    :return:
    """
    metrics = load_metric('glue', 'mrpc')
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    return metrics.compute(predictions=preds, references=labels)

# -------------------------------创建模型、配置对应参数、训练器-------------------------------
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification(checkpoint)

training_args = TrainingArguments(
    output_dir='results',
    learning_rate = 2e-05,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb['train'],
    eval_dataset=tokenized_imdb['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# f1在0.934左右