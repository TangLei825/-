import pandas as pd

# ---------------------------读取数据---------------------------
train_df = pd.read_csv('data/train.csv', sep=',')
test_df = pd.read_csv('data/test.csv', sep=',')

# ---------------------------数据预处理（这里需要根据不同的数据进行不同的操作）---------------------------
train_df = train_df[~train_df['Topic(Label)'].isnull()]     # 除去没有label的样本
train_df['Topic(Label)'], lbl = pd.factorize(train_df['Topic(Label)'])

train_df['Title'] = train_df['Title'].apply(lambda x: x.strip())
train_df['Abstract'] = train_df['Abstract'].fillna('').apply(lambda x: x.strip())
train_df['text'] = train_df['Title'] + ' ' + train_df['Abstract']
train_df['text'] = train_df['text'].str.lower()

test_df['Title'] = test_df['Title'].apply(lambda x: x.strip())
test_df['Abstract'] = test_df['Abstract'].fillna('').apply(lambda x: x.strip())
test_df['text'] = test_df['Title'] + ' ' + test_df['Abstract']
test_df['text'] = test_df['text'].str.lower()
print('数据加载完毕...')

# --------------------------------------------词化--------------------------------------------
from transformers import AutoTokenizer

# 使用预训练模型 bert-base-uncased，模型内容详见https://huggingface.co/bert-base-uncased
checkpoint = 'bert-base-uncased'     # 可以通过改变checkpoint快速更换模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 注意：tokenizer的输入必须是list
train_encoding = tokenizer(train_df['text'].to_list()[:], truncation=True, padding=True, max_length=512)
test_encoding = tokenizer(test_df['text'].to_list()[:], truncation=True, padding=True, max_length=512)

# --------------------------------------------将数据转换成dataset格式（方便用于装进dataloader）--------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader

class XunFeiDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = XunFeiDataset(train_encoding, train_df['Topic(Label)'].to_list())
test_dataset = XunFeiDataset(test_encoding, [0] * len(test_df))

# 单个读取到批量读取，batch_size可以调整，越大所需算力越多
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# --------------------------------------------定义评价指标（在验证的时候使用）--------------------------------------------
import numpy as np

def flat_accuracy(preds, labels):
    """
    :param preds:    [batch_size, class_num]
    :param labels:   [batch_size]
    :return:         acc
    如果需要其他指标（如F1），需要自己再进行定义
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# --------------------------------------------创建模型--------------------------------------------
# download 440MB预训练模型 bert-base-uncased，模型内容详见https://huggingface.co/bert-base-uncased
from transformers import AutoModelForSequenceClassification, AdamW

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU
model.to(device)     # 将模型加载进GPU

# -----------------------------------------创建优化器-----------------------------------------
optim = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_loader) * 1


# --------------------------------------------模型训练与验证--------------------------------------------
# 训练函数
def train():    # 执行完一次为训练完一个epoch
    model.train()            # 训练模型， model返回的是loss和logits
    total_train_loss = 0     # 总的损失
    iter_num = 0             # 一个iter是一个batch
    total_iter = len(train_loader)      # 所有batch的个数就是迭代次数
    # 取出一个batch
    for batch in train_loader:
        # 前向传播
        optim.zero_grad()    # 梯度清零

        # 将数据存入GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # outputs输出是SequenceClassification定义的一种类型，就包含两个元素loss和logits
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]      # outputs : loss(tensor), logits(tensor)
        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # 梯度裁剪

        # 参数更新
        optim.step()
        # scheduler.step()

        iter_num += 1
        if(iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" %
                  (epoch, iter_num, loss.item(), iter_num/total_iter*100))

    print("Epoch: %d, Average training loss: %.4f" %
          (epoch, total_train_loss/len(train_loader)))

# 验证函数
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" %
          (total_eval_loss/len(test_dataloader)))
    print("-------------------------------")


# --------------------------------------------开始训练--------------------------------------------
nums_epoch = 3     # 训练次数
for epoch in range(nums_epoch):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    # validation()

model.save_pretrained("output/model/%s-%sepoch" % (checkpoint, nums_epoch))   # 保存模型


# 预测函数
def prediction():
    model.eval()      # 评估的时候，model返回的是logits和labels
    test_label = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            pred = model(input_ids, attention_mask).logits
            test_label += list(pred.argmax(1).data.cpu().numpy())
    return test_label


# --------------------------------------------预测--------------------------------------------
test_predict = prediction()
test_df['Topic(Label)'] = [lbl[x] for x in test_predict]             # 根据id，转换成对应的label
test_df[['Topic(Label)']].to_csv('bert_submit3.csv', index=None)     # 保存预测的数据