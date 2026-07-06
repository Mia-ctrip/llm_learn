import os
import jieba
import torch
import torch.nn as nn
import gpt_model as gpt
from torch.utils.data import DataLoader
from tqdm import tqdm
from prepare_data import prepare_zhwiki



class Tokenizer():
    def __init__(self, text):
        self.text = text
        self.vocab = self.build_vocab(text)
        self.vocab_size = len(self.vocab)

    def encoder(self):
        text_list = jieba.lcut(self.text)
        return [self.vocab.index(token) for token in text_list if token in self.vocab]

    def decoder(self, tokens):
        return ''.join([self.vocab[token] for token in tokens])

    def build_vocab(self, text):
        self.vocab = []
        text = text.split('\n')
        for line in text:
            word_list = jieba.lcut(line)
            self.vocab.extend(word_list)
        return sorted(list(set(self.vocab)))    


class TextDataSet(torch.utils.data.Dataset):
    def __init__(self, text, max_length):
        super(TextDataSet, self).__init__()
        self.tokenizer = Tokenizer(text)
        self.id_list = self.tokenizer.encoder()
        self.block_size = max_length
        self.id_token_list = torch.tensor(self.id_list, dtype=torch.long)

    def __len__(self):
        return len(self.id_list) - self.block_size

    def __getitem__(self, index):
        x = self.id_token_list[index : index + self.block_size]
        y = self.id_token_list[index + 1 : index + 1 + self.block_size] 
        return x, y

_DIR = os.path.dirname(os.path.abspath(__file__))

def load_text(file_path):
    with open(os.path.join(_DIR, file_path), 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_data(min_train_rows=5000, min_eval_rows=500):
    """检查语料是否足够，不够则自动下载"""
    train_path = os.path.join(_DIR, 'train.txt')
    eval_path = os.path.join(_DIR, 'eval.txt')

    need_download = False
    if not os.path.exists(train_path):
        print("⚠️  未找到 train.txt，准备下载语料...")
        need_download = True
    else:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_lines = len(f.readlines())
        if train_lines < min_train_rows:
            print(f"⚠️  train.txt 只有 {train_lines} 行（目标 ≥ {min_train_rows}），准备重新下载...")
            need_download = True

    if need_download:
        prepare_zhwiki(train_rows=min_train_rows, eval_rows=min_eval_rows)
    else:
        if not os.path.exists(eval_path):
            print("⚠️  未找到 eval.txt，准备下载...")
            prepare_zhwiki(train_rows=min_train_rows, eval_rows=min_eval_rows)
        else:
            print(f"✅ 语料已就绪: train.txt ({train_lines} 行)")


def train():
    # 自动准备语料（如果不够则下载）
    prepare_data()
    #语料加载
    text = load_text(os.path.join(_DIR, 'train.txt'))
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    #数据加载
    dataset = TextDataSet(text, 128)
    #dataLoader
    batch_size = 8
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #模型加载
    model = gpt.mini_gpt(vocab_size, 256, 4, 10, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #构建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #训练
    epochs = 20
    pbar_epochs = tqdm(range(epochs), desc='训练进度', unit='epoch')
    for i in pbar_epochs:
        pbar_batch = tqdm(dataLoader, desc=f'Epoch {i+1}/{epochs}', unit='batch', leave=False)
        epoch_loss = 0
        batch_count = 0
        #获取数据
        for x, y in pbar_batch:
            x, y = x.to(device), y.to(device)
            #前向传播
            logits = model(x)
            #计算损失
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            pbar_batch.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / batch_count
        pbar_epochs.set_postfix(avg_loss=f'{avg_loss:.4f}')
    model_config = {
        'vocab_size': vocab_size,
        'embed_size': 256,
        'num_heads': 4,
        'num_layers': 10,
        'max_length': 128,
    }
    model_save(model, tokenizer, model_config)    


def model_save(model, tokenizer, model_config):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'vocab_size': tokenizer.vocab_size,
        'model_config': model_config,
    }, os.path.join(_DIR, 'model.pth'))
    print("模型已保存")

def model_load():
    model_path = os.path.join(_DIR, 'model.pth')
    if not os.path.exists(model_path):
        print("⚠️  未找到 model.pth，请先运行 train() 训练模型")
        return None, None
    checkpoint = torch.load(model_path)
    model_config = checkpoint['model_config']
    model = gpt.mini_gpt(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    print("模型已加载")
    return model, vocab


def evaluate():
    # 从训练好的模型中恢复词表
    model, vocab = model_load()
    if model is None:
        return
    # 用恢复的词表构造 tokenizer（保证和训练时一致）
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = vocab
    tokenizer.vocab_size = len(vocab)
    text = load_text('eval.txt')
    tokenizer.text = text
    #数据加载
    dataset = TextDataSet(text, 128)
    dataset.tokenizer = tokenizer
    dataset.id_list = tokenizer.encoder()
    dataset.id_token_list = torch.tensor(dataset.id_list, dtype=torch.long)
    #dataLoader
    batch_size = 16
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            #前向传播
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataLoader)
    print(f'Loss: {total_loss/len(dataLoader)}')
    print(f'Average Loss: {avg_loss}')    


def predict(prompt="今天天气", max_new_tokens=100, max_length=128):
    model, vocab = model_load()
    if model is None:
        print("⚠️  无法进行预测，请先训练模型")
        return None
    # 用恢复的词表构造 tokenizer（保证和训练时一致）
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = vocab
    tokenizer.vocab_size = len(vocab)
    tokenizer.text = prompt
    # 编码 prompt
    id_list = tokenizer.encoder()
    input_tensor = torch.tensor([id_list], dtype=torch.long)  # (1, seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            # 截断到 max_length，防止超出位置编码范围
            input_tensor = input_tensor[:, -max_length:]
            # 前向传播
            logits = model(input_tensor.to(device))
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # (1, vocab_size)
            # 贪心策略：取概率最大的 token
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1, 1)
            # 拼接到序列末尾
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
    # 循环结束后解码输出
    text = tokenizer.decoder(input_tensor[0].tolist())
    print(text)
    return text


