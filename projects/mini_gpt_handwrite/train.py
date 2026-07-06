from inference import llm_memory_calculator
import jieba
import torch
import torch.nn as nn
import gpt_model as gpt
from torch.utils.data import DataLoader



class Tokenizer():
    def __init__(self, text):
        self.text = text
        self.vocab = self.build_vocab(text)
        self.vocab_size = len(self.vocab)

    def encoder(self):
        text_list = jieba.lcut(self.text)
        return [self.vocab.index(token) for token in text_list]

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

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def train():
    #语料加载
    text = load_text('train.txt')
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    #数据加载
    dataset = TextDataSet(text, 128)
    #dataLoader
    batch_size = 16
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #模型加载
    model = gpt.mini_gpt(vocab_size,768,3,10,128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #构建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #训练
    epochs = 100
    for i in range(epochs):
        print(f'Epoch {i+1}/{epochs}')
        #获取数据
        for x, y in dataLoader:
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
        print(f'Loss: {loss.item()}')
    model_config = {
        'vocab_size': vocab_size,
        'embed_size': 768,
        'num_heads': 3,
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
    }, 'model.pth')
    print("模型已保存")

def model_load():
    checkpoint = torch.load('model.pth')
    model_config = checkpoint['model_config']
    model = gpt.mini_gpt(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    print("模型已加载")
    return model, vocab


def evaluate():
    # 从训练好的模型中恢复词表
    model, vocab = model_load()
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


def predict():
    model, vocab = model_load()
    # 后续用 vocab 构造 tokenizer 进行推理


if __name__ == '__main__':
    #训练
    train()
    #评估
    evaluate()
    #预测
    predict()
