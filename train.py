# train_classification.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Chinese_tokenizer import ChineseTransformer
from TextClassification import TaxtClassification
import numpy as np
import os

# 自定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        text_encoded = self.tokenizer.encode(text)[:self.max_len]
        
        # 填充序列
        if len(text_encoded) < self.max_len:
            text_encoded = text_encoded + [self.tokenizer.tokenizer.vocab['<pad>']] * (self.max_len - len(text_encoded))
        else:
            text_encoded = text_encoded[:self.max_len]
        
        return torch.tensor(text_encoded), torch.tensor(label)

# 加载数据函数
def load_classification_data(file_path):
    texts = []
    labels = []
    label_map = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label_text = parts[0]
                text = parts[1]
                
                # 创建标签映射
                if label_text not in label_map:
                    label_map[label_text] = len(label_map)
                
                texts.append(text)
                labels.append(label_map[label_text])
    
    return texts, labels, label_map

# 训练函数
def train_classification_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    num_classes = 10  # 根据您的数据调整
    max_len = 256
    dropout = 0.1
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10
    
    # 初始化分词器
    chinese_tokenizer = ChineseTransformer(vocab_size, d_model, max_len, dropout)
    
    # 加载数据
    train_texts, train_labels, label_map = load_classification_data('cnews/cnews.train.txt')
    test_texts, test_labels, _ = load_classification_data('cnews/cnews.test.txt')
    val_texts, val_labels, _ = load_classification_data('cnews/cnews.val.txt')
    
    # 训练分词器
    all_texts = train_texts + test_texts + val_texts
    chinese_tokenizer.train_tokenizer(all_texts)
    
    # 创建数据集和数据加载器
    train_dataset = ClassificationDataset(train_texts, train_labels, chinese_tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ClassificationDataset(val_texts, val_labels, chinese_tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = TaxtClassification(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=max_len,
        dropout=dropout,
        device=device
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                acc = 100. * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_map': label_map,
                'tokenizer': chinese_tokenizer,
                'max_len': max_len
            }, f'classification_model_epoch_{epoch+1}.pth')
    
    print("训练完成!")
    
    # 测试模型
    test_model(model, chinese_tokenizer, test_texts, test_labels, device, max_len, label_map)

# 测试模型
def test_model(model, tokenizer, test_texts, test_labels, device, max_len, label_map):
    model.eval()
    
    # 反转标签映射
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # 选择几个样本进行测试
    num_samples = 5
    indices = np.random.choice(len(test_texts), num_samples, replace=False)
    
    correct = 0
    total = 0
    
    for i in indices:
        text = test_texts[i]
        true_label = test_labels[i]
        
        # 编码输入
        text_encoded = tokenizer.encode(text)[:max_len]
        if len(text_encoded) < max_len:
            text_encoded = text_encoded + [tokenizer.tokenizer.vocab['<pad>']] * (max_len - len(text_encoded))
        else:
            text_encoded = text_encoded[:max_len]
        
        input_tensor = torch.tensor(text_encoded).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = output.max(1)
            predicted_label = predicted.item()
        
        # 统计
        total += 1
        if predicted_label == true_label:
            correct += 1
        
        print(f"文本: {text[:50]}...")
        print(f"真实类别: {idx_to_label[true_label]}")
        print(f"预测类别: {idx_to_label[predicted_label]}")
        print("-" * 50)
    
    acc = 100. * correct / total
    print(f"测试准确率: {acc:.2f}%")

# 使用模型进行预测
def predict_text(text, model_path='classification_model_epoch_10.pth'):
    """使用训练好的模型进行文本分类"""
    # 加载模型和配置
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    label_map = checkpoint['label_map']
    tokenizer = checkpoint['tokenizer']
    max_len = checkpoint['max_len']
    
    # 反转标签映射
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # 初始化模型
    device = torch.device('cpu')
    model = TextClassificationTransformer(
        vocab_size=len(tokenizer.tokenizer.vocab),
        d_model=512,
        num_heads=8,
        num_layers=6,
        num_classes=len(label_map),
        max_len=max_len,
        dropout=0.1,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 编码输入文本
    text_encoded = tokenizer.encode(text)[:max_len]
    if len(text_encoded) < max_len:
        text_encoded = text_encoded + [tokenizer.tokenizer.vocab['<pad>']] * (max_len - len(text_encoded))
    else:
        text_encoded = text_encoded[:max_len]
    
    input_tensor = torch.tensor(text_encoded).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        predicted_label = predicted.item()
    
    return idx_to_label[predicted_label]

if __name__ == '__main__':
    # 训练模型
    train_classification_model()
    
    # 测试预测
    test_text = "这是一篇关于体育的新闻，讲述了最新的足球比赛结果。"
    prediction = predict_text(test_text)
    print(f"文本: {test_text}")
    print(f"预测类别: {prediction}")