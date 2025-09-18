from position_encoding import transformer_embeding
from tokenizer import JiebaTokenizer
import torch



class ChineseTransformer:
    def __init__(self, vocab_size, d_model, max_len, dropout):
        # 使用基于字符的分词器或基于词的分词器
        self.tokenizer = JiebaTokenizer(vocab_size)
        # 或者使用JiebaTokenizer
        # self.tokenizer = JiebaTokenizer(vocab_size)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.TokenEmbedding = transformer_embeding(vocab_size, d_model, max_len, 'cpu', dropout)
        # 其他组件...
    
    def train_tokenizer(self, texts):
        """使用训练数据训练分词器"""
        self.tokenizer.fit(texts)
    
    def preprocess(self, texts):
        """预处理中文文本数据"""
        # 编码文本
        encoded = [self.tokenizer.encode(text) for text in texts]
        # 转换为张量
        return torch.tensor(encoded)
    
    def forward(self, texts):
        """前向传播"""
        input_ids = self.preprocess(texts)
        embedded = self.TokenEmbedding.forward(input_ids)
        output = embedded
        # 继续处理...
        return output