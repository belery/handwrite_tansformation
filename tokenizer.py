import jieba
import jieba.posseg as pseg  # 用于词性标注
from collections import Counter

class JiebaTokenizer:
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        
    def fit(self, texts, use_hmm=True):
        """从中文文本语料构建词汇表"""
        words = Counter()
        
        for text in texts:
            # 使用jieba进行分词
            if use_hmm:
                tokens = jieba.cut(text, HMM=use_hmm)
            else:
                tokens = jieba.cut(text, HMM=False)
            words.update(tokens)
        
        # 选择最常见的词
        most_common = words.most_common(self.vocab_size - len(self.special_tokens))
        
        # 构建词汇表
        self.vocab = {**self.special_tokens}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        # 构建反向词汇表
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text, use_hmm=True):
        """将中文文本转换为token ID序列"""

        if use_hmm:
            tokens = jieba.cut(text, HMM=use_hmm)
        else:
            tokens = jieba.cut(text, HMM=False)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
    
    def decode(self, token_ids):
        """将token ID序列转换回中文文本"""
        tokens = [self.inverse_vocab.get(token_id, '<unk>') for token_id in token_ids]
        return ''.join(tokens)  # 中文不需要空格分隔
    