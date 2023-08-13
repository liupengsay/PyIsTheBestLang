import unittest
import jieba
import unittest
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


"""
算法：短文本相似度计算
功能：

"""


import numpy as np
from collections import Counter


class BM25Model(object):
    """
    传统方法BM25解决短文本相似度问题
    https://zhuanlan.zhihu.com/p/113224707
    """
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


class BertSimilarity:
    def __init__(self):
        # 加载BERT模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = AutoModel.from_pretrained("bert-base-chinese")
        return

    # 定义计算相似度的函数
    def calc_similarity(self, s1, s2):
        # 对句子进行分词，并添加特殊标记
        inputs = self.tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True)

        # 将输入传递给BERT模型，并获取输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # 计算余弦相似度，并返回结果
        sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return sim


class TestGeneral(unittest.TestCase):

    @staticmethod
    def test_bm25():
        # 1. 使用文档进行分词
        document_list_text = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                         "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                         "我在微信上被骗了，请问被骗多少钱才可以立案？",
                         "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                         "有人走私两万元，怎么处置他？",
                         "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]

        document_list = [list(jieba.cut(doc)) for doc in document_list_text]
        # 2. 计算文档分词的权重
        bm25_model = BM25Model(document_list)
        print(bm25_model.documents_list)
        print(bm25_model.documents_number)
        print(bm25_model.avg_documents_len)
        print(bm25_model.f)
        print(bm25_model.idf)
        # 3. 计算分数值越大越好
        query = "走私了两万元，在法律上应该怎么量刑？"
        query = list(jieba.cut(query))
        scores = bm25_model.get_documents_score(query)
        print(scores)
        print(document_list_text[scores.index(max(scores))])
        return

    @staticmethod
    def test_jieba_stop_words():
        stop_words = {"这是", "，", "、", "。"}
        sentence = "这是一个测试句子，请删除停用词、虚词和标点符号。"
        words = jieba.cut(sentence)
        words = [word for word in words if word not in stop_words and word.strip()]
        print(words)
        print(words)
        return

    @staticmethod
    def test_bert_similarity():
        # 测试函数
        s1 = "文本相似度计算是自然语言处理中的一个重要问题"
        s2 = "自然语言处理中的一个重要问题是文本相似度计算"
        be = BertSimilarity()
        similarity = be.calc_similarity(s1, s2)
        print(f"相似度：{similarity:.4f}")
        document_list_text = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                         "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                         "我在微信上被骗了，请问被骗多少钱才可以立案？",
                         "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                         "有人走私两万元，怎么处置他？",
                         "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
        query = "走私了两万元，在法律上应该怎么量刑？"
        print(query)
        for doc in document_list_text:
            similarity = be.calc_similarity(query, doc)
            print(f"{doc} 相似度：{similarity:.4f}")
        return


if __name__ == '__main__':
    unittest.main()
