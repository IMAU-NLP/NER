from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

"""获取句子序列，用于预测结果"""
def get_sentenses(dataset):
        with open(dataset, "r") as t:
            sentenses = []
            sentense = []
            for line in t.readlines():
                if line.strip() == "":
                    sentenses.append(sentense)
                    sentense = []
                else:
                    sentense.append(line.strip().split('\t')[0])
            return sentenses

"""通过句子序列预测结果"""
def get_predict_values(sentenses):
    config = Config()
    
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    
    predict_values = []
    for sentense in sentenses:
        #此处sentense为一个句子列表
        result = model.predict(sentense)
        predict_values.append(result)
    return predict_values

"""将预测结果写入文件"""
def write_file(filename,sentenses,predict_values):
    with open(filename, "w") as f:
        for sentense,predict_value in zip(sentenses,predict_values):
            for line in zip(sentense,predict_value):
                f.write('\t'.join(line) + "\n")
            f.write("\n")

if __name__ == "__main__":
    dataset = "./data/example.test"
    # ~ dataset = "111.txt"
    sentenses = get_sentenses(dataset)
    
    predict_values = get_predict_values(sentenses)
    write_file("result.txt", sentenses, predict_values)
        
