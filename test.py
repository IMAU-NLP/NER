from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

def align_data(data):
    #这个应该就是个对齐函数，让你的实体和你的类别对齐

    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model
    #这个函数就是一个交互式识别实体的一个函数
    #回头不想用交互式，比如要嵌套到网站里什么的，就在这个基础上改
     #主要关注这个函数就行
    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")
    #这就是个提示信息，没啥用

    while True:
        try:
            # for python 2
            sentence = input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        #这就是句子切分，这块写的简单，直接拿空格切分，把句子切成一个列表
        #breeding. 会出现这种情况，单词和句号切不开，不过思路就是这样，你要有好的切分方法，可以替换这块。
        words_raw = sentence.strip().split(" ")

        #这个也没啥用，就是输入exit退出交互式窗口
        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw) #这块就直接调用了model的预测函数，输入就是你切好的列表，返回结果
        to_print = align_data({"input": words_raw, "output": preds}) #这就是上面的函数，对你的结果进行了下对齐

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model 这几行，就是调用一下你训练好的模型
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    #这块就是把model传递给上面那个函数，给你一个交互式的命令行。
    interactive_shell(model)


if __name__ == "__main__":
    main()