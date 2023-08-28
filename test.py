#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import multiprocessing
import os
import os.path
import sys
import warnings
from collections import Counter

import jieba
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

# 忽略警告
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#全局变量，用来读入数据
DOCUMENTS = list()

#用来保存路径的类
class DataConfig:
    # 词汇表路径
    vocab_path = "./vocab/vocab.txt"
    # 停用词词典路径
    stopwords_path = "./vocab/stopwords.txt"
    # 自定义词典路径
    dict_path = "./vocab/dict.txt"
    # 数据集路径
    dataset_path = "./news_data"
    # 分词后文件保存路径
    cut_words_path = "./vocab/cut_words.txt"
    # 词汇表大小
    vocab_size = 4000
    # 待分类文本的最大长度
    max_length = 200

#构建词汇表
def build_vocab():
    """根据数据集构建词汇表"""
    """注意高频词汇，有时携带的信息量反而很少"""
    all_data = []
    for content in DOCUMENTS:
        all_data.extend(content)

    # 选出出现频率最高的前vocab_size个字
    counter = Counter(all_data)
    count_pairs = counter.most_common(DataConfig.vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 作为填充字符
    words = ['<PAD>'] + list(words)
    # 保存词汇表
    open(DataConfig.vocab_path, mode='w', encoding='utf-8').write('\n'.join(words) + '\n')

#读文件
def read_file(dir_path):
    """
    加载数据集
    :param dir_path:
    :return:
    """
    global DOCUMENTS
    # 列出当前目录下的所有子目录
    dir_list = os.listdir(dir_path)
    # 遍历所有子目录
    for sub_dir in dir_list:
        # 组合得到子目录的路径
        child_dir = os.path.join('%s/%s' % (dir_path, sub_dir))
        if os.path.isfile(child_dir):
            # 获取当前目录下的数据文件
            with open(child_dir, 'r', encoding='utf-8') as file:
                document = ''
                lines = file.readlines()
                for line in lines:
                    # 将文件内容组成一行
                    document += line.strip()
                #目录名和处理后的文本内容以制表符分隔的形式，添加到全局变量 DOCUMENTS 中
            DOCUMENTS.append(dir_path[dir_path.rfind('/')+1:] + "\t" + document)
        else:
            read_file(child_dir)

def load_data(dir_path):
    """
    加载数据集，预处理
    :param dir_path:
    :return:
    """
    global DOCUMENTS
    data_x = []
    data_y = []

    # 读取所有数据文件
    read_file(dir_path)

    # 读取词汇表，词汇表不存在时重新构建
    if not os.path.exists(DataConfig.vocab_path):
        build_vocab()

    with open(DataConfig.vocab_path, 'r', encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    # 构建类标
    categories = ['其它', '时政', '财经']
    cat_to_id = dict(zip(categories, range(len(categories))))

    # contents, labels = read_file(data_path)
    for document in DOCUMENTS:
        y_, x_ = document.split("\t", 1)
        data_x.append([word_to_id[x] for x in x_ if x in word_to_id])
        data_y.append(cat_to_id[y_])

    # 将文本pad为固定长度
    data_x = tf.keras.preprocessing.sequence.pad_sequences(data_x, DataConfig.max_length)
    # 将标签转换为one-hot表示
    data_y = tf.keras.utils.to_categorical(data_y, num_classes=len(cat_to_id))

    return data_x, data_y




# 1.中文分词
def split_sentence():
    """
    训练数据分词处理
    :return:
    """
    global DOCUMENTS

    # 加载停用词
    stopwords = [line.strip() for line in open(DataConfig.stopwords_path, 'r', encoding='utf-8').readlines()]
    # 加载自定义词典
    jieba.load_userdict(DataConfig.dict_path)

    # 读取所有数据文件
#    read_file(DataConfig.dataset_path)
    # 并行分词,win不支持，移除
#    jieba.enable_parallel(4)
    # 分词后数据的文件保存路径
    f_out = open(DataConfig.cut_words_path, 'w', encoding='utf-8')
    count = 0
    length = len(DOCUMENTS)
    for document in DOCUMENTS:
        y_, x_ = document.split("\t", 1)
        word_list = list(jieba.cut(x_))
        # 去停用词
        out_str = ''
        for word in word_list:
            if word not in stopwords:
                out_str += word
                out_str += ' '
        f_out.write(out_str + '\n')
        count += 1
        print(str(count) + ' of ' + str(length))
    f_out.close()

#训练词向量
def train_w2v(model_type):
    program = os.path.basename(sys.argv[0])  # 读取当前文件的文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    data_path = './vocab/cut_words.txt'
    w2v_model = './vocab/w2v.model'
    w2v_vector = './vocab/w2v.vector'

    # 训练skip-gram模型,旧版用的size=300，新版vector_size=300
    model = Word2Vec(LineSentence(data_path), sg=0, vector_size=300, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(w2v_model)
    # 保存词向量
    model.wv.save_word2vec_format(w2v_vector, binary=False)
    print("Train Finished")





#创建RNN模型
def get_rnn_model(model_type):
    """
    常用的RNN模型
    :param model_type:
    :return:
    """
    model = tf.keras.Sequential()

    # 嵌入层
    # model.add(tf.keras.layers.Embedding(DataConfig.vocab_size, 300, input_length=200))

    # 嵌入层，加载预训练词向量
    w2v_Model = word2vec.Word2Vec.load("./vocab/w2v.model")
    # 加载词汇表
    vocab_list = w2v_Model.wv.index_to_key
    # 词典
    word_index = {" ": 0}
    word_vector = {}

    # 初始化矩阵，首行padding补零。
    # 行数为所有单词数+1，列数为词向量维度
    embeddings_matrix = np.zeros((len(vocab_list) + 1, w2v_Model.vector_size))
    # 填充字典和矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = w2v_Model.wv[word]
        embeddings_matrix[i + 1] = w2v_Model.wv[word]

    model.add(tf.keras.layers.Embedding(len(embeddings_matrix),
                                        300,
                                        weights=[embeddings_matrix],
                                        input_length=200,
                                        trainable=False
                                        ))

    if model_type == 'bi-lstm':
        # 使用LSTM的双向循环神经网络
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
    elif model_type == 'lstm':
        # 使用LSTM的单向循环神经网络
        model.add(tf.keras.layers.LSTM(16))
    elif model_type == 'gru':
        # 单向循环神经网络
        model.add(tf.keras.layers.GRU(16))
    else:
        pass
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_textcnn_model():
    """
    Text-CNN 模型
    :param model_type:
    :return:
    """
    model = tf.keras.Sequential()
    # 嵌入层
    # model.add(tf.keras.layers.Embedding(DataConfig.vocab_size, 300, input_length=200))

    # 嵌入层，加载预训练词向量
    w2v_Model = word2vec.Word2Vec.load("./vocab/w2v.model")
    # 加载词汇表
    vocab_list = w2v_Model.wv.index_to_key
    # 词典
    word_index = {" ": 0}
    word_vector = {}

    # 初始化矩阵，首行padding补零。
    # 行数为所有单词数+1，列数为词向量维度
    embeddings_matrix = np.zeros((len(vocab_list) + 1, w2v_Model.vector_size))
    # 填充字典和矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = w2v_Model.wv[word]
        embeddings_matrix[i + 1] = w2v_Model.wv[word]

    model.add(tf.keras.layers.Embedding(len(embeddings_matrix),
                                        300,
                                        weights=[embeddings_matrix],
                                        input_length=200,
                                        trainable=False
                                        ))

    # TextCNN
    model.add(tf.keras.layers.Conv1D(256, 5, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3, 3, padding='same'))
    model.add(tf.keras.layers.Conv1D(128, 5, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3, 3, padding='same'))
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.BatchNormalization())  # (批)规范化层
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # 数据集路径
    data_path = "./news_data"

    # 加载数据集
    train_x, train_y = load_data(data_path)

    # 分词
#    split_sentence()

    #训练词向量
#    train_w2v(0)


    # 随机打乱数据集顺序
    np.random.seed(116)
    np.random.shuffle(train_x)
    np.random.seed(116)
    np.random.shuffle(train_y)

    # 划分训练集和验证集
    x_val = train_x[:10000]
    partial_x_train = train_x[10000:]
    y_val = train_y[:10000]
    partial_y_train = train_y[10000:]

    model = get_textcnn_model()
#    model = get_rnn_model('bi-lstm')
    model.fit(partial_x_train, partial_y_train,
              epochs=20, batch_size=512,
              validation_data=(x_val, y_val))
    print(model.evaluate(x_val,  y_val, verbose=2))

    # 保存训练好的模型
    model.save('./saved_models/model.h5')













