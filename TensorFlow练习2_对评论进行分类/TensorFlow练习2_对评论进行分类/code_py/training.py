import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# load lexicon
with open('lexcion.pickle', 'rb') as f:
    lex = pickle.load(f)
lex_dict = {v: i for i, v in enumerate(lex)}


def get_test_feature(filename="test.csv"):
    df = pd.read_csv(filename, encoding="utf-8", header=0, sep="|")
    y_test = df[["negative", "neutral", "positive"]].as_matrix()
    lemmatizer = WordNetLemmatizer()
    X_test = []

    def pipeline_line(x):
        words = word_tokenize(x.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex_dict[word]] = 1
        X_test.append(features)

    list(map(pipeline_line, df.text))
    X_test = np.array(X_test)
    return y_test, X_test


test_y, test_x = get_test_feature("test.csv")


def get_batch_feature(filename="train.csv", batch_size=1000):
    df = pd.read_csv(filename, encoding="utf-8", header=0, sep="|")
    df = df.sample(batch_size)
    y_test = df[["negative", "neutral", "positive"]].as_matrix()
    lemmatizer = WordNetLemmatizer()
    X_test = []

    def pipeline_line(x):
        words = word_tokenize(x.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex_dict[word]] = 1
        X_test.append(features)

    list(map(pipeline_line, df.text))
    X_test = np.array(X_test)
    return y_test, X_test


train_y, train_x = get_batch_feature(filename="train.csv")


## we may join the two functions above into one

def neural_network(data):
    n_input_layer = len(lex)
    n_layer_1 = 200  # hide layer neurons
    n_layer_2 = 200  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层
    n_output_layer = 3  # 输出层
    # 定义第一层"神经元"的权重和biases,   matrix W &  vector b
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


def train_neural_network():
    X = tf.placeholder('float')
    Y = tf.placeholder('float')
    batch_size = 200
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    # metrics
    n_correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))  # bool vector
    accuracy = tf.reduce_mean(tf.cast(n_correct, 'float'))
    accuracy_tmp = 0.3

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        # if model.ckpt文件已存在:saver.restore(session, 'model.ckpt')  恢复保存的session
        loop, bigloop = 0, 0
        while bigloop < 100:  # dead loop
            loop += 1
            if not loop % 4:
                print(bigloop, loop)
            batch_y, batch_x = get_batch_feature(batch_size=batch_size)
            try:
                session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})
            except Exception as e:
                print(e)
                continue  # break
            if loop > 10:
                # accuracy = accuracy.eval({X: test_x, Y: test_y})
                accuracy_j = session.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                accuracy_i = session.run(accuracy, feed_dict={X: test_x, Y: test_y})
                print('accuracy_on_valid: ', accuracy_i, 'accuracy_on_train: ', accuracy_j)
                if accuracy_i > accuracy_tmp:  # 保存准确率最高的训练模型,
                    accuracy_tmp = accuracy_i
                    saver.save(session, 'model.ckpt')  # 保存session
                bigloop += 1
                loop = 0


train_neural_network()
