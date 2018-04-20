from __future__ import print_function

import tensorflow as tf
# import iris_data
import numpy as np
import pandas as pd
rng = np.random
# import types

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
batch_size = 5
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def My_load_data():
    (train_x, train_y), (test_x, test_y) = load_data()
    train_y = tf.one_hot(train_y, depth=3, dtype=tf.float32, name='train_y_divide')
    test_y = tf.one_hot(test_y, depth=3, dtype=tf.float32, name='test_y_divide')
    # print(test_y)
    return (train_x, train_y), (test_x, test_y)


def My_dense(input_rlt, dense_node_num=10, relu_activity=True, softmax_activity=False):
        in_node_array_num = int(input_rlt.get_shape()[1])
        # in_node_col_num = batch_size 
        # in_node_col_num = int(input_rlt.get_shape()[0])
        w = tf.Variable(tf.random_normal(shape=[in_node_array_num, dense_node_num], mean=0.1, stddev=0.5))
        b = tf.Variable(tf.random_normal(shape=[dense_node_num], mean=0.1, stddev=0.1))

        output_rlt = tf.matmul(input_rlt, w) + b
        # print(output_rlt.shape[0], " and ", output_rlt.shape[1])

        if relu_activity:
            output_rlt = tf.nn.relu(output_rlt)
        if softmax_activity:
            output_rlt = tf.nn.softmax(output_rlt)

        return output_rlt


'''
def transfer_train_y(raw_data):
    data = np.zeros(raw_data.shape[0] * 3)
    data.shape = (raw_data.shape[0],3)
    for i in range(0, raw_data.shape[0]):
        index = int(raw_data[i])
        data[i][index]=1
    return data
'''


def test_true(test_y, pred_test):
    pred_test = pred_test.tolist()
    test_y = test_y.tolist()
    '''
    likest = np.where(pred_test == max(pred_test))
    ans = np.where(test_y == max(test_y))
    '''
    likest = pred_test.index(max(pred_test))
    ans = test_y.index(max(test_y))
    if ans == likest:
        return True
    else:
        return False


def main(argv):

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = My_load_data()
    train_num = train_x.shape[0]
    # tf Graph Input
    X = tf.placeholder(tf.float32, shape=[None,4], name='X') #  
    Y = tf.placeholder(tf.float32, shape=[None,3], name='Y') #  
    # Set model weights
    # W = tf.Variable([[rng.randn()],[rng.randn()],[rng.randn()],[rng.randn()]], name="weight")
    # b = tf.Variable(rng.randn(), name="bias")
    # print(tf.random_normal(shape=tf.Variable([10,None]),mean = 2 ,stddev = 0.1))
    Hide_layer1 = My_dense(X, 10)
    Hide_layer2 = My_dense(Hide_layer1, 10)
    Hide_layer3 = My_dense(Hide_layer2, 10)
    pred = My_dense(Hide_layer3, 3, relu_activity=False, softmax_activity=False) #softmax 正则化 在交叉熵里实现了
    pred_show = tf.nn.l2_normalize(pred)
    '''
    w = tf.Variable(tf.random_normal(shape=[10,None],mean = 1 ,stddev = 0.1))
    b = tf.Variable(tf.random_normal(shape=[10,1],mean = 1 ,stddev = 0.1))
    print (w,b)
    Hide_layer1 = tf.layers.dense(X, 10,tf.nn.relu,kernel_initializer = w,bias_initializer = b)# 只接受一维
    Hide_layer2 = tf.layers.dense(Hide_layer1, 10, tf.nn.relu,kernel_initializer = w,bias_initializer = b)
    Hide_layer3 = tf.layers.dense(Hide_layer2, 10, tf.nn.relu,kernel_initializer = w,bias_initializer = b) 
    pred = tf.layers.dense(Hide_layer3,1)
    '''
    # Construct a linear model
    # pred = tf.add(tf.multiply(X, W), b)

    # Mean squared error
    # cost = tf.reduce_sum(tf.pow(pred-Y, 2))
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred) # 交叉熵
    cost = tf.reduce_mean(cost) # 取平均
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        train_y = train_y.eval()
        test_y = test_y.eval()
        # Fit all training data
        for epoch in range(training_epochs):
            for i in range(0,train_x.shape[0]-batch_size,batch_size): # batch_size 可去，就会很慢
                tx=train_x[i:i+batch_size]
                # print(tx)
                ty=train_y[i:i+batch_size]
                # print(ty)
                sess.run(optimizer,feed_dict={X:tx,Y:ty})
            # Display logs per epoch step           
            if (epoch+1) % display_step == 0:
                '''
                cost_all = 0.0
                for j in range(0,train_x.shape[0]): # batch_size 可去
                    tx=test_x[i:i+1] 
                    ty=test_y[i:i+1]
                    cost_now = sess.run(cost, feed_dict={X: tx, Y:ty})
                    cost_all += cost_all
                cost_all /= train_num
                '''    
                cost_all = sess.run(cost, feed_dict={X: train_x, Y:train_y})
                print("Epoch:", '%04d' % (epoch+1), "cost_for_now=", "{:.9f}".format(cost_all))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
        print("Training cost=", training_cost, '\n')
        # test it
        correct = 0
        cost_test_all = 0.0
        for i in range(0,test_x.shape[0]): # batch_size 可去
            tx=test_x[i:i+1]
            # print(tx)
            ty=test_y[i:i+1]
            # print(ty)
            pred_test,cost_test=sess.run([pred_show,cost],feed_dict={X:tx,Y:ty})
            # pred_test = tf.nn.softmax(pred_test)
            cost_test_all += cost_test
            # print("my_ans:",pred_test,"ans:",ty)
            if test_true(ty, pred_test):
                correct += 1
        cost_test_all /= test_y.shape[0]
        print("Correct_rate = ", '%.4f' %(correct/test_x.shape[0]), "Test_cost=", "{:.9f}".format(cost_test_all))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)