import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials import mnist
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
#one element is on or hot class


n_nodes_h1=500
n_nodes_h2=500
n_nodes_h3=500

n_class=10

batch=100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_net_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random.normal([784,n_nodes_h1])),
                    'biases':tf.Variable(tf.random.normal([n_nodes_h1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_h1, n_nodes_h2])),
                      'biases': tf.Variable(tf.random.normal([n_nodes_h2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_h2, n_nodes_h3])),
                      'biases': tf.Variable(tf.random.normal([n_nodes_h3]))}

    output_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_h3, n_class])),
                    'biases': tf.Variable(tf.random.normal([n_class]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    return output

def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer=tf.train.AdamOptimizer().minimize(cost)
    hm_epochs=20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(hm_epochs):
            loss=0
            for _ in range(int(mnist.train.num_examples/batch)):
                ep_x,ep_y=mnist.train.next_batch(batch)

                _,c =sess.run([optimizer,cost],feed_dict={x:ep_x,y:ep_y})
                loss += c
            print("epoch",ep,"completed out of",hm_epochs,"\n loss :",loss)
        corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        acc=tf.reduce_mean(tf.cast(corr,'float'))

        print("Accuracy : ",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
train_network(x)




