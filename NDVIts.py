

"""
Code adapted from Chenting Hao's github repo https://github.com/chentinghao/time-series-prediction to predict sine
and cosine waves, which is in turn based on Guillaume Chevalier's work
https://github.com/guillaume-chevalier/seq2seq-signal-prediction

This uses a seq2seq Recurrent Neural Network with a Gated Recurrent Unit to predict a time series of NDVI values given
in the dataset convertcsv.csv. The values are taken every 16 days.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
from DataPrep import pre_data
from Smooth import smooth

tf.set_random_seed(seed=2018)
np.random.seed(seed=2018)


# parameters
batch_size = 32
seq_len = 212  # sequence length
n_hidd_layers = 3  # number of hidden layers # (can be 3)
n_neurons = 8
lr = 0.01  # learning rate
n_iters = 1200  # number of iterations
lambda_l2_reg = 0.001  # L2 regularization of weights


# preparing the data
X_batch, Y_batch = pre_data(file_path="convertcsv.csv", batch_size=batch_size, seq_len=seq_len)
inp_dim = X_batch.shape[2]
out_dim = X_batch.shape[2]

# reset tensorflow graph
tf.reset_default_graph()

# model - Seq2Seq model
with tf.variable_scope('Seq2Seq'):
    # encoder input
    enc_inp = [tf.placeholder(tf.float32, [None, inp_dim], name='enc_inp_{}'.format(t))
               for t in range(seq_len)]  # shape: [seq_len, batch_size, inp_dim] where batchsize is None and inp_dim is 1

    # decoder expected output
    dec_exp_out = [tf.placeholder(tf.float32, [None, inp_dim], name='dec_exp_out_{}'.format(t))
                   for t in range(seq_len)]  # shape: [seq_len, batch_size, inp_dim]

    # give a "GO" token to the decoder input
    dec_inp = [tf.zeros_like(enc_inp[0], dtype=tf.float32, name='GO')] + enc_inp[:-1]
    # print(dec_inp)

    layers = []
    print('log: creating layers')
    for i in range(n_hidd_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            layers.append(tf.nn.rnn_cell.GRUCell(n_neurons))
    multi_rnn = tf.nn.rnn_cell.MultiRNNCell(layers)
    dec_out, dec_state = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, multi_rnn)

    # reshape decorder out
    W_out = tf.Variable(tf.random_normal([n_neurons, out_dim]))
    b_out = tf.Variable(tf.random_normal([out_dim]))
    out_scale_factor = tf.Variable(1.0, name='out_scale_factor')
    reshaped_dec_out = [out_scale_factor * (tf.matmul(i, W_out) + b_out) for i in dec_out]
    # shape: [seq_len, batch_size, out_dim]

# model - loss function
with tf.variable_scope('Loss'):
    # L2 loss
    out_loss = 0
    for exp_out, pred_out in zip(dec_exp_out, reshaped_dec_out):
        out_loss += tf.reduce_mean(tf.nn.l2_loss(exp_out - pred_out))

    # L2 regularization
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ('out_scale_factor' in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
    loss = out_loss + lambda_l2_reg * reg_loss

# model - Optimizer
with tf.variable_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

print('log: all parameters defined. Beginning training now...')
# print(tf.default_graph())
# initialize the variables
init = tf.global_variables_initializer()

# train
sess = tf.Session()
sess.run(init)
train_losses = []

print('log: Training now...')
for iteration in range(n_iters):
    feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_len)}
    feed_dict.update({dec_exp_out[t]: Y_batch[t] for t in range(seq_len)})
    loss_train, _ = sess.run([loss, train_op], feed_dict)
    train_losses.append(loss_train)
    if iteration % 100 == 0:
        print('Iter: {} Train Loss: {}'.format(iteration, loss_train))

loss_test = sess.run(loss, feed_dict)
print('Test Loss: {}'.format(loss_test))

# Plot loss over time:
plt.figure(figsize=(15, 5))
# plt.plot(train_losses, label='Train Loss')
plt.plot(np.log(train_losses), label='Train Loss')
plt.title("Training Errors over Time (logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='upper right')
plt.show()

# predict
n_pred = 5 # number of predictions
X_batch, Y_batch = pre_data(file_path="convertcsv.csv", batch_size=n_pred, seq_len=seq_len)
feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_len)}
output = sess.run(reshaped_dec_out, feed_dict)
output = np.array(output)

print(output)
plt.figure(figsize=(15, n_pred))
for j in range(out_dim):
    for i in range(n_pred):
        past = X_batch[:, i, j]
        exp = Y_batch[:, i, j]
        pred = output[:, i, j]
        label_past = 'Past (True) Values' if j == 0 else '_nolegend_'
        label_exp = 'Expected (True) Values' if j == 0 else '_nolegend_'
        label_pred = 'Predicted Values' if j == 0 else '_nolegend_'
        plt.plot(range(len(past)), past, 'o--b', label=label_past)
        plt.plot(range(len(past), len(past) + len(exp)), exp, 'x--b', label=label_exp)
        plt.plot(range(len(past), len(past) + len(pred)), pred, 'o--r', label=label_pred)
    plt.title('Predicted Values vs True Values')
    plt.legend(loc='lower right')
plt.show()














