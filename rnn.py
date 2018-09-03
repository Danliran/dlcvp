import tensorflow as tf
import numpy as np

int2binary = {}
binary_dim = 8
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]


def binary_generation(numbers, reverse=False):
    binary_x = np.array([int2binary[num] for num in numbers], dtype=np.uint8)
    if reverse:
        binary_x = np.fliplr(binary_x)

    return binary_x


def batch_generation(batch_size, largest_number):
    n1 = np.random.randint(0, largest_number // 2, batch_size)
    n2 = np.random.randint(0, largest_number // 2, batch_size)
    add = n1 + n2

    binary_n1 = binary_generation(n1, True)
    binary_n2 = binary_generation(n2, True)
#    print(binary_n1)
#    print(binary_n2)

    batch_y = binary_generation(add, True)
    batch_x = np.dstack((binary_n1, binary_n2))
#    print(batch_x)
    return batch_x, batch_y, n1, n2, add


def binary2int(binary_array):
    out = 0
    for index, x in enumerate(reversed(binary_array)):
        out +=x * pow(2, index)

    return out


#batch_x, batch_y, n1, n2, add = batch_generation(3, 256)


batch_size =64
lstm_size = 20
lstm_layers =2

x = tf.placeholder(tf.float32, [None, binary_dim, 2], name="input_x")

y_ = tf.placeholder(tf.float32, [None, binary_dim], name="input_y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

#lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
#drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)


cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
print(outputs)
weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
bias = tf.zeros([1])

outputs = tf.reshape(outputs, [-1, lstm_size])
logits = tf.sigmoid(tf.matmul(outputs, weights) + bias)
predictions = tf.reshape(logits, [-1, binary_dim])
cost = tf.losses.mean_squared_error(y_, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)

steps = 2000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    iteration = 1
    for i in range(steps):
        input_x, input_y, _, _, _ = batch_generation(batch_size, largest_number)
        _, loss = sess.run([optimizer, cost], feed_dict={x: input_x, y_: input_y})

        if iteration % 100 == 0:
            print("Iter:{}, loss:{}".format(iteration, loss))
        iteration +=1
