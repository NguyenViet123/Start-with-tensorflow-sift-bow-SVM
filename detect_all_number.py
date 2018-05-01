import cv2
import numpy as np
from mnist import MNIST
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import NearestNeighbors

start_time = time.time()

mndata = MNIST('./MNIST/')
# Train
(X_train, labels_train) = mndata.load_training()
X_train = np.array(X_train, dtype='uint8')
labels_train = np.array(labels_train)
# num = 16
X_train = X_train[:, :]
labels_train = labels_train[:]
descriptors = []
sift = cv2.xfeatures2d.SIFT_create()

des_keypoint_train_final = []
labels_train_final = []
for i in range(labels_train.shape[0]):
    pixels = X_train[i, :].reshape((28, 28))
    kp_mnist, des_mnist = sift.detectAndCompute(pixels, None)

    if des_mnist is not None:
        des_keypoint_train_final.append(des_mnist)
        labels_train_final.append(labels_train[i])
        des_mnist = des_mnist.tolist()
        for des in des_mnist:
            descriptors.append(des)

labels_train_final = np.array(labels_train_final)
descriptors = np.array(descriptors)

# Tao tu dien
K = 100
kmeans = KMeans(n_clusters=K).fit(descriptors)
bow = kmeans.cluster_centers_.tolist()

# Dua anh ve vector theo bow
X_train_final = []
neigh = NearestNeighbors(1).fit(bow)
for des_keypoint in des_keypoint_train_final:
    x = np.array([0] * K)
    for i in des_keypoint:
        dist, nearest_id = neigh.kneighbors([i], 1)
        x[nearest_id] += 1

    X_train_final.append(x)

X_train_final = np.array(X_train_final)

# Test_data
(X_test, labels_test) = mndata.load_testing()
X_test = np.array(X_test, dtype='uint8')
labels_test = np.array(labels_test)

des_keypoint_test_final = []
labels_test_final = []
for i in range(labels_test.shape[0]):
    pixels = X_test[i, :].reshape((28, 28))
    kp_mnist, des_mnist = sift.detectAndCompute(pixels, None)

    if des_mnist is not None:
        des_keypoint_test_final.append(des_mnist)
        labels_test_final.append(labels_test[i])

labels_test_final = np.array(labels_test_final)


X_test_final = []
for des_keypoint in des_keypoint_test_final:
    x = np.array([0] * K)
    for i in des_keypoint:
        dist, nearest_id = neigh.kneighbors([i], 1)
        x[nearest_id] += 1

    X_test_final.append(x)

X_test_final = np.array(X_test_final)

# X_train_final labels_train_final
#X_test_final labels_test_final

# Train 
import tensorflow as tf


x_data = tf.placeholder(shape=[None, K], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, ], dtype=tf.int32)
# print(x_data, y_target)

W = tf.Variable(tf.random_normal(shape=[K, 10]))
b = tf.Variable(tf.random_normal(shape=[1, 10]))

model_output = tf.add(tf.matmul(x_data, W), b)

l2_norm = tf.norm(W)
# print(np.array([tf.reduce_sum(tf.multiply(x_data, tf.gather(tf.transpose(W), y_target)), axis=1)]).T.shape)

hinge = tf.reduce_sum(tf.maximum(0., 1. - tf.reduce_sum(tf.multiply(x_data, tf.gather(tf.transpose(W), y_target)), axis=1, keepdims=True)
                                 + tf.matmul(x_data, W)))

loss = hinge + l2_norm

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# labels_train_final = labels_train_final.reshape(labels_train_final.shape[0], 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, loss_ = sess.run([optimizer, loss], feed_dict={x_data: X_train_final, y_target: labels_train_final})

    W_out, b_out = sess.run([W, b])
    out_test = sess.run(model_output, feed_dict={x_data: X_test_final})
    labels_predict = np.argmax(out_test, axis=1)

    count = 0.0
    for i in range(labels_predict.shape[0]):
        if labels_predict[i] != labels_test_final[i]:
            count += 1.0
    print(count, '===', labels_predict.shape[0])
    print('Ti le: ', 100.0 - count * 100.0 / labels_predict.shape[0] )

end_train = time.time()

print('Thoi gian chay: ', end_train - start_time)

print('==================================================================')
print('W = ', W_out.T, '\n=======\n', 'b = ', b_out)






