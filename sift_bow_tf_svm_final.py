import cv2
import numpy as np
from mnist import MNIST
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import NearestNeighbors

start_time = time.time()

k1 = 5
k2 = 6

mndata = MNIST('./MNIST/')
# Train
(X_train, labels_train) = mndata.load_training()
X_train = np.array(X_train, dtype='uint8')
labels_train = np.array(labels_train)
X_train_k1 = X_train[labels_train == k1, :]
X_train_k2 = X_train[labels_train == k2, :]
X_train = np.concatenate((X_train_k1, X_train_k2), axis=0)
labels_train = np.concatenate(([k1] * X_train_k1.shape[0], [k2] * X_train_k2.shape[0]), axis=0)


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

descriptors = np.array(descriptors)
# descriptors = descriptors / np.max(descriptors)

# Tao tu dien
K = 100
kmeans = KMeans(n_clusters=K).fit(descriptors)
bow = kmeans.cluster_centers_.tolist()

# Dua anh ve vector theo bow
X_train_final = []
neigh = NearestNeighbors(5).fit(bow)
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
X_test_k1 = X_test[labels_test == k1, :]
X_test_k2 = X_test[labels_test == k2, :]
X_test = np.concatenate((X_test_k1, X_test_k2), axis=0)
labels_test = np.concatenate(([k1] * X_test_k1.shape[0], [k2] * X_test_k2.shape[0]), axis=0)
labels_test = np.array([1 if label == k1 else -1 for label in labels_test])

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

# Train 
import tensorflow as tf

labels_train_final = np.array([1 if y == k1 else -1 for y in labels_train_final])

x_data = tf.placeholder(shape=[None, K], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
print(x_data, y_target)

W = tf.Variable(tf.random_normal(shape=[K, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

model_output = tf.add(tf.matmul(x_data, W), b)

l2_norm = tf.norm(W)
hinge = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = hinge + l2_norm

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

labels_train_final = labels_train_final.reshape(labels_train_final.shape[0], 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, loss_ = sess.run([optimizer, loss], feed_dict={x_data: X_train_final, y_target: labels_train_final})

    W_out, b_out = sess.run([W, b])
    out_test = sess.run(model_output, feed_dict={x_data: X_test_final})
    labels_predict = np.array([1 if out > 0 else -1 for out in out_test])

    count = 0.0
    for i in range(labels_predict.shape[0]):
        if labels_predict[i] != labels_test_final[i]:
            count += 1.0
    print(count, '===', labels_predict.shape[0])
    print(100.0 - 100 * count / labels_predict.shape[0])

print('W = ', W_out.T, '\n=======\n', 'b = ', b_out)

end_train = time.time()

print(end_train - start_time)

f = open('./W_b.txt', 'w+')
f.write('%s \n' % W_out)
f.write('%s \n' % b_out)





