import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np  
import pandas as pd 
import sys
import argparse
import sklearn.preprocessing
import featuresExtract as extractor
import musicClassifier as rec

"""
 This function uses tensorflow to create a trained model that 
 will then be used to classify.
 The function takes in a set of npy files that contain arrays of 
 extracted features. The funtion then creates and saves a model based on 
 these features
 Returns an array containing all the features
 """
def train(trainFeats, trainLabels, testFeats, testLabels):
	# loading saved features and labels for training and testing
	x_train, y_train = np.load(trainFeats), np.load(trainLabels)
	x_test, y_test = np.load(testFeats), np.load(testLabels)
	# this is to convert the labels to a one-hot format suitable for
	# tensorflow machine learning
	label_binarizer = sklearn.preprocessing.LabelBinarizer()
	label_binarizer.fit(range(1,4))
	y_train = np.array(label_binarizer.transform(y_train))
	y_test = np.array(label_binarizer.transform(y_test))

	# defining hyper-parameters and can be changed for experimenting
	training_epochs = 5000
	learning_rate = 0.000001
	hidden_units = 285 #number of nuerons in hidden layer (about features*1.5)

	# other parameters specific to features and labels
	n_features = 196
	n_classes = 3
	stddev = 1 / np.sqrt(n_features)
	
	# defining place holders for our variables, x is the input features and 
	# x is the labels returns
	x = tf.placeholder(tf.float32,shape = (None,n_features), name = 'x.')
	y = tf.placeholder(tf.float32,[None,n_classes] , name = 'y.')

	#hidden network layer
	w_h = tf.Variable(tf.random_normal([n_features,hidden_units], mean = 0, stddev=stddev),name= 'w_h')
	b_h = tf.Variable(tf.random_normal([hidden_units], mean = 0, stddev=stddev), name = 'b_h')
	y_h = tf.nn.sigmoid(tf.matmul(x,w_h) + b_h, name = 'y_h')

	#output layer
	w_o = tf.Variable(tf.random_normal([hidden_units,n_classes], mean = 0, stddev=stddev), name= 'w_o')
	b_o = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=stddev), name = 'b_o')
	y_o = tf.nn.softmax(tf.matmul(y_h,w_o) + b_o, name = 'y_o')

	init = tf.global_variables_initializer()

	cost_function = -tf.reduce_sum(y * tf.log(y_o))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

	correct_prediction = tf.equal(tf.argmax(y_o,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(init)

	for epoch in range(training_epochs):
		none,cost = sess.run([optimizer,cost_function],feed_dict={x:x_train,y:y_train})
		#I will like to see the costs, but I don't want to print it everytime but every 50th step
		if epoch % 50 == 0:
			print cost
	print('Accuracy: ',round(sess.run(accuracy, feed_dict={x: x_test, y: y_test}) , 4))

	#This portion records a sound classifies individually
	rec.recordSound()
	features = extractor.parsefiles('currentTest', 
		None, None, None, None, 0, 0)
	y = sess.run(y_o, feed_dict={x: features[2]})
	print "The disribution is: ", y
	print "The Genre is: ",  rec.getGenre(sess.run(tf.argmax(y,1))[0])

	#save session to disk
	saver.save(sess, 'my-model-10000')


if __name__ == '__main__':
	train("trainFeatures.npy", "labels.npy", "testFeatures.npy", "testLabels.npy")
	