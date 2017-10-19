import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pyaudio
import wave
import numpy as np 
import featuresExtract as extractor

"""
 This fucntion is for audio extraction.
 The function record an on-going music and saves it as a wav file
 The funtion was adapted from a sample I found online
 """
def recordSound():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 30
	WAVE_OUTPUT_FILENAME = "currentTest/currentFile.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)

	print("recording ...")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

"""
 This functions takes and array of class probability distribution
 and returns the class with the higher probability
 """
def getGenre(a):
	if a == 0:
		return "Heavy Metal"
	elif a == 1:
		return "Afro-Cuban"
	else:
		return "Malian Blues"
	return ans


if __name__ == '__main__':
	# recordSound()
	features = extractor.parsefiles('currentTest', 
		None, None, None, None, 0, 0)
	print np.shape(features[2])
	# graph = tf.get_default_graph()
	# with graph.as_default():
	# 	saver = tf.train.Saver()  # Gets all variables in `graph`.
	# with tf.Session(graph = graph) as sess:
	# 	saver = tf.train.Saver()
	# 	saver.restore(sess, 'myTensor2.ckpt')
	# 	session.run(y_pred, feed_dict={x: input_data})
	sess = tf.Session()
	saver = tf.train.import_meta_graph('my-model-10000.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	sess.run(tf.global_variables_initializer())
	graph = tf.get_default_graph()
	w_o = graph.get_tensor_by_name("w_o:0")
	b_o = graph.get_tensor_by_name("b_o:0")
	Y = graph.get_tensor_by_name("y.:0")
	# b_o = graph.get_tensor_by_name("b_o:0")
	X = graph.get_tensor_by_name("x.:0")
	feed_dict= {X: features[2]}
	y_o = graph.get_tensor_by_name("y_o:0")
	y = sess.run(y_o, feed_dict)
	print y
	print getGenre(sess.run(tf.argmax(y,1))[0])
	# with tf.Graph().as_default() as graph:
	# 	dummy = tf.Variable(0)
	# 	init_op = tf.global_variables_initializer()
	# 	with tf.Session() as sess:
	# 		saver = tf.train.import_meta_graph('my-model-10000.meta')
	# 		saver.restore(sess, tf.train.latest_checkpoint('./'))
	# 		sess.run(init_op)
	# 		w_o = graph.get_tensor_by_name("w_o:0")
	# 		b_o = graph.get_tensor_by_name("b_o:0")
	# 		y_h = graph.get_tensor_by_name("y_h:0")
	# 		X_l = tf.placeholder(tf.float32,[None,197])
	# 		feed_dict= {X_l: features[2]}
	# 		y_o = tf.nn.softmax(tf.matmul(y_h,w_o) + b_o)
	# 		y_o = sess.run(y_h, feed_dict)
			# print sess.run(output_layer, feed_dict={X_l: features[2]})
	#saver = tf.train.import_meta_graph('myTensor1.meta')
	# saver.restore(sess,tf.train.latest_checkpoint('./'))
	# graph = tf.get_default_graph()
	# output_layer = graph.get_tensor_by_name("y_:0")
	# predict = tf.argmax(output_layer, 1)
	# with sess.as_default():
	# 	pred = predict.eval({X: features[2].reshape(-1, 196)})
	# 	print pred
	# sess.run(tf.global_variables_initializer())
	# print output_layer
	# print sess.run(predict, feed_dict = {X:features[2]})

	# new_outputs=2
	# weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
	# biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
	# output = tf.matmul(fc7, weights) + biases
	# pred = tf.nn.softmax(output)

