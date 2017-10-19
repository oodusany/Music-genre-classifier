import scipy
import numpy as np
import librosa #library for audio features extraction
import os #for iterating through me training file
import librosa.display
import random
import numpy as np 


"""
 This function extracts the features of the loaded file.
 The inputs are y and sr which are the time series and sampling
 rate respectively.

 Returns an array containing all the features
 """
def extract_features(y, sr):
	S = np.abs(librosa.stft(y))
	mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
	zcrossRate = np.mean(librosa.feature.zero_crossing_rate(y=y).T,axis=0)
	rmse = np.mean(librosa.feature.rmse(y=y).T,axis=0)
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
	chroma = np.mean(librosa.feature.chroma_stft(S=S, sr=sr).T,axis=0)
	# spec_centroid = np.mean(librosa.feature.spectral_centroid(S=S).T,axis = 0)
	mel_spec = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
	spec_contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T,axis=0)
	return [mfccs,chroma,mel_spec,spec_contrast,tonnetz, zcrossRate, rmse, [tempo]]

"""
 This function is to convert the given genre names to numbers
 In this case we have 3 genres and they are Heavy Metal, Afro-Cuban 
 and Malian Blues indexed 1, 2, and 3 respectively
 """
def getLabel(filename):
	if filename.startswith("H"):
		ans = 1
	elif filename.startswith("A"):
		ans = 2
	elif filename.startswith("M"):
		ans = 3 
	else:
		# print "ERROR"
		ans = -1
	return ans

"""
 This function, given a filename randomizes the order of the music files,
 splits it into test and train sections and extracts the music features
 of the files and dumps into the respective filepaths if store is one
 """
def parsefiles(file, feat1, lab1, featt, labt, split, store):
	print 'Extracting Features ...'
	files = os.listdir(file)
	random.shuffle(files)
	#196 is the length of the features array. len(extract_features) is 196
	features, labels = np.empty((0,196)), np.empty(0, dtype = int)
	featuresTest, labelsTest = np.empty((0,196)), np.empty(0 , dtype = int)
	i = 0
	for filename in files[:int(len(files)*split)]:
		print i
		print filename
		try:
			if filename.endswith(".wav"):
				y,sr = librosa.load(file + '/' + filename)
				label = getLabel(filename)
				feat = np.hstack(extract_features(y,sr))
				features = np.vstack([features,feat])
				labels = np.append(labels, label)
		except:
			print filename, "pass"
		i = i + 1
	print np.shape(features), type(features), features.dtype
	for filename in files[int(len(files)*split):]:
		print i
		print filename
		try:
			if filename.endswith(".wav"):
				y,sr = librosa.load(file + '/' + filename)
				label = getLabel(filename)
				feat = np.hstack(extract_features(y,sr))
				featuresTest = np.vstack([featuresTest,feat])
				labelsTest = np.append(labelsTest, label)
		except:
			print filename, "pass"
		i = i +1
	features, labels = np.array(features), np.array(labels,  dtype = np.int)
	featuresTest, labelsTest = np.array(featuresTest), np.array(labelsTest,  dtype = np.int)
	if store:
		np.save(feat1,features)
		np.save(featt, featuresTest)
		np.save(lab1, labels)
		np.save(labt, labelsTest)
	return features, labels, featuresTest, labelsTest

if __name__ == '__main__':
	features, labels, featuresTest, labelsTest = parsefiles('training2', 
		"trainFeatures.npy", "labels.npy", "testFeatures.npy", "testLabels.npy", 0.80, 1)











