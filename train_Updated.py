import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def userResponse(message):
	'''
	Prints message and waits for user to reply with yes or no. Returns user's response
	'''
	print(message)
	response = ""
	while(True):
		response = raw_input()
		if (response != "yes" and response != "no"):
			print("Invalid response. Write either yes or no:")
		else: break
	return response

response = userResponse("Do you want to run this on GPU (if available) (yes/no)?")
if response == "no":
	os.environ['CUDA_VISIBLE_DEVICES'] = ""

def load_XML_Images():
	'''
	Reads XML file and returns xml element images
	'''
	# Read XML file
	tree = ET.parse('training_with_face_landmarks.xml')
	root = tree.getroot()
	# Return images
	return root.find('images')

def initialize_Keypoints_DataFrame():
	'''
	Initializes the dataframe which will contain keypoints, sets column names
	'''
	# Define column names
	names = ['filePath']
	for i in range(70):
		# Add two columns for each keypoint, one for x and one for y
		names.append("%02d_x" % (i,))
		names.append("%02d_y" % (i,))
	return pd.DataFrame(columns=names)

def get_face_bounding_box(image):
	'''
	Get bounding box location from xml element image
	'''
	# Get bounding box for face
	box = image.find('box') # image here is xml element
	height = int(box.attrib["height"])
	left = int(box.attrib["left"])
	top = int(box.attrib["top"])
	width = int(box.attrib["width"])

	return box, left, top, height, width


def extract_Face(filePath,box,newShape=(128,128)):
	'''
	Reads an image, crops the face from it, resizes it and returns
	'''
	(left, top, height, width) = box
	# Crop the face and resize to desired size
	cvImg = cv2.imread(filePath)
	cvImg = cvImg[top:top+height,left:left+width] # Crop face
	oldShape = (cvImg.shape[1],cvImg.shape[0]) # Shape is (width,height)
	cvImg = cv2.resize(cvImg,newShape) # Resize cropped face to desired size
	return cvImg, oldShape

def save_checkpoint(images,keypoints):
	'''
	saves Pre-processed image checkpoints (Images as numpy array and Keypoints in CSV format)
	'''
	# Save processed images so far along with keypoints
	np.savez('Images.npz',images=images)
	keypoints.to_csv('keypoints.csv',index=False)

def load_checkpoint():
	'''
	Loads preprocessed images (numpy array) and their Keypoints from CSV file
	'''
	# load latest checkpoint of images and their keypoints
	images = np.load('Images.npz')['images']
	keypoints = pd.read_csv('keypoints.csv')
	return images, keypoints

def loadData():
	'''
	Reads XML file and images from datasets directory. Preprocesses them, saves them, and returns them
	'''

	# Check if checkpoint exists
	if os.path.exists('Images.npz'):
		response = userResponse("A checkpoint for pre-processed images exist, Would you like to load it? (yes/no)")
		if response == "yes":
			images, keypoints = load_checkpoint()
			print("Checkpoint loaded with",len(images),"images")
			return images, keypoints
	else:
		print("Checkpoint for pre-processed images does NOT exist")
	print("Starting pre-processing of Images. This will take quite some time")

	xmlImages = load_XML_Images()

	keypoints = initialize_Keypoints_DataFrame()

	imgCount = 0
	images = [] # This will store all images as numpy arrays
	for image in xmlImages.findall('image'):
		# Image path
		filePath = image.attrib['file']

		# Location of face in the image
		box, left, top, height, width = get_face_bounding_box(image)

		# Ignore if face is not completely in Image
		if height <= 0 or left <= 0 or top <= 0 or width <=0: continue

		newShape = (128,128)
		# Crop face and resize
		cvImg, oldShape = extract_Face(filePath,(left, top, height, width),newShape)
		
		# Add to the list
		images.append(cvImg)

		# Add keypoints of current image to dataframe 
		keypoints.loc[imgCount,'filePath'] = filePath
		for keypoint in box.findall('part'):
			keypoints.loc[imgCount,keypoint.attrib['name'] + '_x'] = (float(keypoint.attrib['x']) - left) * (float(newShape[0]) / oldShape[0])
			keypoints.loc[imgCount,keypoint.attrib['name'] + '_y'] = (float(keypoint.attrib['y']) - top) * (float(newShape[1]) / oldShape[1])
		imgCount += 1

		if imgCount % 100 == 0:
			print (imgCount, "images processed")
			if imgCount % 500 == 0:
				# Save progress
				save_checkpoint(images,keypoints)
				display = '''Checkpoint saved, You can press Ctrl+C and run the code again to skip directly to training with this much data.
Or wait to get more data. In case you have less GPU resources, either run with less data (~3000 would be fine),
or run without GPU (Reply no when asked in the beginning)\n'''
				print (display)
	
	save_checkpoint(images,keypoints)
	return images, keypoints

# Add helper functions
def weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convLayer(X,W,b):
	layer = conv2d(X,W)
	layer = tf.nn.bias_add(layer, b)
	layer = maxpool2d(layer)
	layer = tf.nn.relu(layer)
	return layer

def fcLayer(X,W,b,out_layer=False):
	layer = tf.matmul(X, W)
	layer = tf.nn.bias_add(layer, b)
	if not out_layer:
	    layer = tf.nn.relu(layer)	    
	return layer

def model(X):

	# Conv Layer 1
	W1 = weights("W1",[3,3,3,8])
	b1 = biases(8)
	layer1 = convLayer(X,W1,b1)

	# Conv Layer 2
	W2 = weights("W2",[2,2,8,16])
	b2 = biases(16)
	layer2 = convLayer(layer1,W2,b2)

	# Conv Layer 3
	W3 = weights("W3",[2,2,16,32])
	b3 = biases(32)
	layer3 = convLayer(layer2,W3,b3)

	layer3 = tf.reshape(layer3,[-1,8192])

	# Fully connected layer 1
	W4 = weights("W4",[8192,500])
	b4 = biases(500)
	layer4 = fcLayer(layer3,W4,b4)

	# Fully connected layer 2
	W5 = weights("W5",[500,500])
	b5 = biases(500)
	layer5 = fcLayer(layer4,W5,b5)

	layer5 = tf.layers.dropout(layer5,rate=0.2)

	# output layer
	W6 = weights("W6",[500,140])
	b6 = biases(140)
	y_pred = fcLayer(layer5,W6,b6,out_layer=True)

	return y_pred

def next_batch(trainX,trainY,batch_index,batch_size):
	start = (batch_index*batch_size) % len(trainX)
	end = start + batch_size
	if end > len(trainX):
		end = len(trainX)
	batch_x = trainX[start:end]
	batch_y = trainY[start:end]
	return batch_x, batch_y

def train(trainX, trainY, testX, testY, num_epochs = 200, batch_size = 128):

	global X, Y, cost, optimizer, std, learning_rate

	saver = tf.train.Saver()
	
	with tf.Session() as sess:

		# Initialize tensorflow variables
		sess.run(tf.global_variables_initializer())

		
		for epoch in range(num_epochs):
			# Train on batches
			for batch_index in range(len(trainX)/batch_size+1):
				batch_x, batch_y = next_batch(trainX, trainY,batch_index,batch_size)
				sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})	
			
			# Loss on Training examples after each epoch
			trainLoss = sess.run(cost, feed_dict={X: trainX,Y: trainY})
			print "Train ",epoch,") ",trainLoss * std
			
			if epoch % 10 == 0 and epoch > 0:
				# Loss on unseen examples every 10th epoch
				testLoss = sess.run(cost, feed_dict={X: testX,Y: testY})
				print "\nTest : ",testLoss * std,"\n"

			if epoch % 50 == 0 and epoch > 0:
				# reduce learning rate to 50% after every 50 iterations
				old_learning_rate = sess.run(learning_rate)
				sess.run(learning_rate.assign(0.7 * old_learning_rate))
				print "Learning rate reduced, new learning rate = ",old_learning_rate * 0.7

		if not os.path.exists('./tf_checkpoint'):
			os.makedirs('./tf_checkpoint')
		saver.save(sess, './tf_checkpoint/tf_model')		
		print("Tensorflow checkpoint saved in tf_checkpoint directory")

	print("Training Finished!")

def main():
	global X, Y, cost, optimizer, Y_pred, std, learning_rate
	
	images, keypoints = loadData()
	print("Data loaded successfully")

	mean = keypoints.iloc[:,1:].stack().mean()
	std = keypoints.iloc[:,1:].stack().std()
	
	keypoints.iloc[:,1:] = (keypoints.iloc[:,1:] - mean) / std

	print images.shape
	print keypoints.iloc[:,1:].shape # First column is file name

	trainX, testX, trainY, testY = train_test_split(images,keypoints.iloc[:,1:],test_size=0.2)

	
	# Make placeholders for X and Y, to be fed later
	X = tf.placeholder(tf.float32, shape=[None,128,128,3], name="X")
	Y = tf.placeholder(tf.float32, shape=[None,140], name="Y")
	learning_rate = tf.Variable(initial_value=0.001,trainable=False,dtype=tf.float32)

	Y_pred = model(X)
	Y_pred = tf.identity(Y_pred, name="Y_pred")
	# learning_rate = 0.0005
	cost = tf.reduce_mean(tf.square(Y - Y_pred), name="cost")
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name="optimizer")

	train(trainX,trainY, testX, testY)


if __name__ == "__main__":
	main()
