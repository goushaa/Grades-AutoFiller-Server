from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import cv2
import os

def train_shapes():
	main_relative_path = "./flask-server/module1/training/"
	relative_path= main_relative_path + "dataset/"

	images = []
	labels = []

	# get all the image folder paths
	image_paths = os.listdir(relative_path + "shapes")

	for path in image_paths:
		# get all the image names
		all_images = os.listdir(f"{relative_path}shapes/{path}")

		# iterate over the image names, get the label
		for image in all_images:
			image_path = f"{relative_path}shapes/{path}/{image}"
			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (32, 64))

			# get the HOG descriptor for the image
			hog_desc = hog(image, orientations=9, pixels_per_cell=(8, 8),
					cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

			# update the data and labels
			images.append(hog_desc)
			labels.append(path)


	# train Linear SVC 
	print('Training on train images...')
	svm_model = LinearSVC(random_state=42, tol=1e-5)
	svm_model.fit(images, labels)

	# save the model
	joblib.dump(svm_model, main_relative_path + "HOG_Model_SHAPES.npy")

def train_digits():
	images = []
	labels = []

	main_relative_path = "./flask-server/module1/training/"
	relative_path= main_relative_path + "dataset/"

	# get all the image folder paths
	image_paths = os.listdir(relative_path + "numbers")

	for path in image_paths:
		# get all the image names
		all_images = os.listdir(f"{relative_path}numbers/{path}")

		# iterate over the image names, get the label
		for image in all_images:
			image_path = f"{relative_path}numbers/{path}/{image}"
			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (32, 64))

			# get the HOG descriptor for the image
			hog_desc = hog(image, orientations=9, pixels_per_cell=(8, 8),
					cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

			# update the data and labels
			images.append(hog_desc)
			labels.append(path)


	# train Linear SVC 
	print('Training on train images...')
	svm_model = LinearSVC(random_state=42, tol=1e-5)
	svm_model.fit(images, labels)

	# save the model
	joblib.dump(svm_model, main_relative_path + "HOG_Model_DIGITS.npy")

def test():
	main_relative_path = "./flask-server/module1/training/"
	relative_path= main_relative_path + "dataset/"
	
	HOG = joblib.load(main_relative_path + "HOG_Model_DIGITS.npy")

	#all_images= sorted(os.listdir(f"./process_images/shapes/"))
	all_images= os.listdir(f"{relative_path}numbers/9/")
	for imagename in all_images:
		#image = cv2.imread(f"./process_images/cells/Row37.jpg", cv2.IMREAD_GRAYSCALE)
		image = cv2.imread(f"{relative_path}numbers/9/{imagename}", cv2.IMREAD_GRAYSCALE)

		resized_image = cv2.resize(image, (32, 64))

		# get the HOG descriptor for the test image
		(hog_desc, hog_image) = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
				cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)

		# prediction
		pred = HOG.predict(hog_desc.reshape(1, -1))[0]
		
		print(imagename,pred.title())


#test()
train_shapes()
train_digits()