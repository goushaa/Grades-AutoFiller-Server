from skimage.feature import hog
import joblib
import cv2

def get_predict(image,model="SHAPES"):
	main_relative_path = "./flask-server/module1/"
	relative_path= main_relative_path + "dataset/"

	if model =="SHAPES":
		HOG = joblib.load(main_relative_path + "HOG_Model_SHAPES.npy")
	elif model == "DIGITS":
		HOG = joblib.load(main_relative_path + "HOG_Model_DIGITS.npy")

	resized_image = cv2.resize(image, (32, 64))

	# get the HOG descriptor for the test image
	(hog_desc, hog_image) = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
			cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)

	# prediction
	pred = HOG.predict(hog_desc.reshape(1, -1))[0]
	return pred.title()
