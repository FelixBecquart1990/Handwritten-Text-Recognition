import argparse
import shutil
import pathlib
import os
import cv2
import pandas as pd
import numpy as np

# Path of folder
img_path = './data/0916_Data Samples 2'
img_test_path = './data/1015_Private Test'
img_sample_path = './data/0825_DataSamples 1'

data_path = './data/train'
test_path = './data/test'
sample_path = './data/sample'

img_train_folder = pathlib.Path(img_path)
img_test_folder = pathlib.Path(img_test_path)
img_sample_folder = pathlib.Path(img_sample_path)


# labels_json_file = './data/0916_Data Samples 2/labels.json'
# labels_json = pd.read_json(labels_json_file, orient='index').reset_index()
# labels_json.columns = ['name', 'label']


# Get images paths from origine folder
all_img_path = [str(item) for item in img_train_folder.glob('**/*.*') if item.is_file()]
len_img_path = len(all_img_path)
print(len_img_path, " (length of len_img_path)")

# Get all paths from test folder
all_test_path = [str(item) for item in img_test_folder.glob('**/*.*') if item.is_file()]
len_test_path = len(all_test_path)
print(len_test_path, " (length of len_test_path)")

# Get all paths from test folder
all_sample_path = [str(item) for item in img_sample_folder.glob('**/*.*') if item.is_file()]
len_sample_path = len(all_sample_path)
print(len_sample_path, " (length of len_sample_path)")

def sort_contours(cnts):
  
  # initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def preprocess(image_ori, input_size):
  rows, cols = image_ori.shape[:2]
  # Let's find ende and convert to b&w image
  image = cluster(image_ori)

  # Finding Contours
  cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

  #sort contour
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  cnts, boundingBoxes = sort_contours(cnts)

  x, _, _, _  = boundingBoxes[-1]
  
  if x + 60 < cols: 
    #resize image to the right size
    image = image[ 0:rows, 0: x + 60 ]
    image = cv2.resize(image, input_size , interpolation = cv2.INTER_AREA)
    return image
  
  #resize image to the right size
  image = cv2.resize(image, input_size , interpolation = cv2.INTER_AREA)
  
  return image 

def cluster(image):
  # define criteria, number of clusters(K) and apply kmeans()
  K = 2
  attempts=10
  
  # reshape and convert to np.float32
  vectorized = image.reshape((-1,3)) 
  vectorized = np.float32(vectorized)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
  
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  final_result = center[label.flatten()]
  final_result = final_result.reshape((image.shape))
  
  #precess image to b&w
  final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
  final_result = cv2.GaussianBlur(final_result, (7,7), 0)
  (_, final_result) = cv2.threshold(final_result, 125, 255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  return final_result

def images_preprocessing_and_saving(dst, all_img_path):
  if os.path.isdir(dst):
    shutil.rmtree(dst)
    os.mkdir(dst)
    print('folder created and remove old folder')
  else:
    os.mkdir(dst)
    print('Created')
  
  for path in all_img_path:
    file_name = pathlib.Path(path).name
    print(file_name)
    if file_name.split('.')[1].lower() != 'json':
      image = cv2.imread(str(path))
      
      dsize = (1280, 300)
      output = preprocess(image, dsize)
    
      path = os.path.join(dst, file_name)
      cv2.imwrite(path, output)


def tester(param):
    print(param)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default='oui', type=str)
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    # tester(args.test)

    images_preprocessing_and_saving(data_path, all_img_path)
    # images_preprocessing_and_saving(test_path, all_test_path)
    # images_preprocessing_and_saving(sample_path, all_sample_path)

