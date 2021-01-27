# -*- coding: utf-8 -*-
"""dTect
Requirements
Tensorflow 1.14.0
Foolbox 2.1.0 
Randomgen==1.15.1
keras==2.1.2
Code is For CIFAR10 dataset. For MNIST or any other dataset, just change the image dimension and number of image parameters.
"""


import randomgen
import numpy as np
import keras
import random
from numpy.linalg import norm
from keras.datasets import cifar10
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import foolbox
from keras import backend
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils


##this model if for CIFAR10 only, for other datasets, replace this code-------------
from foolbox import zoo
url = "https://github.com/bethgelab/cifar10_challenge.git"
dNet = zoo.get_model(url)

#for loading some other keras model
backend.set_learning_phase(False)
model = keras.models.load_model('/address/to/your/model.h5')
fmodel = foolbox.models.KerasModel(model, bounds=(0,1))
dNet = fmodel


#Loading cifar-10 dataset-----------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train=np.reshape(y_train,(50000))
y_test=np.reshape(y_test,(10000))

#Performing PCA on training data to get all principal components and Loading test and train data-----------

data = np.reshape(x_train,(50000,3072))
pca=PCA(n_components=3072).fit(data)

train_img = np.load('location/to/file.npy')
train_label = np.load('location/to/file.npy')

test_img = np.load('location/to/file.npy')
test_label = np.load('location/to/file.npy')


#DETECTION FROM HERE---------------------


''' 

    n_components : number of least significant coefficients to perturb
    n_sample     : number of samples to generate
    A_i			 : Inital A
	B_i			 : Inital B
	lmbd1		 : Inital lambda1
	lmbd2		 : Inital lambda2
	alpha		 : False Alarm bound
	beta		 : Miss Probability bound
	A_L			 : Lower Bound for A
	A_U			 : Upper Bound for A
	B_L			 : Lower Bound for B
	B_U			 : Upper Bound for B

'''
n_components = #
n_sample = #
#Training Phase----------------
A, B = getAandB(dNet, np.reshape(train_img, (len(train_img), 3072)), train_label, n_sample, n_components, A_i, B_i, lmbd1, lmbd2, alpha, beta, A_L, A_U, B_L, B_U )
print(A)
print(B)



#Testing Phase------------------

'''rng = randomgen.RandomGenerator()'''


result = []

for k in tqdm(range(int(len(test_Img)))):

  is_adv, itr = isAdvOrNot(dNet, test_Img[k], A, B, n_sample, n_components)
  result.append([is_adv, itr])

#print(result)

''' finding number of iterations and other stats'''
predsAlg =[]
for i in range(len(result)):
  predsAlg.append(result[i][0])
itrAlg =[]
for i in range(len(result)):
  itrAlg.append(result[i][1])

true_adv = 0
true_clean = 0
det_clean = 0
det_adv = 0
miss_adv = 0
false_alrm = 0

true_adv_indx = []
true_clean_indx = []
det_clean_indx = []
det_adv_indx = []
miss_adv_indx = []
false_alrm_indx = []

num_itr_det_adv = []
num_itr_det_clean = []
num_itr_miss = []
num_itr_false = []

for i in range(int(len(testLabel))):
  if(testLabel[i] == 0):
    true_clean = true_clean + 1
    true_clean_indx.append(i)
    if(predsAlg[i] == 0):
      det_clean = det_clean + 1
      det_clean_indx.append(i)
      num_itr_det_clean.append(result[i][1])
    if(predsAlg[i] == 1):
      false_alrm = false_alrm + 1
      false_alrm_indx.append(i)
      num_itr_false.append(result[i][1])
  if(testLabel[i] == 1):
    true_adv = true_adv + 1
    true_adv_indx.append(i)
    if(predsAlg[i] == 0):
      miss_adv = miss_adv + 1
      miss_adv_indx.append(i)
      num_itr_miss.append(result[i][1])
    if(predsAlg[i] == 1):
      det_adv = det_adv + 1
      det_adv_indx.append(i)
      num_itr_det_adv.append(result[i][1])

print('Detected True Adversarial Images ' + str(true_adv) + ' Detected ' + str(det_adv) + ' Percentage ' + str(100*det_adv/true_adv) + ' Mean Itr ' + str(np.mean(num_itr_det_adv)))
print('True Clean Images ' + str(true_clean) + ' Detected ' + str(det_clean) + ' Percentage ' + str(100*det_clean/true_clean) + ' Mean Itr ' + str(np.mean(num_itr_det_clean)))
print('Correctly Classified Images ' + ' Detected ' + str(det_adv + det_clean) + ' Percentage ' + str(100*(det_adv + det_clean)/(true_adv + true_clean)) + ' Mean Itr ' + str((np.sum(num_itr_det_adv) + np.sum(num_itr_det_clean))/(len(num_itr_det_adv) + len(num_itr_det_clean))))
print('Missed Detection Images ' + str(miss_adv) + ' Percentage ' + str(100*miss_adv/true_adv) + ' Mean Itr ' + str(np.mean(num_itr_miss)))
print('False Detection Images ' + str(false_alrm) + ' Percentage ' + str(100*false_alrm/true_clean) + ' Mean Itr ' + str(np.mean(num_itr_false)))


def getAandB(net, images, labels, N_i, n_components, A_i, B_i, lmbd1_i, lmbd2_i, alpha, beta, AprL, AprU, BprL, BprU):
  '''takes in images as vector'''
  ''' '''
  ''' import random
      from sklearn.utils import shuffle '''
  no_of_pert = N_i
  no_of_images = len(images) 
  ''' assuming image dim 32*32*3'''

  '''assuming label 0 for clean and 1 for adv'''
  
  setOA = np.zeros((no_of_images))
  setOB = np.zeros((no_of_images))
  setOlmbd1 = np.zeros((no_of_images))
  setOlmbd2 = np.zeros((no_of_images))
  setOitr1 = np.zeros((no_of_images))
  setOitr2 = np.zeros((no_of_images))

  
  lst1 = [1, -1]

  A = A_i
  B = B_i
  lmbd1 = lmbd1_i
  lmbd2 = lmbd2_i

  n_goodImg = 0
  n_advImg = 0

  images = pca.transform(images)

  for j in range(no_of_images):
    
    a1 = #
    a2 = #
    dl1 = #
    dl2 = #
    d1 = #
    d2 = #

    print('Iteration ' + str(j) + ' of ' + str(no_of_images))

    #plt.imshow(np.reshape(images[j], (32, 32, 3))/255)
    #plt.show()
	
	
    b1 = random.choice(lst1)
    b2 = random.choice(lst1)
    A1 = A + dl1*b1
    if(A1 < 0):
      A1 = 10**(-20)
    B1 = B + dl2*b2
    if(B1 < 0):
      B1 = 0
    A2 = A - dl1*b1
    if(A2 < 0):
      A2 = 10**(-20)
    B2 = B - dl2*b2
    if(B2 < 0):
      B2 = 0
    
    #Handle the case of A1 or A2 and B1 and B2 being negative

    if(A1 > B1):
      B1 = A1 + A1/10
    if(A2 > B2):
      B2 = A2 + A2/10
    pred1, itr1 = isAdvOrNotAB(net, images[j], A1, B1, N_i, n_components)
    pred2, itr2 = isAdvOrNotAB(net, images[j], A2, B2, N_i, n_components)
    
    I_falarm1 = 0  
    I_miss1 = 0
    I_falarm2 = 0  
    I_miss2 = 0

    '''print('isactuallyAdv = ' + str(labels[j]))
    print('pred1 = ' + str(pred1) + ' itr1 = ' + str(itr1))
    print('pred2 = ' + str(pred2) + ' itr1 = ' + str(itr2))'''

    if(pred1 == 0 and labels[j] == 1):
      I_miss1 = 1
    elif(pred1 == 1 and labels[j] == 0):
      I_falarm1 = 1

    if(pred2 == 0 and labels[j] == 1):
      I_miss2 = 1
    elif(pred2 == 1 and labels[j] == 0):
      I_falarm2 = 1  

    '''print('falsealarm1 = ' + str(I_falarm1))
    print('falsealarm2 = ' + str(I_falarm2))
    print('miss1 = ' + str(I_miss1))
    print('miss2 = ' + str(I_miss2))'''


    c1 = itr1 + lmbd1*(I_falarm1) + lmbd2*(I_miss1)
    c2 = itr2 + lmbd1*(I_falarm2) + lmbd2*(I_miss2)

    A = A - a1*(c1 - c2)/(2*b1*dl1)
    if(A < AprL):
      A = AprL

    B = B - a2*(c1 - c2)/(2*b2*dl2)
    if(B < BprL):
      B = BprL 

    if(A > AprU):
      A = AprU
    if(B > BprU):
      B = BprU
    if(A == 0 and B == 0):
      if(j != 0):
        B = setOB[j-1]
      else:
        B = B_i
    if(A > B):
      B = A + A/10

    I_goodImg = 0
    I_advImg = 0
    

    if(labels[j] == 0):
      I_goodImg = 1
      n_goodImg = n_goodImg + 1
    else:
      I_advImg = 1
      n_advImg = n_advImg + 1

    '''print('c1 = ' + str(c1))
    print('c2 = ' + str(c2))
    print('GoodImg = ' + str(I_goodImg))
    print('AdvImg = ' + str(I_advImg))
    print('GImgs = ' + str(n_goodImg))
    print('advImgs = ' + str(n_advImg))'''

    pred_new, itr_new = isAdvOrNotAB(net, images[j], A, B, N_i, n_components)
    
	I_miss_new = 0
    I_falarm_new = 0
    
	if(pred_new == 0 and labels[j] == 1):
      I_miss_new = 1
    if(pred_new == 1 and labels[j] == 0):
      I_falarm_new = 1


    lmbd1 = lmbd1 + I_goodImg*(d1)*(I_falarm_new - alpha)
    lmbd2 = lmbd2 + I_advImg*(d2)*(I_miss_new - beta)

    '''print('A now = ' + str(A))
    print('B now = ' + str(B))
    print('lmbd1 now = ' + str(lmbd1))
    print('lmbd2 now = ' + str(lmbd2))
    print('**************************************************************')'''

    setOA[j] = A
    setOB[j] = B
    setOlmbd1[j] = lmbd1
    setOlmbd2[j] = lmbd2
    setOitr1[j] = itr1
    setOitr2[j] = itr2
  
  '''plt.title('A')
  plt.plot(setOA)
  plt.show()
  plt.title('B')
  plt.plot(setOB)
  plt.show()
  plt.title('Lambda1')
  plt.plot(setOlmbd1)
  plt.show()
  plt.title('Lambda2')
  plt.plot(setOlmbd2)
  plt.show()

  print('Mean iterations 1 = ' + str(np.mean(setOitr1)))
  print('Mean iterations 2 = ' + str(np.mean(setOitr2)))'''

  return A, B





def isAdvOrNotAB(net, image, A, B, N_i, n_components):
  '''takes a single image in as pca transformed vector'''
  is_adv = 0
  itr = 1
  adv_ratio = 1
  rng = randomgen.RandomGenerator()
  img_act = pca.inverse_transform(image)
  
  initial_vector = logit2probab(net.forward(np.reshape(img_act, (1, 32, 32, 3))))
  
  '''print('My A = ' + str(A))
  print('My B = ' + str(B))'''

  #plt.imshow(np.reshape(img_act, (32, 32, 3))/255)
  #plt.show()

  while(is_adv == 0 and itr < N_i+1):
    '''perturb'''
    '''assuming dimension 3072'''
    
    perturbation = rng.standard_normal(size=(n_components,), dtype=trans_x.dtype)
    new_x = np.copy(image)
    new_x[(3072-n_components):] += 10*perturbation
    
    inv_x = pca.inverse_transform(new_x)
    invx = np.reshape(inv_x,(1,32,32,3))

    tmp_pred = net.forward(np.reshape(invx, (1,32,32,3)))
    prob_vec = logit2probab(tmp_pred)

    #plt.imshow(np.reshape(invx, (32, 32, 3))/255)
    #plt.show()

    adv_prob = getQ(prob_vec, initial_vector)   

    #adv_ratio = adv_ratio*adv_prob
    adv_ratio = adv_ratio*adv_prob/abs(1-adv_prob)

    '''print('*adv_ratio now* ' + str(adv_ratio))'''
    toPert = chkBounds(adv_ratio, A, B)
    '''print('chkBounds result = ' + str(toPert))
    print('perturbed cat = ' + str(np.argmax(prob_vec)))
    print('original cat = ' + str(np.argmax(initial_vector)))'''


    if(toPert == -1):
       return 0, itr
    elif(toPert == 1):
      return 1, itr
      break
    else:
      itr = itr + 1
  if(itr > N_i):
    itr = N_i
  return is_adv, itr

def isAdvOrNot(net, image, A, B, N_i, n_components):
  '''takes a single in image as pca vector'''
  is_adv = 0
  itr = 1
  adv_ratio = 1
  rng = randomgen.RandomGenerator()
  img_act = pca.inverse_transform(image)
  
  initial_vector = logit2probab(net.forward(np.reshape(img_act, (1, 32, 32, 3))))
  
  #plt.imshow(np.reshape(img_act, (32, 32, 3))/255)
  #plt.show()

  while(is_adv == 0 and itr < N_i+1):
    '''perturb'''
    '''assuming dimension 3072'''
    
    perturbation = rng.standard_normal(size=(n_components,), dtype=trans_x.dtype)
    new_x = np.copy(image)
    new_x[(3072-n_components):] += 10*perturbation
    
    inv_x = pca.inverse_transform(new_x)
    invx = np.reshape(inv_x,(1,32,32,3))

    tmp_pred = net.forward(np.reshape(invx, (1,32,32,3)))
    prob_vec = logit2probab(tmp_pred)

    #plt.imshow(np.reshape(invx, (32, 32, 3))/255)
    #plt.show()

    adv_prob = getQ(prob_vec, initial_vector)   
    '''make this function'''

    #adv_ratio = adv_ratio*adv_prob
    adv_ratio = adv_ratio*adv_prob/abs(1-adv_prob)

    #print('*adv_ratio now* ' + str(adv_ratio))
    toPert = chkBounds(adv_ratio, A, B)
    #print('chkBounds result = ' + str(toPert))
    #print('perturbed cat = ' + str(np.argmax(prob_vec)))
    #print('original cat = ' + str(np.argmax(initial_vector)))


    if(np.argmax(prob_vec) != np.argmax(initial_vector)):
      #print('Category Changed')  
      is_adv = 1
      break

    if(toPert == -1):
      return 0, itr
    elif(toPert == 1):
      return 1, itr
    else:
      itr = itr + 1
  if(itr > N_i):
    itr = N_i  
  return is_adv, itr

def logit2probab(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x[0]) / np.sum(np.exp(x[0]))

def getQ(prob_vec, in_prob):
  #first order
  q = norm()
  return q


def chkBounds(qc, A, B):
  if(qc > B):
    return 1
  elif(qc < A):
    return -1
  else:
    return 0

#for classification algorithms----------------------------
#for CIFAR
#to get threshold value, getAandB(..) was used after minor modifications and appropritate dataset with appropriate labels.


def getQCln(prob_vec, initial_prob, net):
  
  #fill in the method of determining q
  q = #Function to get q
  return q


def chkBoundsCln(qc, B):
  
  a = np.zeros(len(qc))

  for i in range(len(qc)):
    if(qc[i] > B):
       a[i] = 1
  return a


def chkCleaned(bounds, prob_v, init_v):
  
  if(np.argmax(prob_v) != np.argmax(init_v)):
    '''category changed'''
    return np.argpartition(prob_v, -1)[-1:]
    '''threshold crossed'''
  if(np.amax(bounds) == 1):
    return np.argpartition(bounds, -1)[-1:]

  return [-1]


def isClnOrNot(net, image, B, N_i, n_components):
  
  is_adv = [-1]
  itr = 1
  #adv_ratio = 1
  rng = randomgen.RandomGenerator()
  img_act = pca.inverse_transform(image)
  
  initial_vector = logit2probab(net.forward(np.reshape(img_act, (1, 32, 32, 3))))
  
  adv_ratio = np.ones(len(initial_vector))


  #plt.imshow(np.reshape(img_act, (32, 32, 3))/255)
  #plt.show()

  while(np.amin(is_adv) == -1 and itr < N_i+1):
    '''perturb'''
    '''assuming dimension 3072'''
    
    perturbation = rng.standard_normal(size=(n_components,), dtype=trans_x.dtype)
    new_x = np.copy(image)
    new_x[(3072-n_components):] += 10*perturbation
    
    inv_x = pca.inverse_transform(new_x)
    invx = np.reshape(inv_x,(1,32,32,3))

    tmp_pred = net.forward(np.reshape(invx, (1,32,32,3)))
    prob_vec = logit2probab(tmp_pred)

    #plt.imshow(np.reshape(invx, (32, 32, 3))/255)
    #plt.show()

    adv_prob = getQCln(prob_vec, initial_vector, net)   

    adv_ratio = np.divide(np.multiply(adv_ratio, adv_prob), [abs(1-x) for x in adv_prob])
    
    toPert = chkBoundsCln(adv_ratio, B)

    is_adv = chkCleaned(toPert, prob_vec, initial_vector)
    
    if(np.amin(is_adv) != -1):
      	#just a numerical hack
	adv_ratio[is_adv[-1]] = adv_ratio[is_adv[-1]] + 2*B
    
    itr = itr+1

  ##just another numnerical hack, for an easier code
  ##after analyzing the list returned we can easily determine the predicted correct category
  return [x-B for x in adv_ratio], itr






