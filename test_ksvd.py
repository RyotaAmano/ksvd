import numpy as np
from scipy import linalg
from ksvd import KSVD
from PIL import Image
import os
import sys

def ksvd_random_signal(): # random signal test
    A = np.random.randn(30, 60)
    for j in range(A.shape[1]):
        A[:,j] /= np.linalg.norm(A[:,j])

    X = np.zeros((A.shape[1], 4000))
    candidate = np.array(list(range(X.shape[1])))

    for j in range(X.shape[1]):
          marker = np.random.choice(candidate, 4, replace=False)

    X[:,marker] = np.random.normal(0.0, 1.0, 4)
    Y_a = X + np.random.normal(0.0, 0.1, X.shape)

    ksvd = KSVD(rank=np.linalg.matrix_rank(Y_a.T),num_of_NZ=4)
    A, X= ksvd.fit(Y_a)
    return A, X

def ksvd_image(filename,rank): # image compression
    path, ext = os.path.splitext(filename)
    image = Image.open(filename)
    w = image.width
    h = image.height
    gray_image = image.convert('L')
    a = np.asarray(gray_image)
    ksvd = KSVD(rank=rank)
    A,X = ksvd.fit(a)
    image = Image.fromarray(np.uint8(A.dot(X)))
    print(np.sum(A>0))
    file = path+'_r' + str(rank) + '_ksvd' + ext
    image.save(file)
    print('Saved as ' + file)


#A,X = ksvd_random_signal()
ksvd_image('cat.jpg',25)
