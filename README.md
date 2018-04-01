# ksvd

A K-SVD implementation written in Python


Requirements
---------------------------------
ksvd.py requires the following to run:
Python 3 or more


About
----------------------------------
This is a k-svd implementation written in Python.
This k-svd uses OMP(Othogonal Matching Pursuit) to estimate sparse coefficients
and Approximate svd to estimate dictionary more quickly than normal method.


Feature
----------------------------------
*Approximate k-svd (Default)
*Normal k-svd (using np.linalg.svd)


Usage  
----------------------------------
```
import numpy as np
from ksvd import KSVD

Y=random.randn(60,1000)
ksvd = KSVD(rank=np.linalg.matrix_rank(Y),num_of_NZ=4)
A, X= ksvd.fit(Y)

"""
Y = AX
Y: shape = [n_features,n_samples]
A: Dictionary = [n_features, rank]
X: Sparse = [rank, n_samples]
"""
```

'test_ksvd.py' contains two demo-programs using k-svd.
