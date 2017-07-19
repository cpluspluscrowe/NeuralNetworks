import numpy as np

def nonlin(x,deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1)) - 1
old_l1 = None
for iter in range(10):
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    print(l1)
    l1_error = y - l1
    l1_delta = l1_error*nonlin(l1,True)
    syn0 += np.dot(l0.T,l1_delta)
    old_l1 = l1
    print()
print(l1)
