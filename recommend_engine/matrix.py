import pandas as pd
import numpy as np

a = pd.DataFrame(
    [
        [2, 2],
        [3, 3]
    ]
)
b = pd.DataFrame(
    [
        [1, 2],
        [4, 4]
    ]
)
# Use the .head() method to view the contents of matrices a and b
print("Matrix A: ")
print (a.head())

print("Matrix B: ")
print (b.head())

# Complete the matrix with the product of matrices a and b
product = np.array([[10,12], [15,18]])

# Run this validation to see how your estimate performs
product == np.dot(a,b)



C = pd.DataFrame(
    [
        [3, 4, 5, 1, 2],
        [2, 5, 7, 6, 8],
        [1, 9, 0, 7, 6],
        [2, 2, 3, 3, 1]
    ]
)
D = pd.DataFrame(
    [
        [1, 2],
        [3, 3],
        [9, 8]
    ]
)
# Print the dimensions of C
print(C.shape)

# Print the dimensions of D
print(D.shape)

# Can C and D be multiplied together?
C_times_D = None if C.shape[1] != D.shape[0] else np.matmul(C, D)



G = pd.DataFrame(
    [
        [6, 6],
        [3, 3]
    ]
)
H = pd.DataFrame(
    [
        [2, 2],
        [1, 1]
    ]
)
I = pd.DataFrame(
    [
        [3, 3],
        [3, 3]
    ]
)
J = pd.DataFrame(
    [
        [1, 1],
        [2, 2]
    ]
)


# Take a look at Matrix G using the following print function
print("Matrix G:")
print(G)

# Take a look at the matrices H, I, and J and determine which pair of those matrices will produce G when multiplied together. 
print("Matrix H:")
print(H)
print("Matrix I:")
print(I)
print("Matrix J:")
print(J)

# Multiply the two matrices that are factors of the matrix G
prod = np.matmul(H, J)
print(G == prod)



def getRMSE(pred, actual):
    """
        Returns RMSE between predictions and actual observations
        Parameters:
            predictions: pandas dataframe of value predictions
            actual values: pandas dataframe of actual values that predictions are trying to predict
        Returns: RMSE value in decimal format
    """
    RMSE = (((pred - actual)**2).sum().sum()/(pred.shape[0]*pred.shape[1]))**.5
    return round(RMSE, 3)

L = pd.DataFrame(
    {
        0: {0: 1.0, 1: 0.01, 2: 1.0, 3: 0.1},
        1: {0: 0.0, 1: -0.42105263, 2: 0.0, 3: 1.0},
        2: {0: 0.0, 1: 0.09831579, 2: 1.0, 3: 0.0},
        3: {0: 0, 1: 1, 2: 0, 3: 0}
    }
)
U = pd.DataFrame(
    {
        0: {0: 1, 1: 0, 2: 0, 3: 0},
        1: {0: 2.0, 1: -0.19, 2: 0.0, 3: 0.0},
        2: {0: 1.0, 1: -0.099, 2: 1.0, 3: 0.0},
        3: {0: 2.0, 1: -0.198, 2: -1.0, 3: 0.19494737}
    }
)
LU = pd.DataFrame(
    {
        0: {0: 1.0, 1: 0.01, 2: 1.0, 3: 0.1},
        1: {0: 2.0, 1: 0.1, 2: 2.0, 3: 0.01},
        2: {0: 1.0, 1: 0.15, 2: 2.0, 3: 0.0},
        3: {0: 2.0, 1: 0.2, 2: 1.0, 3: 0.0}
    }
)
W = pd.DataFrame(
    {
        0: {0: 2.61, 1: 0.0, 2: 1.97, 3: 0.05},
        1: {0: 0.24, 1: 0.05, 2: 0.0, 3: 0.0},
        2: {0: 0.0, 1: 0.02, 2: 0.58, 3: 0.0},
        3: {0: 0.12, 1: 0.17, 2: 0.83, 3: 0.0}
    }
)
H = pd.DataFrame(
    {
        0: {0: 0.38, 1: 0.0, 2: 0.42, 3: 0.0},
        1: {0: 0.65, 1: 1.2, 2: 1.09, 3: 0.11},
        2: {0: 0.34, 1: 0.15, 2: 1.38, 3: 0.65},
        3: {0: 0.41, 1: 3.72, 2: 0.07, 3: 0.17}
    }
)
WH = pd.DataFrame(
    {
        0: {0: 0.99, 1: 0.01, 2: 0.99, 3: 0.02},
        1: {0: 2.0, 1: 0.1, 2: 2.0, 3: 0.03},
        2: {0: 1.0, 1: 0.15, 2: 2.01, 3: 0.02},
        3: {0: 1.98, 1: 0.22, 2: 0.99, 3: 0.02}
    }
)
# View the L, U, W, and H matrices.
print("Matrices L and U:") 
print(L)
print(U)

print("Matrices W and H:")
print(W)
print(H)

# Calculate RMSE between LU and M
print("RMSE of LU: ", getRMSE(LU, M))

# Calculate RMSE between WH and M
print("RMSE of WH: ", getRMSE(WH, M))