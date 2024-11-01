import pandas as pd
import numpy as np


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


def getRMSEs(listOfPredMatrices, actualValues):
    """ 
    Computes the RMSE between predictions and actual values for various sets of predictions.
    
    Parameters:
      - pred: list of matrices each containing predictions
      - actual: array of true values
      
    Returns: Printed statement of integer index of predition and respective RMSE based on actual values provided.
    """
    for i, pred in enumerate(listOfPredMatrices):
        MSE = (((pred - actualValues)**(2)).sum().sum()/(pred.shape[0]*pred.shape[1]))
        print ("F" + str(i+2) + ":", MSE**.5)


U = pd.DataFrame(
    {
        'U_LF_1': {
            'User_1': 0.8,
            'User_2': 0.4,
            'User_3': 0.05,
            'User_4': 0.3,
            'User_5': 0.1,
            'User_6': 0.0,
            'User_7': 0.01,
            'User_8': 0.9,
            'User_9': 1.0
        },
        'U_LF_2': {
            'User_1': 0.01,
            'User_2': 0.01,
            'User_3': 2.1,
            'User_4': 0.01,
            'User_5': 1.5,
            'User_6': 0.03,
            'User_7': 0.02,
            'User_8': 0.7,
            'User_9': 2.0
        },
        'U_LF_3': {
            'User_1': 0.3,
            'User_2': 0.06,
            'User_3': 0.01,
            'User_4': 0.2,
            'User_5': 0.9,
            'User_6': 0.4,
            'User_7': 0.66,
            'User_8': 0.0,
            'User_9': 0.04
        },
        'U_LF_4': {
            'User_1': 0.8,
            'User_2': 0.2,
            'User_3': 2.2,
            'User_4': 0.2,
            'User_5': 0.0,
            'User_6': 0.5,
            'User_7': 0.4,
            'User_8': 1.0,
            'User_9': 0.2
        }
    }
)
P = pd.DataFrame(
    {
        'Movie_1': {'P_LF_1': 0.5, 'P_LF_2': 0.2, 'P_LF_3': 0.3, 'P_LF_4': 1.0},
        'Movie_2': {'P_LF_1': 0.1, 'P_LF_2': 2.0, 'P_LF_3': 1.9, 'P_LF_4': 0.2},
        'Movie_3': {'P_LF_1': 0.4, 'P_LF_2': 0.0, 'P_LF_3': 0.6, 'P_LF_4': 1.0},
        'Movie_4': {'P_LF_1': 1.1, 'P_LF_2': 0.01, 'P_LF_3': 0.9, 'P_LF_4': 0.89}
    }
)
# Multiply factor matrices
UP = np.matmul(U,P)

# Convert to pandas DataFrame
print(pd.DataFrame(UP, columns = P.columns, index = U.index))


T = pd.DataFrame(
    {
        0: {
            0: 1.292,
            1: 0.42,
            2: 0.08,
            3: 0.412,
            4: 0.62,
            5: 0.626,
            6: 0.0,
            7: 1.59,
            8: 0.0
        },
        1: {
            0: 0.0,
            1: 0.0,
            2: 4.664,
            3: 0.47,
            4: 0.0,
            5: 0.0,
            6: 1.375,
            7: 1.69,
            8: 4.216
        },
        2: {
            0: 1.3,
            1: 0.396,
            2: 2.226,
            3: 0.0,
            4: 0.58,
            5: 0.0,
            6: 0.8,
            7: 1.36,
            8: 0.624
        },
        3: {
            0: 0.0,
            1: 0.6721,
            2: 2.043,
            3: 0.0,
            4: 0.935,
            5: 0.8053,
            6: 0.9612,
            7: 0.0,
            8: 0.0
        }
    }
)
F1 = pd.DataFrame(
    {
        0: {0: 2, 1: 1, 2: 1, 3: 1, 4: 3, 5: 4, 6: 2, 7: 4, 8: 4},
        1: {0: 4, 1: 3, 2: 4, 3: 4, 4: 3, 5: 2, 6: 4, 7: 3, 8: 1},
        2: {0: 3, 1: 2, 2: 4, 3: 4, 4: 1, 5: 4, 6: 3, 7: 3, 8: 3},
        3: {0: 3, 1: 1, 2: 3, 3: 3, 4: 3, 5: 1, 6: 4, 7: 4, 8: 2}
    }
)
F2 = pd.DataFrame(
    {
        0: {
            0: 0.7276845605456583,
            1: 0.2972162515949431,
            2: 0.6684103841772332,
            3: 0.10356292108571152,
            4: 0.2788049127462368,
            5: 0.1833175716784896,
            6: 0.0,
            7: 0.9819681824878523,
            8: 0.0
        },
        1: {
            0: 0.0,
            1: 0.0,
            2: 5.133314382547687,
            3: 0.4494655185366921,
            4: 0.0,
            5: 0.0,
            6: 1.817580601657848,
            7: 2.720610558441755,
            8: 2.667366674847455
        },
        2: {
            0: 0.7816137663665544,
            1: 0.2898334245675139,
            2: 2.090947607279091,
            3: 0.0,
            4: 0.5643273601209946,
            5: 0.0,
            6: 0.7181012176124341,
            7: 1.029218987922719,
            8: 0.8200154534789706
        },
        3: {
            0: 0.0,
            1: 0.1499720792763105,
            2: 1.7276154180386931,
            3: 0.0,
            4: 0.32621995264108916,
            5: 0.22984785075635933,
            6: 0.6873269305813324,
            7: 0.0,
            8: 0.0
        }
    }
)
F3 = pd.DataFrame(
    {
        0: {
            0: 1.492244544405576,
            1: 0.3971790359091946,
            2: 0.1412048503383586,
            3: 0.1092555086372637,
            4: 0.6192730100600089,
            5: 0.37792092895633184,
            6: 0.0,
            7: 1.391170523003995,
            8: 0.0
        },
        1: {
            0: 0.0,
            1: 0.0,
            2: 4.940441255371812,
            3: 0.3430329338282129,
            4: 0.0,
            5: 0.0,
            6: 1.3506756720195645,
            7: 1.8109451554711358,
            8: 3.820890212142733
        },
        2: {
            0: 0.9906046679345504,
            1: 0.43573289233260093,
            2: 1.8814918023645575,
            3: 0.0,
            4: 0.6868592910926297,
            5: 0.0,
            6: 0.6762102891509367,
            7: 1.3902807929383285,
            8: 1.1602964864686154
        },
        3: {
            0: 0.0,
            1: 0.5634621047651994,
            2: 1.9051201248296687,
            3: 0.0,
            4: 0.9732927573503258,
            5: 0.6082012051404062,
            6: 0.8832044907058253,
            7: 0.0,
            8: 0.0
        }
    }
)

# Use getRMSE(preds, actuals) to calculate the RMSE of matrices T and F1.
getRMSE(F1, T)

# Create list of F2, F3, F4, F5, and F6
Fs = [F2, F3, F4, F5, F6]

# Calculate RMSE for F2, F3, F4, F5, and F6.
getRMSEs(Fs, T)