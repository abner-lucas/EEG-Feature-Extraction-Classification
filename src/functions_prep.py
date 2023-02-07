import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pywt

def tukeys(df, variable):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3-q1

    # inner fence lower and upper end
    inner_fence_le = q1-1.5*iqr
    inner_fence_ue = q3+1.5*iqr

    # outer fence lower and upper end
    outer_fence_le = q1-3*iqr
    outer_fence_ue = q3+3*iqr

    outliers_prob, outliers_poss = [], []
    for index, x in enumerate(df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    return outliers_prob, outliers_poss

def z_score(df, variable):
    mean = df[variable].mean()
    std = df[variable].std()
    outliers = []
    # for index, x in enumerate(df[variable]):
    #     if x <= mean-3*std or x >= mean+3*std:
    #         outliers.append(index)
    outliers = df[variable].loc[(
        df[variable] <= mean-3*std) | (df[variable] >= mean+3*std)].index
    return outliers

def data_norm(data, v_min, v_max):
    # data = np.reshape(data, (v_min, 1))
    norm = MinMaxScaler(feature_range=(v_min, v_max))
    return norm.fit_transform(data)

def data_cat_for_num(data):
    labelencoder = LabelEncoder()
    return labelencoder.fit_transform(data)

def calculate_energy(coeffs):
    # Calcular energia do sinal = raiz quadrada da soma dos quadrados dos módulos dos elementos, ao quadrado 
    # Norma de Frobenius/L2/euclidiana em np.linalg.norm elevado ao quadrado
    eng = np.linalg.norm(coeffs, ord=2)**2
    return [eng]

def get_eeg_features(data, wavelet, level):
    eeg_features = []

    for channel in range(data.shape[1]):
        features = []
        signal = [x[channel] for x in data]
        list_coeff = pywt.wavedec(signal, wavelet, level=level)
        
        for coeff in list_coeff:  
            features += calculate_energy(coeff)
                
        eeg_features.append(features)

    # mudar de [channel, features] para o formato [features, channel]
    eeg_features = np.array(eeg_features).T

    return eeg_features
