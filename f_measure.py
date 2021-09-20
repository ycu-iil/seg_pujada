import keras.backend as K

def f_measure(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    
    TP = K.sum(y_true*y_pred)
    FP = K.sum((1-y_true)*y_pred)
    FN = K.sum(y_true*(1-y_pred))
    epsilon = 1.0
    return (TP + epsilon) / (TP + (FP + FN)/2. + epsilon)

def f_measure_loss(y_true, y_pred):
    return -f_measure(y_true, y_pred)
