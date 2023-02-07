import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import json

def get_evaluations(y_test, y_pred):
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    #UAC ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    #Acurácia
    acc = accuracy_score(y_test, y_pred)
    #Matriz de confusão
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()
    #Sensibilidade/Recall
    recall = true_pos / (true_pos + false_neg)
    #Especificidade
    specificity = true_neg / (true_neg + false_pos)
    #Precisão
    precision = true_pos / (true_pos + false_pos)
    #F1-Score
    f1 = 2 * ((precision * recall) / (precision + recall))

    return {
        'fpr': np.round(fpr, 4).tolist(),
        'tpr': np.round(tpr, 4).tolist(),
        'roc_auc' : np.round(roc_auc, 4),
        'acc': np.round(acc, 4),
        'recall': np.round(recall, 4),
        'specificity': np.round(specificity, 4),
        'precision': np.round(precision, 4),
        'f1': np.round(f1, 4)
    }

def save_avg_eval(scores, name, fprs, tprs, rocs_aucs, accs, recalls, specificitys, precisions, f1s):
    scores[name] = {
    'roc': {'avg': {'taxa_fp': np.mean(fprs, axis=0).tolist(), 'taxa_tp': np.mean(tprs, axis=0).tolist(), 'roc_auc': np.mean(rocs_aucs, axis=0).tolist()},
            'std': {'taxa_fp': np.std(fprs, axis=0).tolist(), 'taxa_tp': np.std(tprs, axis=0).tolist(), 'roc_auc': np.std(rocs_aucs, axis=0).tolist()}},
    'acc': {'avg': np.mean(accs, axis=0).tolist(), 'std': np.std(accs, axis=0).tolist()},
    'recall': {'avg': np.mean(recalls, axis=0).tolist(), 'std': np.std(recalls, axis=0).tolist()},
    'specificity': {'avg': np.mean(specificitys, axis=0).tolist(), 'std': np.std(specificitys, axis=0).tolist()},
    'precision': {'avg': np.mean(precisions, axis=0).tolist(), 'std': np.std(precisions, axis=0).tolist()},
    'f1': {'avg': np.mean(f1s, axis=0).tolist(), 'std': np.std(f1s, axis=0).tolist()}
}
    #salva os resultados
    with open('results/resultados.json', 'w') as f:
        json.dump(scores, f, indent=4)
    
    return scores

def save_eval(scores, name, eval):
    scores[name] = {
    'roc': {'avg': {'taxa_fp': eval['fpr'], 'taxa_tp': eval['tpr'], 'roc_auc': eval['roc_auc']},
            'std': {'taxa_fp': 0, 'taxa_tp': 0, 'roc_auc': 0}},
    'acc': {'avg': eval['acc'], 'std': 0},
    'recall': {'avg': eval['recall'], 'std': 0},
    'specificity': {'avg': eval['specificity'], 'std': 0},
    'precision': {'avg': eval['precision'], 'std': 0},
    'f1': {'avg': eval['f1'], 'std': 0}
}
    #salva os resultados
    with open('results/resultados.json', 'w') as f:
        json.dump(scores, f, indent=4)
    
    return scores

def save_acc_scores(acc_scores, name, accs):
    acc_scores = acc_scores
    acc_scores[name] = accs
    #salva os resultados
    with open('results/acuracias.json', 'w') as f:
        json.dump(acc_scores, f, indent=4)
    return acc_scores

def my_accuracy(y_test, y_pred):
    soma = 0
    # y_pred = np.where(np.array(y_pred) < 0.5, 0, 1)
    # soma = np.sum(y_test == y_pred)

    # y_pred = tf.where(y_pred < 0.5, 0, 1)
    # soma = tf.reduce_sum(tf.equal(y_test, y_pred))

    # norm y_pred entre 0 e 1
    y_pred = tf.divide(tf.subtract(y_pred, tf.reduce_min(y_pred)), tf.subtract(
        tf.reduce_max(y_pred), tf.reduce_min(y_pred)))
    y_pred = tf.cast(tf.where(y_pred < 0.5, 0, 1), tf.float32)
    soma = tf.reduce_sum(tf.cast(tf.equal(y_test, y_pred), tf.float32))
    return (soma / tf.cast(len(y_test), tf.float32))
