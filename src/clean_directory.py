import os
import sys
import shutil


def clean_dir():
    sys.path.append(os.path.abspath('..'))
    # removing local directory
    dir_hyperband_tuner = "D:/Documentos/Mestrado/2021/UFPA Ciência Computação/EEG/time_series_prediction_on_EEG_data/notebooks/hyperband_tuner"
    dir__pycache__ = "D:/Documentos/Mestrado/2021/UFPA Ciência Computação/EEG/time_series_prediction_on_EEG_data/notebooks/src/__pycache__"
    dir_bests = "D:/Documentos/Mestrado/2021/UFPA Ciência Computação/EEG/time_series_prediction_on_EEG_data/notebooks/bests/"

    if os.path.exists(dir_hyperband_tuner):
        shutil.rmtree(dir_hyperband_tuner)
    if os.path.exists(dir__pycache__):
        shutil.rmtree(dir__pycache__)
    if os.path.exists(dir_bests):
        shutil.rmtree(dir_bests)
