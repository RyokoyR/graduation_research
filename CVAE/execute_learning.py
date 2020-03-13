#モジュール読み込み
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda, Layer, Activation, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras
import optuna
import pydot
import graphviz
from keras.utils import plot_model
from keras_tqdm import TQDMNotebookCallback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn import manifold
from matplotlib.cm import get_cmap
from matplotlib import ticker
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import plotly.offline as offline
import h5py

#自作モジュール読み込み
import data_loader
import utils
import model.cvae

#data_loaderでデータ読み込み
X_train,condition_train,X_test,condition_test = data_loader.data_loader("/Users/yokosaka/Desktop/python/1_54rnaseq_drop_.txt","/Users/yokosaka/Desktop/python/1_54rnaseq_drop_label.txt",0.9)

#学習条件の指定
original_dim = X_train.shape[1]
label_dim=label_dim = condition_train.shape[1]
epsilon_std = 1.0
epoch = 50

#モデルのハイパーパラメータは変数に入れて渡す。
cvae_model = model.cvae.CVAE(original_dim=original_dim,label_dim=label_dim,
                    first_hidden_dim = first_hidden_dim,
                    second_hidden_dim = second_hidden_dim,
                    third_hidden_dim = third_hidden_dim,
                    latent_dim = latent_dim,
                    batch_size = batch_size,
                    epochs = epoch,
                    learning_rate = learning_rate,
                    kappa = kappa,
                    beta = beta,
                    layer_depth=3,
                    leaky_alpha = leaky_alpha,
                    dr_rate = dr_rate)

#エンコーダーモデルを構築
cvae_model.build_encoder_layer()
#デコーダーモデルを構築
cvae_model.build_decoder_layer()
モデルのコンパイル
cvae_model.compile_cvae()
#モデルをトレーニング
cvae_model.train_cvae()
