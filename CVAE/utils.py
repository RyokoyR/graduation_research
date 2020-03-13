import tensorflow as tf
from keras.callbacks import Callback

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):

        #最適化する起きはこちら(betaにはテンソルが格納されている)
        #if K.get_value(self.beta) <= 1:
        #    K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

        #最適化後にobject()関数以外からbetaを渡す場合こちらを使う(betaにはテンソルが格納)
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta,K.get_value(self.beta) + self.kappa)



def sampling(args):
  #global z
  # Function with args required for Keras Lambda function
  z_mean, z_log_var = args

  # Draw epsilon of the same shape from a standard normal distribution
  epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                          stddev=epsilon_std)

  # The latent vector is non-deterministic and differentiable
  # in respect to z_mean and z_log_var
  z = z_mean + K.exp(z_log_var / 2) * epsilon
  return z

  class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training

    """
    def __init__(self, var_layer, mean_layer, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        self.var_layer = var_layer
        self.mean_layer = mean_layer
        self.beta = beta
        super(CustomVariationalLayer, self).__init__(**kwargs) #superはクラスの多重継承(3つ以上のクラスの継承)

    def vae_loss(self, x_input, x_decoded):

        #self.beta = trial.suggest_uniform("beta",0,1)

        #再構成誤差の計算法は2種類ある。
        #バイナリークロスエントロピーは0~1の範囲の入出力の時に使える?
        #平均0分散1の標準化された入出力に使える。
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        #reconstruction_loss = mse(x_input, x_decoded)
        kl_loss = -0.5 * K.sum(1 + self.var_layer - K.square(self.mean_layer) -
                                K.exp(self.var_layer), axis=-1)
        return K.mean(reconstruction_loss + (self.beta * kl_loss)) #元はK.get_value(beta)

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
