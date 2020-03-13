import sys
sys.path.append("../")
import utils

from keras.layers.advanced_activations import LeakyReLU

class CVAE():
    #def __init__(self,) ではインスタンス変数を定義する。(インスタンス変数はインスタンス間で共有されることはない)
    def __init__(self, original_dim,label_dim,first_hidden_dim,second_hidden_dim,third_hidden_dim,latent_dim,batch_size, epochs, learning_rate, kappa, beta,layer_depth,leaky_alpha,dr_rate):
        self.original_dim = original_dim
        self.label_dim = label_dim
        self.first_hidden_dim = first_hidden_dim
        self.second_hidden_dim = second_hidden_dim
        self.third_hidden_dim = third_hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.beta = beta
        self.layer_depth = layer_depth
        self.leaky_alpha = leaky_alpha
        self.dr_rate = dr_rate
        self.LeakyReLU = from keras.layers.advanced_activations import LeakyReLU
    #これはインスタンスメソッド
    def build_encoder_layer(self):     #trialはoputunaが最適化を行うモジュール
        # Input place holder for RNAseq data with specific input size
        self.leaky_relu = self.LeakyReLU(alpha=self.leaky_alpha)
        self.rnaseq_input = Input(shape=(self.original_dim, ))
        self.y_label_input = Input(shape=(self.label_dim,))
        # Input layer is compressed into a mean and log variance vector of size `latent_dim`
        # Each layer is initialized with glorot uniform weights and each step (dense connections, batch norm,
        # and relu activation) are funneled separately
        # Each vector of length `latent_dim` are connected to the rnaseq input tensor

        #rnaseqデータとラベルデータはクラスの外で前処理したデータを使う。
        self.merged_encode = keras.layers.concatenate([self.rnaseq_input,self.y_label_input], axis=-1)

        if self.layer_depth == 1:
            z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
            z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
            self.z_mean_encoded = self.leaky_relu(z_mean_dense_batchnorm)

            z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
            z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
            self.z_log_var_encoded = self.leaky_relu(z_log_var_dense_batchnorm)

            # return the encoded and randomly sampled z vector
            # Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
            z = Lambda(sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])
            self.z_condition = keras.layers.concatenate([z,self.y_label_input], axis=-1)
        elif self.layer_depth == 2:
            first_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
            first_hidden_dense_batchnorm = BatchNormalization()(first_hidden_dense_linear)
            first_hidden_encoded = self.leaky_relu(first_hidden_dense_batchnorm)

            z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded)
            z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
            self.z_mean_encoded = self.leaky_relu(z_mean_dense_batchnorm)

            z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded)
            z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
            self.z_log_var_encoded = self.leaky_relu(z_log_var_dense_batchnorm)

            z = Lambda(sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])
            self.z_condition = keras.layers.concatenate([z,self.y_label_input], axis=-1)
        elif self.layer_depth == 3:
            first_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
            first_hidden_dense_batchnorm = BatchNormalization()(first_hidden_dense_linear)
            first_hidden_encoded = self.leaky_relu(first_hidden_dense_batchnorm)
            first_hidden_encoded_drop = Dropout(self.dr_rate)(first_hidden_encoded)

            second_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded_drop)
            second_hidden_dense_batchnorm = BatchNormalization()(second_hidden_dense_linear)
            second_hidden_encoded = self.leaky_relu(second_hidden_dense_batchnorm)
            second_hidden_encoded_drop = Dropout(self.dr_rate)(second_hidden_encoded)

            third_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(second_hidden_encoded_drop)
            third_hidden_dense_batchnorm = BatchNormalization()(third_hidden_dense_linear)
            third_hidden_encoded = self.leaky_relu(third_hidden_dense_batchnorm)
            third_hidden_encoded_drop = Dropout(self.dr_rate)(third_hidden_encoded)

            z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(third_hidden_encoded_drop)
            z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
            self.z_mean_encoded = self.leaky_relu(z_mean_dense_batchnorm)

            z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(third_hidden_encoded_drop)
            z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
            self.z_log_var_encoded = self.leaky_relu(z_log_var_dense_batchnorm)

            z = Lambda(sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])
            self.z_condition = keras.layers.concatenate([z,self.y_label_input], axis=-1)

            self.encoder = Model(inputs=[self.rnaseq_input, self.y_label_input], outputs=[self.z_mean_encoded, self.z_log_var_encoded,z])
        return self.z_mean_encoded, self.z_log_var_encoded, self.encoder

    def build_decoder_layer(self):
        # The decoding layer is much simpler with a single layer glorot uniform initialized and sigmoid activation
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(self.third_hidden_dim,input_dim=self.latent_dim + self.label_dim))
        self.decoder_model.add(BatchNormalization())
        self.decoder_model.add(self.leaky_relu)
        self.decoder_model.add(Dropout(self.dr_rate))
        self.decoder_model.add(Dense(self.second_hidden_dim))
        self.decoder_model.add(BatchNormalization())
        self.decoder_model.add(self.leaky_relu)
        self.decoder_model.add(Dropout(self.dr_rate))
        self.decoder_model.add(Dense(self.third_hidden_dim))
        self.decoder_model.add(BatchNormalization())
        self.decoder_model.add(self.leaky_relu)
        self.decoder_model.add(Dropout(self.dr_rate))
        self.decoder_model.add(Dense(self.original_dim, activation='sigmoid'))#シグモイド関数で活性化させる場合
        #self.decoder_model.add(Dense(self.original_dim)
        self.rnaseq_reconstruct = self.decoder_model(self.z_condition)

    def compile_cvae(self):

        adam = optimizers.Adam(lr=self.learning_rate)
        cvae_layer = CustomVariationalLayer(self.z_log_var_encoded,
                                           self.z_mean_encoded)([self.rnaseq_input, self.rnaseq_reconstruct])
        self.cvae = Model([self.rnaseq_input,self.y_label_input], cvae_layer)
        self.cvae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])

        return self.cvae

    def get_summary(self):
        self.cvae.summary()

    def visualize_architecture(self, output_file):
        # Visualize the connections of the custom VAE model
        plot_model(self.cvae, to_file=output_file)
        SVG(model_to_dot(self.cvae).create(prog='dot', format='svg'))

    def train_cvae(self):
        self.hist = self.cvae.fit([np.array(X_train),np.array(condition_train)],
                                  shuffle=True,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=([np.array(X_test),np.array(condition_test)],None),
                                  callbacks=[WarmUpCallback(self.beta, self.kappa),TQDMNotebookCallback(leave_inner=True, leave_outer=True)]) #元はself.beta
        return self.hist

    def get_cvae_loss(self):
        loss=float(self.hist.history['val_loss'][-1])
        return loss

    def visualize_training(self, output_file):#(self, output_file)
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history)
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('VAE Loss')
        fig = ax.get_figure()
        #fig.savefig(output_file)

        return fig

    def compress(self, df):
        # エンコーディングするオブジェクト    #dfにはラベルと合わせたrna-seqデータを入力してください。
        self.encoder = Model([self.rnaseq_input,self.y_label_input], z)
        encoded_df = self.encoder.predict_on_batch(df)
        encoded_df = pd.DataFrame(encoded_df, columns=range(1, self.latent_dim + 1))#インデックスはあとで指定する。
        return encoded_df

    def get_decoder_weights(self):
        # build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim, ))  # can generate from any sampled z vector
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)
        weights = []
        for layer in self.decoder.layers:
            weights.append(layer.get_weights())
        return(weights)

    def reconstruct(self,df):

        self.decoder = Model(self.latent_variable_label_input,self.rnaseq_reconstruct)
        rna_reconstruction = self.decoder.predict(df)
        rna_reconstruction_df = pd.DataFrame(rna_reconstruction,columns=X_train.columns)
        return rna_reconstruction_df

    def predict(self, df):
        #学習済みネットワークを利用。
        decoder_input = Input(shape=(self.latent_dim + self.label_dim, ))
        _x_decoded = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded)
        return self.decoder.predict(df)

    def save_models(self, encoder_file, decoder_file): #ファイル名を指定して学習ずみモデルを保存
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)
