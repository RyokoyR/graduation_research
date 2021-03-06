#optunaが最適化する関数
def objective(trial):
    global layer_depth
    global first_hidden_dim
    global second_hidden_dim
    global third_hidden_dim
    global latent_dim
    global beta
    global kappa
    global batch_size
    global learning_rate
    #leaky_reluの時に使用する
    global leaky_alpha
    #ドロップアウトを用いる場合
    global dr_rate

    leaky_alpha = trial.suggest_loguniform('lealy_alpha',1e-5,1e-2)
    dr_rate = trial.suggest_loguniform('dr_rate',1e-5,1e-2)
    #layer_depth = trial.suggest_int("layer_num",1,3)
    layer_depth = trial.suggest_int("layer_num",3,3)#3層に固定
    #レイヤー数に応じたノードの数をとるハイパーパラメータ変数を作る
    if layer_depth == 1:
        first_hidden_dim = 0
        second_hidden_dim = 0
    elif layer_depth == 2:
        first_hidden_dim = int(trial.suggest_int("first_layer_dim",100,2000))
        second_hidden_dim = 0
    elif layer_depth == 3:
        first_hidden_dim = int(trial.suggest_int("first_layer_dim",100,2000))
        second_hidden_dim = int(trial.suggest_int("second_layer_dim",100,first_hidden_dim))
        third_hidden_dim = int(trial.suggest_int("third_layer_dim",100,second_hidden_dim))
    latent_dim = trial.suggest_int("latent_dim",25,250)

    #潜在空間の次元が隠れそうより大きくならないように条件分岐
    if layer_depth ==2:
        if latent_dim > first_hidden_dim:
            latent_dim = trial.suggest_int("latent_dim",25,first_hidden_dim)
        else:
            pass
    elif layer_depth == 3:
        if latent_dim > second_hidden_dim:
            latent_dim = trial.suggest_int("latent_dim",25,second_hidden_dim)
        else:
            pass
    else:
        pass

    beta = trial.suggest_discrete_uniform("beta",0.0,0.5,0.01)
    beta = K.variable(beta)
    kappa = trial.suggest_discrete_uniform("kappa",0.0,0.5,0.01)
    batch_size=trial.suggest_categorical('batch_size', [64,128,256, 512, 1024])
    learning_rate = trial.suggest_loguniform('learning_rate',1e-5,1e-2)

    cvae_model = CVAE(original_dim=original_dim,label_dim=label_dim,
                    first_hidden_dim = first_hidden_dim,
                    second_hidden_dim = second_hidden_dim,
                    third_hidden_dim = third_hidden_dim,
                    latent_dim = latent_dim,
                    batch_size = batch_size,
                    epochs = epoch,
                    learning_rate = learning_rate,
                    kappa = kappa,
                    beta = beta,
                    layer_depth = layer_depth,
                    leaky_alpha = leaky_alpha,
                    dr_rate = dr_rate)

    cvae_model.build_encoder_layer()
    cvae_model.build_decoder_layer()
    cvae_model.compile_cvae()
    cvae_model.train_cvae()
    loss=cvae_model.get_cvae_loss()
    return loss
