import optuna
import optuna_object

def optimize_parameter(n_trials):
    #optunaのstudyインスタンスを作成
    study = optuna.create_study()
    #optuna最適化を実行する
    study.optimize(optuna_object.objective,n_trials=n_trials)
    return study.trials_dataframe()#最適化結果のdfを返す
