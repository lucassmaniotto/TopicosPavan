import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
  root_mean_squared_error,
  mean_absolute_error,
  r2_score,
)

import joblib
import optuna

def objective(trial, X_train, y_train):
  n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)
  hidden_layer_sizes = []
  for i in range(n_hidden_layers):
    hidden_layer_size = trial.suggest_int(f'hidden_layer_size_{i}', X_train.shape[1], 32)
    hidden_layer_sizes.append(hidden_layer_size)

  activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
  # only suggest solvers supported by sklearn's MLPRegressor
  solver = trial.suggest_categorical('solver', ['lbfgs', 'adam', 'sgd'])
  learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True)
  alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)


  mlp = MLPRegressor(
    hidden_layer_sizes=tuple(hidden_layer_sizes),
    activation=activation,
    solver=solver,
    learning_rate_init=learning_rate_init,
    alpha=alpha,
    max_iter=200,
    random_state=42,
  )

  from sklearn.model_selection import cross_val_score
  try:
    score = cross_val_score(mlp, X_train, y_train, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
    return score.mean()
  except Exception as exc:
    try:
      trial.set_user_attr('error', str(exc))
    except Exception:
      pass
    return -1e6


def main():
  base = Path(__file__).resolve().parent / "content"

  if not base.exists():
    base = Path(__file__).resolve().parents[1] / "content"

  # Busca os arquivos de dados pré-processados
  files = {
    "X_train": base / "X_train_scaled.pkl",
    "y_train": base / "y_train_scaled.pkl",
    "X_test": base / "X_test_scaled.pkl",
    "y_test": base / "y_test_scaled.pkl",
    "X_val": base / "X_val_scaled.pkl",
    "y_val": base / "y_val_scaled.pkl",
  }

  # Joga os dados para dentro de variáveis
  X_train = joblib.load(files["X_train"])
  y_train = joblib.load(files["y_train"])
  X_test  = joblib.load(files["X_test"])
  y_test  = joblib.load(files["y_test"])
  X_val = joblib.load(files["X_val"])
  y_val = joblib.load(files["y_val"])

  # Converte tudo p array
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)
  X_test = np.asarray(X_test)
  y_test = np.asarray(y_test)
  X_val = np.asarray(X_val)
  y_val = np.asarray(y_val)

  # Parâmetros iniciais do modelo
  mlp = MLPRegressor(
    max_iter=200,          # Número de iterações ou épocas (reduzido para teste rápido)
    random_state=42,
  )

  # Configura o Optuna (menos trials para teste rápido)
  study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

  # Executa o Optuna (n_trials reduzido para execução rápida)
  study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=5)

  print("Best trial:")
  print(f"  Value: {study.best_trial.value}")
  print(f"  Params: {study.best_trial.params}")

  best_params = study.best_trial.params

  # Treina o melhor modelo pelo Optuna
  best_mlp_model = MLPRegressor(
    hidden_layer_sizes=tuple([best_params[f'hidden_layer_size_{i}'] for i in range(best_params['n_hidden_layers'])]),
    activation=best_params['activation'],
    solver=best_params['solver'],
    learning_rate_init=best_params['learning_rate_init'],
    alpha=best_params['alpha'],
    max_iter=1000,
    random_state=42,
  )

  best_mlp_model.fit(X_train, y_train)

  # Avalia no conjunto de validação
  y_final = best_mlp_model.predict(X_val)


  y_pred = np.ravel(y_final)
  y_true = np.ravel(y_val)

  # align lengths if needed
  if y_pred.shape != y_true.shape:
    n = min(y_pred.shape[0], y_true.shape[0])
    y_pred = y_pred[:n]
    y_true = y_true[:n]

  residuals = y_pred - y_true
  # scatter plot of residuals (only points, no connecting lines)
  plt.figure(figsize=(8, 4))
  plt.plot(residuals, marker='o', linestyle='None', markersize=3)
  plt.axhline(0, color='red', linewidth=1)
  plt.xlabel('Amostra')
  plt.ylabel('Erro (y_pred - y_true)')
  plt.title('Resíduos: diferença entre predição e valor real')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  # histogram of residuals
  plt.figure(figsize=(6, 4))
  plt.hist(residuals, bins=30)
  plt.xlabel('Erro (y_pred - y_true)')
  plt.title('Histograma dos resíduos')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  # basic metrics
  print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
  print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
  print(f"R2: {r2_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
  main()