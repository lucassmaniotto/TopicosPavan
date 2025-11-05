import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
	root_mean_squared_error,
	mean_absolute_error,
	r2_score,
)

import joblib

def main():
	base = Path(__file__).resolve().parents[1] / "content"
	
	if not base.exists():
		base = Path(__file__).resolve().parent / ".." / "content"

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
		max_iter=1000,		# Número de iterações ou épocas
		random_state=42,
	)

	# Parametros para o Grid Search
	param_grid = {
        'hidden_layer_sizes': [(12,32), (32,), (12, 50), (32, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'nadam'],
        'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01],
		'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    }

	# Descomentar se quiser testar sem esperar o Grid Search completo
	#param_grid = {
	#	'hidden_layer_sizes': [(32,50)],
	#	'activation': ['relu'],
	#	'solver': ['adam'],
	#	'learning_rate_init': [0.01],
	#	'alpha': [0.001],
	#}

	# Configura o Grid Search
	grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

	# Executa o Grid Search
	grid_search.fit(X_train, y_train)

	print(f"Best parameters: {grid_search.best_params_}")
	print(f"Best score (negative MSE): {grid_search.best_score_}")
	best_mlp_model = grid_search.best_estimator_

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

