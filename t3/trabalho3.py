import matplotlib.pyplot as plt
import joblib
import pandas as pd # Mantido caso precise para outras operações, mas joblib será usado para carregar


X_train_scaled = joblib.load('../content/X_train_scaled.pkl')
y_train_scaled = joblib.load('../content/y_train_scaled.pkl')
X_val_scaled = joblib.load('../content/X_val_scaled.pkl')
y_val_scaled = joblib.load('../content/y_val_scaled.pkl')
X_test_scaled = joblib.load('../content/X_test_scaled.pkl')
y_test_scaled = joblib.load('../content/y_test_scaled.pkl')

print("Dados carregados com sucesso!")
print(f"Shape dos dados de treino: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
print(f"Shape dos dados de validação: X={X_val_scaled.shape}, y={y_val_scaled.shape}")
print(f"Shape dos dados de teste: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

from sklearn.ensemble import RandomForestRegressor

# Instanciar o modelo com parâmetros padrão
model = RandomForestRegressor(random_state=42) # Adicionado random_state para reprodutibilidade

# Treinar o modelo com os dados de treino, usando apenas a primeira coluna de y_train_scaled
model.fit(X_train_scaled, y_train_scaled.iloc[:, 0])

print("Modelo RandomForestRegressor treinado com sucesso!")


from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Previsões nos dados de treino
y_train_pred = model.predict(X_train_scaled)

# Avaliação nos dados de treino (usando apenas a primeira coluna de y_train_scaled)
rmse_train = np.sqrt(root_mean_squared_error(y_train_scaled.iloc[:, 0], y_train_pred))
mae_train = mean_absolute_error(y_train_scaled.iloc[:, 0], y_train_pred)
r2_train = r2_score(y_train_scaled.iloc[:, 0], y_train_pred)

print("Métricas de avaliação nos dados de treino:")
print(f"  RMSE: {rmse_train:.4f}")
print(f"  MAE: {mae_train:.4f}")
print(f"  R²: {r2_train:.4f}")

# Previsões nos dados de validação
y_val_pred = model.predict(X_val_scaled)

# Avaliação nos dados de validação (usando apenas a primeira coluna de y_val_scaled)
rmse_val = np.sqrt(root_mean_squared_error(y_val_scaled.iloc[:, 0], y_val_pred))
mae_val = mean_absolute_error(y_val_scaled.iloc[:, 0], y_val_pred)
r2_val = r2_score(y_val_scaled.iloc[:, 0], y_val_pred)

print("\nMétricas de avaliação nos dados de validação:")
print(f"  RMSE: {rmse_val:.4f}")
print(f"  MAE: {mae_val:.4f}")
print(f"  R²: {r2_val:.4f}")

# Define the hyperparameter grid for RandomForestRegressor
# Grid completo para busca de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'] # Number of features to consider when looking for the best split
}

# Grid reduzido para teste rápido
#param_grid = {
#    'n_estimators': [10],
#    'max_depth': [5],
#    'min_samples_split': [2],
#    'min_samples_leaf': [1],
#    'max_features': ['sqrt']
#}

print("Hyperparameter grid defined successfully:")
print(param_grid)

from sklearn.model_selection import GridSearchCV

# Define the number of folds for cross-validation
cv_folds = 5

print(f"Cross-validation folds set to: {cv_folds}")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from IPython.display import display # Import display

# Instantiate a RandomForestRegressor model with random_state=42
rf = RandomForestRegressor(random_state=42)

# Instantiate a GridSearchCV object
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=cv_folds,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)

# Fit the GridSearchCV object to the training data
# Use only the first column of y_train_scaled as the target
grid_search.fit(X_train_scaled, y_train_scaled.iloc[:, 0])

print("GridSearchCV completed.")

# a) Registre as combinações testadas e as métricas de validação correspondentes.
# Access the cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)

# Select relevant columns: parameters and mean test score (negative MSE)
# Convert negative MSE to positive MSE or RMSE for better interpretation
cv_results['mean_test_rmse'] = np.sqrt(-cv_results['mean_test_score'])

# Display the results, sorted by mean test RMSE
print("\nResultados da busca por hiperparâmetros (ordenados por RMSE de validação):")
display(cv_results[['params', 'mean_test_rmse', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']].sort_values(by='mean_test_rmse').head())


# b) Escolha o melhor conjunto de hiperparâmetros
best_params = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_) # Convert negative MSE to RMSE

print(f"\nMelhor conjunto de hiperparâmetros encontrado:")
print(best_params)
print(f"\nMelhor RMSE de validação (neg_mean_squared_error): {best_score:.4f}")

# Treino do modelo com os melhores hyperparametros com a combinação de dados acima
best_model = grid_search.best_estimator_

# Avalia no conjunto de validação
y_final = best_model.predict(X_val_scaled)

y_pred = np.ravel(y_final)
y_true = np.ravel(y_val_scaled)

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
