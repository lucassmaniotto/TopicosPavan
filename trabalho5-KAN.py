"""
Script de comparação de modelos de regressão (KAN, MLP e Random Forest).

Fluxo geral:
- Carrega dados pré-processados (train/val/test) do diretório `content`.
- Usa Optuna para ajustar hiperparâmetros do KAN com early stopping simples.
- Treina o KAN final com os melhores hiperparâmetros.
- Treina baselines (MLP e RandomForest) para comparação.
- Reporta métricas e mostra gráficos de resíduos e de dispersão y_true vs y_pred.
"""

from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
from kan import KAN

from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

import optuna


def load_data():
    """Carrega e alinha os conjuntos de dados (train/val/test) em float32."""
    base = Path(__file__).resolve().parent / "content"
    if not base.exists():
        base = Path(__file__).resolve().parents[1] / "content"

    files = {
        "X_train": base / "X_train_scaled.pkl",
        "y_train": base / "y_train_scaled.pkl",
        "X_test": base / "X_test_scaled.pkl",
        "y_test": base / "y_test_scaled.pkl",
        "X_val": base / "X_val_scaled.pkl",
        "y_val": base / "y_val_scaled.pkl",
    }

    X_train = np.asarray(joblib.load(files["X_train"]), dtype=np.float32)
    y_train = np.asarray(joblib.load(files["y_train"]), dtype=np.float32)
    X_test = np.asarray(joblib.load(files["X_test"]), dtype=np.float32)
    y_test = np.asarray(joblib.load(files["y_test"]), dtype=np.float32)
    X_val = np.asarray(joblib.load(files["X_val"]), dtype=np.float32)
    y_val = np.asarray(joblib.load(files["y_val"]), dtype=np.float32)

    # Garante dimensão correta do alvo (vetor 1D)
    y_train = np.ravel(y_train).astype(np.float32)
    y_test = np.ravel(y_test).astype(np.float32)
    y_val = np.ravel(y_val).astype(np.float32)

    # Alinha comprimentos X/y por split (evita mismatch)
    n_tr = min(len(X_train), len(y_train))
    X_train = X_train[:n_tr]
    y_train = y_train[:n_tr]
    n_te = min(len(X_test), len(y_test))
    X_test = X_test[:n_te]
    y_test = y_test[:n_te]
    n_va = min(len(X_val), len(y_val))
    X_val = X_val[:n_va]
    y_val = y_val[:n_va]

    return X_train, y_train, X_test, y_test, X_val, y_val


def build_kan(width, grid, k, device):
    """Cria o modelo KAN com largura, grade e k especificados."""
    return KAN(
        width=width,
        grid=grid,
        k=k,
        seed=42,
        device=device,
    )


def kan_objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim, steps_base=50):
    """Objetivo do Optuna: treina o KAN em blocos e avalia por RMSE de validação.

    Usa early stopping simples (paciência fixa) para interromper se não houver melhora.
    Retorna o negativo do RMSE (maior é melhor para o Optuna).
    """
    # Espaço de busca dos hiperparâmetros
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    hidden_units = trial.suggest_int("hidden_units", max(1, input_dim), 32)
    grid_size = trial.suggest_int("grid_size", 5, 20)
    k_spline = trial.suggest_categorical("k", [2, 3])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "Nadam", "LBFGS"])
    l2_reg = trial.suggest_float("l2", 1e-6, 1e-2, log=True)

    # Define largura conforme exemplo: [in, hidden..., out]
    width = [input_dim] + [hidden_units] * n_hidden_layers + [output_dim]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_kan(width=width, grid=grid_size, k=k_spline, device=device)

    # Prepara tensores do dataset para o KAN (em device ativo)
    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32, device=device),
        "train_output": torch.tensor(y_train[:, None], dtype=torch.float32, device=device),
        "train_label": torch.tensor(y_train[:, None], dtype=torch.float32, device=device),
        "val_input": torch.tensor(X_val, dtype=torch.float32, device=device),
        "val_output": torch.tensor(y_val[:, None], dtype=torch.float32, device=device),
        "val_label": torch.tensor(y_val[:, None], dtype=torch.float32, device=device),
        "test_input": torch.tensor(X_test, dtype=torch.float32, device=device),
        "test_output": torch.tensor(y_test[:, None], dtype=torch.float32, device=device),
        "test_label": torch.tensor(y_test[:, None], dtype=torch.float32, device=device),
    }

    # Early stopping simples: treina em blocos e monitora RMSE de validação
    best_val = None
    best_steps = 0
    total_steps = steps_base
    patience = 3
    remain = patience

    # KAN.fit aceita: steps (épocas), opt (otimizador), lamb (L2), lr (taxa de aprendizado)
    # Treinamos em blocos menores (chunks) para verificar melhoria e parar cedo
    chunk = 10
    steps_done = 0
    while steps_done < total_steps:
        steps_now = min(chunk, total_steps - steps_done)
        try:
            model.fit(dataset, opt=optimizer, steps=steps_now, lamb=l2_reg, lr=learning_rate)
        except Exception as exc:
            trial.set_user_attr("kan_error", str(exc))
            return -1e9

        # Avaliação rápida na validação
        with torch.no_grad():
            y_hat = model(dataset["val_input"]).detach().cpu().numpy().squeeze()
        # Alinha comprimentos se necessário e usa -RMSE como métrica para o Optuna
        n = min(len(y_val), len(y_hat))
        rmse = root_mean_squared_error(y_val[:n], y_hat[:n])
        val_score = -rmse

        if (best_val is None) or (val_score > best_val):
            best_val = val_score
            best_steps = steps_done + steps_now
            remain = patience
        else:
            remain -= 1
            if remain <= 0:
                break
        steps_done += steps_now

    # Retorna melhor valor alcançado (ou penalização em caso de falha)
    return best_val if best_val is not None else -1e9


def evaluate_regression(y_true, y_pred):
    """Calcula métricas MAE, RMSE e R2 e retorna também resíduos e vetores."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # Alinha tamanhos se necessário
    n = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "residuals": y_pred - y_true,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_residuals(residuals, title_prefix=""):
    """Plota série de resíduos e histograma para rápida inspeção."""
    plt.figure(figsize=(8, 4))
    plt.plot(residuals, marker="o", linestyle="None", markersize=3)
    plt.axhline(0, color="red", linewidth=1)
    plt.xlabel("Amostra")
    plt.ylabel("Erro (y_pred - y_true)")
    plt.title(f"{title_prefix} Resíduos: diferença entre predição e valor real")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Erro (y_pred - y_true)")
    plt.title(f"{title_prefix} Histograma dos resíduos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    input_dim = X_train.shape[1]
    output_dim = 1

    # Ajuste de hiperparâmetros (Optuna) para o KAN
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: kan_objective(
            t,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            input_dim,
            output_dim,
            # Passos base menores aceleram os trials; early stop ajuda a parar cedo
            steps_base=30,
        ),
        n_trials=10,
        # Executa trials em paralelo para acelerar a busca
        n_jobs=4
    )

    print("Best KAN trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Treina KAN final com melhores parâmetros e avalia em validação
    bp = study.best_trial.params
    width = [input_dim] + [bp["hidden_units"]] * bp["n_hidden_layers"] + [output_dim]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kan_model = build_kan(width, bp["grid_size"], bp["k"], device)
    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32, device=device),
        "train_output": torch.tensor(y_train[:, None], dtype=torch.float32, device=device),
        "train_label": torch.tensor(y_train[:, None], dtype=torch.float32, device=device),
        "val_input": torch.tensor(X_val, dtype=torch.float32, device=device),
        "val_output": torch.tensor(y_val[:, None], dtype=torch.float32, device=device),
        "val_label": torch.tensor(y_val[:, None], dtype=torch.float32, device=device),
        "test_input": torch.tensor(X_test, dtype=torch.float32, device=device),
        "test_output": torch.tensor(y_test[:, None], dtype=torch.float32, device=device),
        "test_label": torch.tensor(y_test[:, None], dtype=torch.float32, device=device),
    }
    # Treinamento final com menos passos para terminar mais rápido (ajustável)
    kan_model.fit(dataset, opt=bp["optimizer"], steps=30, lamb=bp["l2"], lr=bp["learning_rate"])
    with torch.no_grad():
        y_val_pred_kan = kan_model(torch.tensor(X_val, dtype=torch.float32, device=device)).detach().cpu().numpy().squeeze()

    metrics_kan = evaluate_regression(y_val, y_val_pred_kan)
    print(f"KAN -> MAE: {metrics_kan['MAE']:.4f} | RMSE: {metrics_kan['RMSE']:.4f} | R2: {metrics_kan['R2']:.4f}")
    plot_residuals(metrics_kan["residuals"], title_prefix="KAN:")

    # Baselines: MLP e RandomForest para comparar desempenho
    # MLP com parâmetros padrão razoáveis para comparação rápida
    mlp = MLPRegressor(
        hidden_layer_sizes=(max(8, input_dim), max(8, input_dim // 2)),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=200,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    y_val_pred_mlp = mlp.predict(X_val)
    metrics_mlp = evaluate_regression(y_val, y_val_pred_mlp)
    print(f"MLP -> MAE: {metrics_mlp['MAE']:.4f} | RMSE: {metrics_mlp['RMSE']:.4f} | R2: {metrics_mlp['R2']:.4f}")
    plot_residuals(metrics_mlp["residuals"], title_prefix="MLP:")

    # Random Forest (usa todos os núcleos de CPU)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        # use all cores for faster RF training
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_val_pred_rf = rf.predict(X_val)
    metrics_rf = evaluate_regression(y_val, y_val_pred_rf)
    print(f"RF  -> MAE: {metrics_rf['MAE']:.4f} | RMSE: {metrics_rf['RMSE']:.4f} | R2: {metrics_rf['R2']:.4f}")
    plot_residuals(metrics_rf["residuals"], title_prefix="RandomForest:")

    # Gráfico de dispersão: y_true vs y_pred (cores/markers distintos para evitar sobreposição)
    plt.figure(figsize=(6, 6))
    # Distinct colors + markers + edgecolors to improve overlap visibility
    kan_color, mlp_color, rf_color = "#1f77b4", "#ff7f0e", "#2ca02c"  # blue, orange, green
    plt.scatter(
        metrics_kan["y_true"], metrics_kan["y_pred"],
        s=24, marker="o", label="KAN",
        alpha=0.7, c=kan_color, edgecolors="k", linewidths=0.3, zorder=3,
    )
    plt.scatter(
        metrics_mlp["y_true"], metrics_mlp["y_pred"],
        s=24, marker="s", label="MLP",
        alpha=0.7, c=mlp_color, edgecolors="k", linewidths=0.3, zorder=2,
    )
    plt.scatter(
        metrics_rf["y_true"], metrics_rf["y_pred"],
        s=24, marker="^", label="RF",
        alpha=0.7, c=rf_color, edgecolors="k", linewidths=0.3, zorder=1,
    )
    min_y = min(metrics_kan["y_true"].min(), metrics_mlp["y_true"].min(), metrics_rf["y_true"].min())
    max_y = max(metrics_kan["y_true"].max(), metrics_mlp["y_true"].max(), metrics_rf["y_true"].max())
    plt.plot([min_y, max_y], [min_y, max_y], color="#666666", linestyle="--", label="Ideal", zorder=0)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Comparação: y_true vs y_pred (Validação)")
    plt.legend(frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()