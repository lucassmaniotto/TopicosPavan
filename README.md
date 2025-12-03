# Atividade 3 - Modelagem com Random Forest

Aplicação do algoritmo Random Forest Regressor para construir um modelo capaz de prever os níveis do separador trifásico a partir dos
dados tratados e preparados na fase de pré-processamento.

O objetivo é treinar o modelo, otimizar seus hiperparâmetros, avaliar seu desempenho e interpretar os resultados obtidos. 

## Instruções para configurar o ambiente de desenvolvimento para Windows

### Criar o ambiente virtual
```bash
python -m venv ml_env
```

### Ativar o ambiente virtual
```bash
ml_env\Scripts\Activate.ps1
```

### Atualizar o pip
```bash
python -m pip install --upgrade pip
```

### Instalar bibliotecas necessárias
```bash
pip install scikit-learn pandas numpy ipython
pip install matplotlib seaborn
pip install torch pykan optuna
```

### Criar o arquivo Python
```bash
notepad trabalho3.py
```

### Cole os códigos fornecidos no colab abaixo no arquivo Python criado no passo anterior:
- https://colab.research.google.com/drive/1MUXLsF76OwWr8Ibrg1bAQ7IC1rntTkRC?usp=sharing

### Executar o arquivo Python
```bash
python trabalho3.py
```

-------------------------------------------------------------------------------------------

## Instruções para configurar o ambiente de desenvolvimento para Linux/Mac
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
```

### Criar o ambiente virtual
```bash
python3 -m venv ml_env
source ml_env/bin/activate
```

### Atualizar o pip
```bash
pip install --upgrade pip
pip install scikit-learn pandas numpy IPython

pip install matplotlib seaborn
```

### Criar o arquivo Python
```bash
nano trabalho3.py
```

### Cole os códigos fornecidos no colab abaixo no arquivo Python criado no passo anterior:
- https://colab.research.google.com/drive/1MUXLsF76OwWr8Ibrg1bAQ7IC1rntTkRC?usp=sharing

### Executar o arquivo Python
```bash
python3 trabalho3.py
```