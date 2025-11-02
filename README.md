# 🏀 Análise Preditiva NBA - Linear e Logistic Regression

Projeto desenvolvido para a disciplina de **Redes Neurais Artificiais** da Universidade Federal do Agreste de Pernambuco (UFAPE).

## 📋 Descrição

Este projeto implementa modelos de **Regressão Linear** e **Regressão Logística** para análise preditiva de desempenho do Los Angeles Lakers na temporada NBA 2024-25.

### Objetivos

**Parte 1 - Regressão Linear:**
- Modelar relações entre variáveis de desempenho
- Prever estatísticas numéricas (pontos, rebotes, assistências)
- Quantificar impacto de variáveis independentes
- Gerar visualizações de tendências e intervalos de confiança

**Parte 2 - Regressão Logística:**
- Prever probabilidade de vitória/derrota
- Calcular probabilidades específicas de resultado
- Identificar variáveis mais impactantes
- Avaliar performance através de métricas de classificação

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/linear-and-logistic-regression.git
cd linear-and-logistic-regression

# Instale as dependências
pip install -r requirements.txt
```

## 📊 Uso

### 1. Coletar dados da NBA API

```bash
python fetch_data.py
```

### 2. Executar a aplicação Streamlit

```bash
streamlit run app.py
```

### 3. Navegar pelas análises

- **Página Principal:** Visão geral e carregamento de dados
- **Regressão Linear:** Análise de variáveis numéricas
- **Regressão Logística:** Predição de vitória/derrota

## 📈 Funcionalidades

### Regressão Linear
- ✅ Seleção dinâmica de variáveis dependentes e independentes
- ✅ Equação de regressão com coeficientes
- ✅ Métricas: R², MSE, RMSE
- ✅ Gráficos:
  - Diagrama de dispersão com linha de regressão
  - Previsão vs. Realidade
  - Matriz de confusão adaptada
  - Tendência com intervalo de confiança
  - Análise de resíduos

### Regressão Logística
- ✅ Predição de probabilidade de vitória
- ✅ Equação logística (sigmoide)
- ✅ Métricas: Acurácia, Precisão, Recall, F1-Score
- ✅ Gráficos:
  - Curva ROC com AUC
  - Distribuição de probabilidades
  - Importância de variáveis
  - Curva sigmoide
  - Matriz de confusão
  - Tendência com intervalo de confiança

## 🛠️ Tecnologias

- **Python 3.8+**
- **Streamlit** - Interface web interativa
- **scikit-learn** - Modelos de machine learning
- **pandas** - Manipulação de dados
- **matplotlib/seaborn** - Visualizações
- **nba_api** - Coleta de dados da NBA

## 📁 Estrutura do Projeto

```
linear-and-logistic-regression/
├── app.py                          # Aplicação principal
├── fetch_data.py                   # Script de coleta de dados
├── requirements.txt                # Dependências
├── README.md                       # Documentação
├── data/                           # Dados coletados (CSV)
├── src/
│   ├── data_loader.py             # Carregamento da API
│   ├── data_preprocessing.py      # Limpeza e feature engineering
│   ├── data_saver.py              # Persistência em CSV
│   ├── models.py                  # Modelos de ML
│   └── plotting.py                # Funções de visualização
└── pages/
    ├── linear_regression.py       # Interface - Regressão Linear
    └── logistic_regression.py     # Interface - Regressão Logística
```