# PIX Detection Fraud

Projeto de detecção de fraudes em transações PIX utilizando Machine Learning com pipeline automatizado via Prefect e rastreamento de experimentos com MLflow.

## Sobre o Projeto

Sistema de ML para identificar transações fraudulentas, implementando boas práticas de MLOps incluindo orquestração de pipelines, versionamento de modelos e rastreamento de métricas.

## Tecnologias Utilizadas

- **Python 3.x**
- **Prefect** - Orquestração de pipelines
- **MLflow** - Rastreamento de experimentos e versionamento de modelos
- **Scikit-learn** - Algoritmos de ML (RandomForestClassifier)
- **Joblib** - Serialização de modelos
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica

## Estrutura do Projeto

```
pix-detection-fraud/
├── artifacts/          # Modelos e pipelines salvos
├── data/              # Dados brutos e processados
├── notebooks/         # EDA, POC fe e model selection
├── pipelines/         # Pipelines 
│   └── train_pipe.py # pipeline de treinamento
|    └── retrain_pipe.py  #Pipeline de retreino de modelo
|    └── monitoring_pipe.py   # pipeline de monitoramento de modelo
├── src/
│   ├── data/         # Processamento de dados
|       └── generate_data.py    # Gera dados sintéticos de clientes e transações
|       └── split_data.py     # Função que splita os dados para treinamento
│   ├── features/     # Feature engineering
|       └── transformers/       # transformadores pipeline
|            └── create_fe.py    # cria novas features
|             └── drop_fe.py      # dropa colunas
|             └── select_cols     # filtra colunas para fe
|        └── new_features.py    # Strategy que cria funções de create feature
|        └── build_pipe_fe.py   # Função que implementa toda a pipeline de preprocessamento
│   └── models/       # Treinamento e avaliação
|        └── train.py    # Função que une pipeline fe com o modelo
|        └── predict.py  # Função que faz predição com novos dados
|        └── evaluator.py # Função que retorna as métricas
├── tests/   # File com testes unitários
├── requirements.txt
└── README.md
```

## Como Executar

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/pix-detection-fraud.git
cd pix-detection-fraud

# Instale as dependências
pip install -r requirements.txt
```

### 2. Iniciar MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Executar o Pipeline de Treinamento

```bash
python pipelines/train_pipe.py
```

### 4. Visualizar Experimentos

Acesse: `http://localhost:5000`

## Pipeline de Treinamento

O pipeline executa as seguintes etapas:

1. **get_data**: Carrega e divide os dados em treino/teste
2. **builder_pipe_train**: Constrói pipeline de feature engineering e treina o modelo
3. **predict_model**: Realiza predições e calcula métricas de avaliação

### Modelo Atual
Melhor modelo de acordo com o POC em notebooks, utilizando GridSearchCV
- **Algoritmo**: Random Forest Classifier
- **Hiperparâmetros**:
  - n_estimators: 200
  - max_depth: None
  - min_samples_split: 2
  - random_state: 42

## Métricas Rastreadas

- Acurácia
- Precisão
- Recall
- F1-Score
- AUC-ROC

##  Próximos Passos

### CI/CD
- [ ] Configurar GitHub Actions para testes automatizados
- [ ] Implementar deploy automático do modelo
- [ ] Criar pipeline de validação de dados

### Retreinamento
- [ ] Agendar retreinamento periódico com Prefect
- [ ] Implementar detecção de drift nos dados
- [x] Criar estratégia de versionamento de modelos

### Monitoramento
- [ ] Integrar Evidently AI para monitoramento de drift
- [ ] Criar dashboards de performance em produção
- [ ] Implementar alertas de degradação do modelo

## 📝 Licença

Este é um projeto de portfólio para fins educacionais.

---

⭐ Se este projeto foi útil, considere dar uma estrela!