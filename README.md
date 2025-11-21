# SentimentML
Pipeline de Machine Learning para análise de sentimentos em avaliações de produtos de e-commerce. Abrange coleta (Kaggle), preparação de dados, treinamento, avaliação e inferência, com enfoque em práticas de MLOps.

## Desafio Proposto
O projeto atende ao desafio final da trilha de Machine Learning Engineering, cujo objetivo é implementar um pipeline de MLOps para análise de sentimentos em avaliações de produtos de e-commerce. O trabalho deve contemplar obtenção dos dados (Kaggle), análise exploratória, seleção e preparação das amostras, treinamento e avaliação do modelo, além da construção de um serviço de inferência e definição de estratégias de monitoramento em produção. A entrega inclui um repositório completo e um vídeo explicativo com até 15 minutos. [Documentação](docs/Projeto%20Final%20Tutorial%20MLE-2.pdf)

## Visão geral
- Baixa o dataset `kritanjalijain/amazon-reviews` via `kagglehub`.
- Lê o CSV (`train.csv` ou `test.csv`) e retorna dataframe com colunas `label`, `title`, `comment`.
- Notebook exploratório em `eda.ipynb` para inspeção rápida dos dados.

## Requisitos
- Python 3.10+


## Configuração rápida
```bash
# Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Como rodar o notebook exploratório
- Garanta o venv ativo (`source venv/bin/activate`) e dependências instaladas.
- Abra `eda.ipynb` no VS Code ou Jupyter (ex.: `jupyter notebook eda.ipynb`).
- Se precisar registrar o kernel do venv: `python -m ipykernel install --user --name sentimentml`.
