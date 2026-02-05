# LLM Benchmark para Detecção de Fake News em Português

Este projeto implementa um benchmark comparativo de Large Language Models (LLMs) locais para a tarefa de detecção de fake news em português brasileiro.

## 📋 Visão Geral

O benchmark avalia diferentes modelos de linguagem usando as seguintes estratégias:
- **Zero-Shot**: Classificação sem exemplos prévios
- **Few-Shot**: Classificação com 3 exemplos demonstrativos
- **Chain-of-Thought**: Classificação com raciocínio passo a passo
- **Fine-tuned**: BERT em português fine-tunado nos datasets (para comparação com os LLMs)

## 🚀 Instalação

### 1. Criar ambiente virtual

```bash
# Com venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Com conda
conda create -n llm-benchmark python=3.13
conda activate llm-benchmark
```

### 2. Instalar dependências

O projeto usa [uv](https://github.com/astral-sh/uv) e `pyproject.toml` (Python ≥3.13).

```bash
# Com uv (recomendado)
uv sync

# Ou com pip
pip install -e .
```

### 3. Instalar PyTorch com CUDA (para GPU)

Se for usar modelos Hugging Face locais com GPU:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Modelos Incluídos

| Categoria | Modelo (Hugging Face) | Parâmetros | VRAM Estimada |
|-----------|------------------------|------------|---------------|
| Pequeno | Qwen/Qwen2-1.5B-Instruct | 1.5B | ~4 GB |
| Pequeno | CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it | 4B | ~6 GB |
| Pequeno | maritaca-ai/sabia-7b | 7B | ~8 GB |
| Médio | Qwen/Qwen2-7B-Instruct | 7B | ~8 GB |
| Médio | meta-llama/Meta-Llama-3-8B-Instruct | 8B | ~10 GB |
| Médio | lucianosb/boto-9B-it | 9B | ~12 GB |

*Valores com quantização 4-bit*

## 📁 Datasets

O benchmark utiliza dois datasets de fake news em português:

1. **Fake.Br Corpus** (7.200 notícias)
   - Fonte: https://github.com/roneysco/Fake.br-Corpus

2. **FakeRecogna** (11.902 notícias)
   - Fonte: https://github.com/Gabriel-Lino-Garcia/FakeRecogna

Os datasets são baixados automaticamente do Hugging Face.

## 🎯 Uso

O fluxo de trabalho é baseado em **Jupyter notebooks**. Execute os notebooks a partir da raiz do projeto (`tcc2/`); eles configuram o `sys.path` para importar os módulos em `src/`.

### Fluxo recomendado

1. **`01_data_exploration.ipynb`** — Carrega e explora os datasets (Fake.Br e FakeRecogna).
2. **`02_single_model_test.ipynb`** — Testa um único modelo/estratégia antes do benchmark completo.
3. **`03_full_benchmark.ipynb`** — Executa o benchmark completo (todos os modelos × estratégias × datasets). Pode levar várias horas. Os resultados são salvos em `reports/*.json`.
4. **`03_train_bert.ipynb`** — Fine-tuning do BERT (neuralmind/bert-base-portuguese-cased) para comparação com os LLMs.
5. **`04_benchmark_analysis.ipynb`** — Agrega os JSONs, gera tabelas e o relatório em `reports/benchmark_report.*`.

### Configurações

Modelos, datasets, estratégias e parâmetros de experimento estão em **`src/config.py`** (`MODELS`, `DATASETS`, `PROMPTING_STRATEGIES`, `EXPERIMENT_CONFIG`). Os templates de prompt ficam em **`src/models/prompts.py`**.

## 📈 Métricas Avaliadas

### Métricas de Desempenho
- **Acurácia**: Percentual de classificações corretas
- **Precisão**: Taxa de verdadeiros positivos entre os preditos como fake
- **Recall**: Taxa de detecção de fake news
- **F1-Score**: Média harmônica entre precisão e recall

### Métricas Práticas
- **Tempo de Inferência**: Segundos por notícia
- **Uso de VRAM**: Memória de vídeo utilizada

## 📂 Estrutura do Projeto

```
tcc2/
├── main.py                 # Ponto de entrada (stub)
├── pyproject.toml          # Dependências e metadados
├── uv.lock                 # Lock file (uv)
├── README.md               # Este arquivo
├── src/
│   ├── config.py           # Modelos, datasets, estratégias, paths
│   ├── data/
│   │   └── data_loader.py  # Carregamento de dados (Hugging Face)
│   └── models/
│       ├── model_handler.py  # ModelHandler (HF) e ModelHandlerOllama
│       ├── prompts.py       # Templates de prompts
│       └── metrics.py       # Cálculo de métricas e relatório
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_model_test.ipynb
│   ├── 03_full_benchmark.ipynb
│   ├── 03_train_bert.ipynb   # Fine-tuning BERT
│   └── 04_benchmark_analysis.ipynb
├── reports/                # Resultados (gerados pelos notebooks)
│   ├── benchmark_report.csv
│   ├── benchmark_report.xlsx
│   ├── benchmark_report.md
│   ├── figures/
│   └── *.json               # Resultados por modelo/estratégia/dataset
├── models/                  # Checkpoints locais (ex.: BERT)
└── references/              # Referências
```

## 🔧 Configuração Avançada

### Modificar modelos

Edite **`src/config.py`** para adicionar ou remover modelos e o mapeamento para Ollama:

```python
MODELS = [
    "seu-modelo/nome",
    # ...
]

OLLAMA_MODEL_MAPPING = {
    "huggingface/model-name": "ollama:model-name",
    # ...
}
```

### Modificar prompts

Edite **`src/models/prompts.py`** para customizar os templates (zero-shot, few-shot, chain-of-thought).

### Usar Ollama

O benchmark usa **Ollama** por padrão nos notebooks (via `ModelHandlerOllama`). Certifique-se de que o Ollama está instalado e que os modelos listados em `OLLAMA_MODEL_MAPPING` estão disponíveis. Nos notebooks, após configurar `sys.path` com `src/`:

```python
from models.model_handler import ModelHandlerOllama

# Nome do modelo no Hugging Face; o handler usa OLLAMA_MODEL_MAPPING
handler = ModelHandlerOllama("meta-llama/Meta-Llama-3-8B-Instruct")
response = handler.generate(prompt)
```

## ⚠️ Requisitos de Hardware

| Configuração | VRAM | Modelos Suportados |
|--------------|------|-------------------|
| Mínima | 8 GB | Qwen2-1.5B, Gemma-4B |
| Recomendada | 16 GB | Todos até 8B |
| Ideal | 24 GB | Todos os modelos |

## 📝 Citação

Se utilizar este código em sua pesquisa, por favor cite:

```bibtex
@misc{llm_fake_news_benchmark,
  author = {João},
  title = {LLM Benchmark para Detecção de Fake News em Português},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/joao-roldi/tcc_llm_bert_benchmark}
}
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.
