"""
=============================================================================
Configurações do Benchmark
=============================================================================
Este arquivo contém todas as configurações do experimento, incluindo:
- Modelos a serem testados
- Datasets disponíveis
- Estratégias de prompting
- Parâmetros de geração
"""

from pathlib import Path

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# MODELOS DE LINGUAGEM
# =============================================================================
# Modelos selecionados com base no Portuguese LLM Leaderboard
# https://huggingface.co/collections/eduagarcia/portuguese-llm-leaderboard-best-models

MODELS = [
    # Modelos Pequenos (< 16GB VRAM)
    "Qwen/Qwen2-1.5B-Instruct",
    "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    "maritaca-ai/sabia-7b",
    
    # Modelos Médios (< 24GB VRAM)
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "lucianosb/boto-9B-it",
]

# Configurações específicas por modelo (opcional)
MODEL_CONFIGS = {
    "Qwen/Qwen2-1.5B-Instruct": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
    "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
    "maritaca-ai/sabia-7b": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
    "Qwen/Qwen2-7B-Instruct": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
    "lucianosb/boto-9B-it": {
        "max_new_tokens": 256,
        "quantization": "4bit",
    },
}

# Mapeamento de modelos HuggingFace para Ollama
# Formato: "huggingface_model_name": "ollama_model_name"
OLLAMA_MODEL_MAPPING = {
    "Qwen/Qwen2-1.5B-Instruct": "qwen2:1.5b-instruct",
    "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it": "brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16",
    "maritaca-ai/sabia-7b": "hf.co/TheBloke/sabia-7B-GGUF:latest",
    "Qwen/Qwen2-7B-Instruct": "qwen2:7b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3:8b-instruct-q8_0",
}

# =============================================================================
# DATASETS
# =============================================================================

DATASETS = {
    "fakebr": {
        "name": "Fake.Br Corpus",
        "source": "https://github.com/roneysco/Fake.br-Corpus",
        "huggingface": "fake-news-UFG/fakebr",
        "text_column": "text",
        "label_column": "label",
        "label_mapping": {"fake": 1, "true": 0},
    },
    "fakerecogna": {
        "name": "FakeRecogna",
        "source": "https://github.com/Gabriel-Lino-Garcia/FakeRecogna",
        "huggingface": "recogna-nlp/fakerecogna2-extrativa",
        "text_column": "Noticia",
        "label_column": "Label",
        "label_mapping": {"fake": 1, "real": 0},
    },
}

# =============================================================================
# ESTRATÉGIAS DE PROMPTING
# =============================================================================

PROMPTING_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]

# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================

EXPERIMENT_CONFIG = {
    # Número de amostras para teste
    "test_sample_size": 1000,
    
    # Número de exemplos para few-shot
    "few_shot_examples": 3,
    
    # Seed para reprodutibilidade
    "random_seed": 42,
    
    # Parâmetros de geração
    "generation_params": {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
    },
    
    # Timeout para inferência (segundos)
    "inference_timeout": 120,
}

# =============================================================================
# CONFIGURAÇÕES DE HARDWARE
# =============================================================================

HARDWARE_CONFIG = {
    # Usar GPU se disponível
    "use_cuda": False,
    
    # Usar quantização para reduzir uso de memória
    "use_quantization": False,
    "quantization_bits": 4,
    
    # Usar Flash Attention 2 se disponível
    "use_flash_attention": False,
    
    # Batch size para inferência (1 para evitar problemas de memória)
    "batch_size": 1,
}

# =============================================================================
# CAMINHOS DE ARQUIVOS (Cookiecutter Structure)
# =============================================================================

PATHS = {
    "data_dir": str(PROJECT_ROOT / "data" / "raw"),
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "models_dir": str(PROJECT_ROOT / "models"),
    "results_dir": str(PROJECT_ROOT / "reports"),
    "figures_dir": str(PROJECT_ROOT / "reports" / "figures"),
    "cache_dir": str(PROJECT_ROOT / "data" / "external"),
    "logs_dir": str(PROJECT_ROOT / "reports" / "logs"),
}
