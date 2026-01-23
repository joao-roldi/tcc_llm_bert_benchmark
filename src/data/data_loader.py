"""
=============================================================================
Módulo de Carregamento de Dados
=============================================================================
Funções para carregar e preparar os datasets de fake news em português.
"""

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from pathlib import Path
from typing import Optional
from loguru import logger

from config import DATASETS, PATHS


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Carrega um dataset de fake news.
    
    Args:
        dataset_name: Nome do dataset ('fakebr' ou 'fakerecogna')
    
    Returns:
        DataFrame com colunas 'text' e 'label'
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado. Opções: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_name]
    
    logger.info(f"Carregando dataset: {config['name']}")
    
    # Tentar carregar do Hugging Face
    try:
        dataset = hf_load_dataset(config["huggingface"], split="train")
        df = dataset.to_pandas()
    except Exception as e:
        logger.error(f"Erro ao carregar do Hugging Face: {e}")
        logger.info("Tentando carregar de arquivo local...")
        df = load_local_dataset(dataset_name)

    df = standardize_columns(df, config)
    
    logger.info(f"Dataset carregado: {len(df)} amostras")
    
    return df


def load_local_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Carrega dataset de arquivo local.
    
    Args:
        dataset_name: Nome do dataset
    
    Returns:
        DataFrame com os dados
    """
    data_dir = Path(PATHS["data_dir"])
    
    # Tentar diferentes formatos
    for ext in [".csv", ".json", ".parquet"]:
        filepath = data_dir / f"{dataset_name}{ext}"
        if filepath.exists():
            if ext == ".csv":
                return pd.read_csv(filepath)
            elif ext == ".json":
                return pd.read_json(filepath)
            elif ext == ".parquet":
                return pd.read_parquet(filepath)
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' não encontrado em {data_dir}. "
        f"Por favor, baixe o dataset manualmente."
    )


def standardize_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Padroniza as colunas do DataFrame.
    
    Args:
        df: DataFrame original
        config: Configuração do dataset
    
    Returns:
        DataFrame com colunas padronizadas ('text', 'label')
    """
    # Renomear colunas
    df = df.rename(columns={
        config["text_column"]: "text",
        config["label_column"]: "label"
    }, errors='ignore')

    return df


def prepare_test_set(
    df: pd.DataFrame,
    sample_size: int = 1000,
    seed: int = 42,
    balanced: bool = True
) -> pd.DataFrame:
    """
    Prepara o conjunto de teste.
    
    Args:
        df: DataFrame completo
        sample_size: Número total de amostras
        seed: Seed para reprodutibilidade
        balanced: Se True, mantém classes balanceadas
    
    Returns:
        DataFrame com o conjunto de teste
    """
    if balanced:
        # Amostrar igual número de cada classe
        samples_per_class = sample_size // 2
        
        df_fake = df[df["label"] == 1].sample(
            n=min(samples_per_class, len(df[df["label"] == 1])),
            random_state=seed
        )
        df_true = df[df["label"] == 0].sample(
            n=min(samples_per_class, len(df[df["label"] == 0])),
            random_state=seed
        )
        
        test_df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=seed)
    else:
        # Amostragem aleatória simples
        test_df = df.sample(n=min(sample_size, len(df)), random_state=seed)
    
    return test_df.reset_index(drop=True)


def get_few_shot_examples(
    df: pd.DataFrame,
    n_examples: int = 3,
    exclude_indices: Optional[list] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Obtém exemplos para few-shot learning.
    
    Args:
        df: DataFrame completo
        n_examples: Número de exemplos (será dividido entre classes)
        exclude_indices: Índices a excluir (ex: conjunto de teste)
        seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame com exemplos balanceados
    """
    if exclude_indices:
        df = df[~df.index.isin(exclude_indices)]
    
    # Garantir exemplos de ambas as classes
    n_per_class = max(1, n_examples // 2)
    
    examples_fake = df[df["label"] == 1].sample(n=n_per_class, random_state=seed)
    examples_true = df[df["label"] == 0].sample(n=n_per_class, random_state=seed)
    
    # Intercalar exemplos
    examples = pd.concat([examples_true, examples_fake]).sample(frac=1, random_state=seed)
    
    return examples


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas do dataset.
    
    Args:
        df: DataFrame do dataset
    
    Returns:
        Dicionário com estatísticas
    """
    return {
        "total_samples": len(df),
        "fake_samples": len(df[df["label"] == 1]),
        "true_samples": len(df[df["label"] == 0]),
        "avg_text_length": df["text"].str.len().mean(),
        "min_text_length": df["text"].str.len().min(),
        "max_text_length": df["text"].str.len().max(),
    }
