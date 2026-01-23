"""
=============================================================================
Módulo de Métricas e Avaliação
=============================================================================
Funções para calcular métricas de desempenho e gerar relatórios.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int] = [0, 1]
) -> Dict[str, float]:
    """
    Calcula métricas de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        labels: Lista de labels possíveis
    
    Returns:
        Dicionário com métricas
    """
    # Tratar predições inválidas (-1) como erros
    y_pred_clean = [p if p in labels else 1 - y for p, y in zip(y_pred, y_true)]
    
    # Contar predições inválidas
    invalid_predictions = sum(1 for p in y_pred if p not in labels)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_clean),
        "precision": precision_score(y_true, y_pred_clean, average="binary", pos_label=1),
        "recall": recall_score(y_true, y_pred_clean, average="binary", pos_label=1),
        "f1_score": f1_score(y_true, y_pred_clean, average="binary", pos_label=1),
        "precision_macro": precision_score(y_true, y_pred_clean, average="macro"),
        "recall_macro": recall_score(y_true, y_pred_clean, average="macro"),
        "f1_macro": f1_score(y_true, y_pred_clean, average="macro"),
        "invalid_predictions": invalid_predictions,
        "invalid_rate": invalid_predictions / len(y_pred) if len(y_pred) > 0 else 0,
    }
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred_clean, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])
    
    return metrics


def generate_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path
) -> pd.DataFrame:
    """
    Gera relatório consolidado de todos os experimentos.
    
    Args:
        all_results: Lista com resultados de todos os experimentos
        output_dir: Diretório de saída
    
    Returns:
        DataFrame com o relatório
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extrair métricas principais
    report_data = []
    for result in all_results:
        if "error" in result:
            row = {
                "model": result["model"],
                "strategy": result["strategy"],
                "dataset": result["dataset"],
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "avg_inference_time": None,
                "vram_usage_gb": None,
                "error": result["error"]
            }
        else:
            metrics = result["metrics"]
            row = {
                "model": result["model"],
                "strategy": result["strategy"],
                "dataset": result["dataset"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "avg_inference_time": metrics.get("avg_inference_time"),
                "vram_usage_gb": metrics.get("vram_usage_gb"),
                "invalid_rate": metrics.get("invalid_rate"),
                "error": None
            }
        report_data.append(row)
    
    # Criar DataFrame
    df_report = pd.DataFrame(report_data)
    
    # Salvar em diferentes formatos
    df_report.to_csv(output_dir / "benchmark_report.csv", index=False)
    df_report.to_excel(output_dir / "benchmark_report.xlsx", index=False)
    
    # Gerar relatório em Markdown
    generate_markdown_report(df_report, all_results, output_dir)
    
    # Imprimir resumo
    logger.info("RELATÓRIO CONSOLIDADO")
    logger.info("\n" + df_report.to_string(index=False))
    
    # Identificar melhores resultados
    if not df_report["f1_score"].isna().all():
        best_idx = df_report["f1_score"].idxmax()
        best = df_report.loc[best_idx]
        logger.info(f"🏆 Melhor resultado:")
        logger.info(f"   Modelo: {best['model']}")
        logger.info(f"   Estratégia: {best['strategy']}")
        logger.info(f"   Dataset: {best['dataset']}")
        logger.info(f"   F1-Score: {best['f1_score']:.4f}")
    
    return df_report


def generate_markdown_report(
    df_report: pd.DataFrame,
    all_results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Gera relatório detalhado em Markdown.
    
    Args:
        df_report: DataFrame com métricas
        all_results: Lista completa de resultados
        output_dir: Diretório de saída
    """
    md_content = """# Relatório do Benchmark - LLMs para Detecção de Fake News

## Resumo Executivo

Este relatório apresenta os resultados do benchmark comparativo de Large Language Models (LLMs) 
locais para a tarefa de detecção de fake news em português brasileiro.

## Resultados Gerais

### Tabela de Métricas

"""
    
    # Formatar tabela
    df_display = df_report.copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "invalid_rate"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    
    md_content += df_display.to_markdown(index=False)
    
    # Análise por modelo
    md_content += "\n\n## Análise por Modelo\n\n"
    
    for model in df_report["model"].unique():
        model_data = df_report[df_report["model"] == model]
        avg_f1 = model_data["f1_score"].mean()
        
        md_content += f"### {model}\n\n"
        md_content += f"- **F1-Score Médio:** {avg_f1:.4f}\n"
        md_content += f"- **Melhor Estratégia:** {model_data.loc[model_data['f1_score'].idxmax(), 'strategy']}\n"
        md_content += "\n"
    
    # Análise por estratégia
    md_content += "\n## Análise por Estratégia de Prompting\n\n"
    
    for strategy in df_report["strategy"].unique():
        strategy_data = df_report[df_report["strategy"] == strategy]
        avg_f1 = strategy_data["f1_score"].mean()
        
        md_content += f"### {strategy.replace('_', ' ').title()}\n\n"
        md_content += f"- **F1-Score Médio:** {avg_f1:.4f}\n"
        md_content += f"- **Melhor Modelo:** {strategy_data.loc[strategy_data['f1_score'].idxmax(), 'model']}\n"
        md_content += "\n"
    
    # Conclusões
    md_content += """
## Conclusões

Os resultados demonstram que [análise a ser completada após execução dos experimentos].

## Metodologia

- **Modelos testados:** """ + ", ".join(df_report["model"].unique()) + """
- **Estratégias de prompting:** """ + ", ".join(df_report["strategy"].unique()) + """
- **Datasets:** """ + ", ".join(df_report["dataset"].unique()) + """

---
*Relatório gerado automaticamente pelo benchmark de LLMs para Fake News.*
"""
    
    # Salvar
    with open(output_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    logger.info(f"Relatório Markdown salvos em: {output_dir / 'benchmark_report.md'}")


def compare_models(
    results: List[Dict[str, Any]],
    metric: str = "f1_score"
) -> pd.DataFrame:
    """
    Compara modelos por uma métrica específica.
    
    Args:
        results: Lista de resultados
        metric: Métrica para comparação
    
    Returns:
        DataFrame com comparação
    """
    data = []
    for result in results:
        if "error" not in result:
            data.append({
                "model": result["model"],
                "strategy": result["strategy"],
                "dataset": result["dataset"],
                metric: result["metrics"][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Pivot para visualização
    pivot = df.pivot_table(
        values=metric,
        index="model",
        columns=["strategy", "dataset"],
        aggfunc="mean"
    )
    
    return pivot


def statistical_analysis(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Realiza análise estatística dos resultados.
    
    Args:
        results: Lista de resultados
    
    Returns:
        Dicionário com análises estatísticas
    """
    # Extrair F1-scores
    f1_scores = {}
    for result in results:
        if "error" not in result:
            key = f"{result['model']}_{result['strategy']}"
            if key not in f1_scores:
                f1_scores[key] = []
            f1_scores[key].append(result["metrics"]["f1_score"])
    
    # Calcular estatísticas
    stats = {}
    for key, scores in f1_scores.items():
        stats[key] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "n": len(scores)
        }
    
    return stats
