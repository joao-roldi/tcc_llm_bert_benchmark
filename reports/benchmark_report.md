# Relatório do Benchmark - LLMs para Detecção de Fake News

## Resumo Executivo

Este relatório apresenta os resultados do benchmark comparativo de Large Language Models (LLMs) 
locais para a tarefa de detecção de fake news em português brasileiro.

## Resultados Gerais

### Tabela de Métricas

| model                                 | strategy         | dataset     |   accuracy |   precision |   recall |   f1_score |   avg_inference_time |   vram_usage_gb |   invalid_rate | error   |
|:--------------------------------------|:-----------------|:------------|-----------:|------------:|---------:|-----------:|---------------------:|----------------:|---------------:|:--------|
| Qwen/Qwen2-7B-Instruct                | few_shot         | fakebr      |      0.361 |      0.0983 |    0.034 |     0.0505 |            1.42318   |               2 |          0     |         |
| Qwen/Qwen2-7B-Instruct                | zero_shot        | fakerecogna |      0.788 |      0.9186 |    0.632 |     0.7488 |            0.619561  |               2 |          0     |         |
| maritaca-ai/sabia-7b                  | zero_shot        | fakebr      |      0.193 |      0.2656 |    0.348 |     0.3013 |            2.71921   |               2 |          0.444 |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | chain_of_thought | fakerecogna |      0.745 |      0.7138 |    0.818 |     0.7623 |            4.39452   |               2 |          0     |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | zero_shot        | fakerecogna |      0.759 |      0.9274 |    0.562 |     0.6999 |            0.710726  |               2 |          0     |         |
| neuralmind/bert-base-portuguese-cased | fine_tuned       | fakebr      |      0.998 |      0.998  |    0.998 |     0.998  |            0.0289664 |             nan |          0     |         |
| maritaca-ai/sabia-7b                  | chain_of_thought | fakebr      |      0     |      0      |    0     |     0      |            2.62684   |               2 |          1     |         |
| maritaca-ai/sabia-7b                  | few_shot         | fakebr      |      0.505 |      0.5309 |    0.086 |     0.148  |            3.29574   |               2 |          0     |         |
| neuralmind/bert-base-portuguese-cased | fine_tuned       | fakerecogna |      0.984 |      0.9879 |    0.98  |     0.9839 |            0.0298001 |             nan |          0     |         |
| Qwen/Qwen2-7B-Instruct                | few_shot         | fakerecogna |      0.837 |      0.8821 |    0.778 |     0.8268 |            0.659771  |               2 |          0     |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | zero_shot        | fakebr      |      0.442 |      0.3986 |    0.228 |     0.2901 |           80.7105    |               0 |          0.002 |         |
| Qwen/Qwen2-1.5B-Instruct              | chain_of_thought | fakebr      |      0.504 |      0.504  |    0.506 |     0.505  |           14.12      |               0 |          0     |         |
| maritaca-ai/sabia-7b                  | chain_of_thought | fakerecogna |      0.001 |      0.002  |    0.002 |     0.002  |            1.74344   |               2 |          0.998 |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | few_shot         | fakebr      |      0.459 |      0.1525 |    0.018 |     0.0322 |            2.56739   |               2 |          0     |         |
| Qwen/Qwen2-1.5B-Instruct              | chain_of_thought | fakerecogna |      0.48  |      0.4858 |    0.684 |     0.5681 |           11.6026    |               0 |          0     |         |
| Qwen/Qwen2-7B-Instruct                | chain_of_thought | fakerecogna |      0.697 |      0.6374 |    0.914 |     0.751  |            3.21918   |               2 |          0.001 |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | chain_of_thought | fakebr      |      0.333 |      0.2749 |    0.204 |     0.2342 |            5.24777   |               2 |          0     |         |
| Qwen/Qwen2-1.5B-Instruct              | few_shot         | fakerecogna |      0.531 |      0.5162 |    0.986 |     0.6777 |           16.2408    |               0 |          0     |         |
| maritaca-ai/sabia-7b                  | few_shot         | fakerecogna |      0.474 |      0.4782 |    0.57  |     0.5201 |            7.27171   |               2 |          0.001 |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | chain_of_thought | fakebr      |      0.397 |      0.3589 |    0.262 |     0.3029 |            4.88984   |               2 |          0     |         |
| Qwen/Qwen2-7B-Instruct                | zero_shot        | fakebr      |      0.4   |      0.0833 |    0.02  |     0.0323 |            1.344     |               2 |          0     |         |
| Qwen/Qwen2-7B-Instruct                | chain_of_thought | fakebr      |      0.482 |      0.4829 |    0.508 |     0.4951 |            4.2225    |               2 |          0     |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | chain_of_thought | fakerecogna |      0.677 |      0.6364 |    0.826 |     0.7189 |            3.877     |               2 |          0     |         |
| maritaca-ai/sabia-7b                  | zero_shot        | fakerecogna |      0.38  |      0.4318 |    0.76  |     0.5507 |            2.20252   |               2 |          0.244 |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | few_shot         | fakerecogna |      0.867 |      0.8522 |    0.888 |     0.8697 |            0.711878  |               2 |          0     |         |
| Qwen/Qwen2-1.5B-Instruct              | few_shot         | fakebr      |      0.504 |      0.5036 |    0.556 |     0.5285 |           23.415     |               0 |          0     |         |
| Qwen/Qwen2-1.5B-Instruct              | zero_shot        | fakebr      |      0.507 |      0.5069 |    0.512 |     0.5095 |           11.9577    |               0 |          0     |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | zero_shot        | fakerecogna |      0.724 |      0.959  |    0.468 |     0.629  |           14.9617    |               2 |          0     |         |
| CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it     | few_shot         | fakerecogna |      0.62  |      0.9839 |    0.244 |     0.391  |            1.87335   |               2 |          0     |         |
| Qwen/Qwen2-1.5B-Instruct              | zero_shot        | fakerecogna |      0.52  |      0.5139 |    0.74  |     0.6066 |           10.1021    |               0 |          0     |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | few_shot         | fakebr      |      0.229 |      0.0698 |    0.044 |     0.054  |            1.52761   |               2 |          0     |         |
| meta-llama/Meta-Llama-3-8B-Instruct   | zero_shot        | fakebr      |      0.364 |      0.0467 |    0.014 |     0.0215 |            1.5261    |               2 |          0     |         |

## Análise por Modelo

### Qwen/Qwen2-7B-Instruct

- **F1-Score Médio:** 0.4841
- **Melhor Estratégia:** few_shot

### maritaca-ai/sabia-7b

- **F1-Score Médio:** 0.2537
- **Melhor Estratégia:** zero_shot

### meta-llama/Meta-Llama-3-8B-Instruct

- **F1-Score Médio:** 0.4403
- **Melhor Estratégia:** few_shot

### neuralmind/bert-base-portuguese-cased

- **F1-Score Médio:** 0.9910
- **Melhor Estratégia:** fine_tuned

### CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it

- **F1-Score Médio:** 0.3940
- **Melhor Estratégia:** chain_of_thought

### Qwen/Qwen2-1.5B-Instruct

- **F1-Score Médio:** 0.5659
- **Melhor Estratégia:** few_shot


## Análise por Estratégia de Prompting

### Few Shot

- **F1-Score Médio:** 0.4099
- **Melhor Modelo:** meta-llama/Meta-Llama-3-8B-Instruct

### Zero Shot

- **F1-Score Médio:** 0.4390
- **Melhor Modelo:** Qwen/Qwen2-7B-Instruct

### Chain Of Thought

- **F1-Score Médio:** 0.4340
- **Melhor Modelo:** meta-llama/Meta-Llama-3-8B-Instruct

### Fine Tuned

- **F1-Score Médio:** 0.9910
- **Melhor Modelo:** neuralmind/bert-base-portuguese-cased


## Conclusões

Os resultados demonstram que [análise a ser completada após execução dos experimentos].

## Metodologia

- **Modelos testados:** Qwen/Qwen2-7B-Instruct, maritaca-ai/sabia-7b, meta-llama/Meta-Llama-3-8B-Instruct, neuralmind/bert-base-portuguese-cased, CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it, Qwen/Qwen2-1.5B-Instruct
- **Estratégias de prompting:** few_shot, zero_shot, chain_of_thought, fine_tuned
- **Datasets:** fakebr, fakerecogna

---
*Relatório gerado automaticamente pelo benchmark de LLMs para Fake News.*
