"""
=============================================================================
Módulo de Construção de Prompts
=============================================================================
Classes e funções para construir prompts para diferentes estratégias.
"""

from typing import Optional
import pandas as pd


# =============================================================================
# TEMPLATES DE PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Você é um especialista em análise de notícias e detecção de desinformação. 
Sua tarefa é analisar notícias em português brasileiro e classificá-las como verdadeiras ou falsas.
Seja objetivo e baseie sua análise apenas no conteúdo fornecido."""

ZERO_SHOT_TEMPLATE = """Analise a seguinte notícia e classifique-a como "Verdadeira" ou "Falsa".

Notícia:
{text}

Responda apenas com uma palavra: "Verdadeira" ou "Falsa".

Classificação:"""

FEW_SHOT_TEMPLATE = """Analise as notícias a seguir e classifique cada uma como "Verdadeira" ou "Falsa".

{examples}

Agora, classifique a seguinte notícia:

Notícia:
{text}

Responda apenas com uma palavra: "Verdadeira" ou "Falsa".

Classificação:"""

FEW_SHOT_EXAMPLE_TEMPLATE = """Notícia:
{text}

Classificação: {label}
---"""

CHAIN_OF_THOUGHT_TEMPLATE = """Analise a seguinte notícia passo a passo para determinar se é verdadeira ou falsa.

Notícia:
{text}

Siga estes passos de análise:
1. Identifique a fonte e credibilidade aparente
2. Verifique se há linguagem sensacionalista ou emocional
3. Analise se as afirmações são verificáveis
4. Considere se há inconsistências lógicas
5. Avalie o tom geral da notícia

Após sua análise, forneça sua classificação final.

Análise:
Passo 1 (Fonte):
Passo 2 (Linguagem):
Passo 3 (Verificabilidade):
Passo 4 (Consistência):
Passo 5 (Tom):

Classificação Final (responda "Verdadeira" ou "Falsa"):"""

# Template alternativo mais simples para CoT
CHAIN_OF_THOUGHT_SIMPLE_TEMPLATE = """Analise a seguinte notícia e determine se é verdadeira ou falsa.

Notícia:
{text}

Primeiro, explique brevemente seu raciocínio (em 2-3 frases).
Depois, forneça sua classificação final.

Raciocínio:

Classificação Final (responda apenas "Verdadeira" ou "Falsa"):"""


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def truncate_text(text: str, max_length: int = 2000) -> str:
    """
    Trunca texto para evitar exceder limite de tokens.
    
    Args:
        text: Texto original
        max_length: Comprimento máximo em caracteres
    
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    # Truncar no último espaço antes do limite
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + "..."


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class PromptBuilder:
    """
    Classe para construir prompts de acordo com a estratégia selecionada.
    
    Attributes:
        strategy: Estratégia de prompting ('zero_shot', 'few_shot', 'chain_of_thought')
        examples: DataFrame com exemplos para few-shot
        max_text_length: Comprimento máximo do texto da notícia
    """
    
    def __init__(
        self,
        strategy: str = "zero_shot",
        max_text_length: int = 2000,
        use_system_prompt: bool = True
    ):
        """
        Inicializa o construtor de prompts.
        
        Args:
            strategy: Estratégia de prompting
            max_text_length: Comprimento máximo do texto
            use_system_prompt: Se deve incluir system prompt
        """
        self.strategy = strategy
        self.max_text_length = max_text_length
        self.use_system_prompt = use_system_prompt
        self.examples = None
        
        # Validar estratégia
        valid_strategies = ["zero_shot", "few_shot", "chain_of_thought"]
        if strategy not in valid_strategies:
            raise ValueError(f"Estratégia inválida. Opções: {valid_strategies}")
    
    def set_examples(self, examples_df: pd.DataFrame):
        """
        Define os exemplos para few-shot learning.
        
        Args:
            examples_df: DataFrame com colunas 'text' e 'label'
        """
        self.examples = examples_df
    
    def build_prompt(self, text: str) -> str:
        """
        Constrói o prompt completo para uma notícia.
        
        Args:
            text: Texto da notícia a ser classificada
        
        Returns:
            Prompt formatado
        """
        # Truncar texto se necessário
        text = truncate_text(text, self.max_text_length)
        
        # Construir prompt de acordo com a estratégia
        if self.strategy == "zero_shot":
            prompt = self._build_zero_shot(text)
        elif self.strategy == "few_shot":
            prompt = self._build_few_shot(text)
        elif self.strategy == "chain_of_thought":
            prompt = self._build_chain_of_thought(text)
        
        # Adicionar system prompt se necessário
        if self.use_system_prompt:
            prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        
        return prompt
    
    def _build_zero_shot(self, text: str) -> str:
        """Constrói prompt zero-shot."""
        return ZERO_SHOT_TEMPLATE.format(text=text)
    
    def _build_few_shot(self, text: str) -> str:
        """Constrói prompt few-shot."""
        if self.examples is None or len(self.examples) == 0:
            raise ValueError("Exemplos não definidos para few-shot. Use set_examples().")
        
        # Formatar exemplos
        examples_text = ""
        for _, row in self.examples.iterrows():
            label_str = "Verdadeira" if row["label"] == 0 else "Falsa"
            example_text = truncate_text(row["text"], self.max_text_length // 2)
            examples_text += FEW_SHOT_EXAMPLE_TEMPLATE.format(
                text=example_text,
                label=label_str
            ) + "\n"
        
        return FEW_SHOT_TEMPLATE.format(examples=examples_text, text=text)
    
    def _build_chain_of_thought(self, text: str) -> str:
        """Constrói prompt Chain-of-Thought."""
        return CHAIN_OF_THOUGHT_SIMPLE_TEMPLATE.format(text=text)
    
    def get_template(self) -> str:
        """Retorna o template atual."""
        templates = {
            "zero_shot": ZERO_SHOT_TEMPLATE,
            "few_shot": FEW_SHOT_TEMPLATE,
            "chain_of_thought": CHAIN_OF_THOUGHT_SIMPLE_TEMPLATE,
        }
        return templates[self.strategy]


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def format_chat_prompt(
    prompt: str,
    model_type: str = "default"
) -> list:
    """
    Formata o prompt para o formato de chat de diferentes modelos.
    
    Args:
        prompt: Prompt em texto simples
        model_type: Tipo de modelo ('llama', 'qwen', 'default')
    
    Returns:
        Lista de mensagens no formato chat
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    return messages


def create_classification_prompt(
    text: str,
    strategy: str = "zero_shot",
    examples: Optional[pd.DataFrame] = None,
    max_length: int = 2000
) -> str:
    """
    Função de conveniência para criar prompts rapidamente.
    
    Args:
        text: Texto da notícia
        strategy: Estratégia de prompting
        examples: Exemplos para few-shot (opcional)
        max_length: Comprimento máximo
    
    Returns:
        Prompt formatado
    """
    builder = PromptBuilder(strategy=strategy, max_text_length=max_length)
    
    if strategy == "few_shot" and examples is not None:
        builder.set_examples(examples)
    
    return builder.build_prompt(text)


# =============================================================================
# PROMPTS ALTERNATIVOS (para experimentação)
# =============================================================================

ALTERNATIVE_PROMPTS = {
    "concise": """Classifique como Verdadeira ou Falsa:
{text}
Classificação:""",
    
    "detailed": """Você é um fact-checker profissional. Analise cuidadosamente a seguinte notícia 
e determine sua veracidade baseado em:
- Uso de linguagem sensacionalista
- Presença de fontes verificáveis
- Consistência interna
- Tom emocional vs. factual

Notícia:
{text}

Sua classificação (Verdadeira/Falsa):""",
    
    "binary": """Notícia: {text}

Esta notícia é fake news? Responda apenas "Sim" ou "Não".
Resposta:""",
    
    "confidence": """Analise a notícia abaixo e classifique-a.

Notícia:
{text}

Forneça:
1. Classificação: Verdadeira ou Falsa
2. Confiança: Alta, Média ou Baixa
3. Justificativa breve (1 frase)

Resposta:"""
}
