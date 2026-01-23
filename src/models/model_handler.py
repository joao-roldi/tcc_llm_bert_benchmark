"""
=============================================================================
Módulo de Manipulação de Modelos
=============================================================================
Classe para carregar, configurar e executar inferência com LLMs locais.
"""

import gc
from typing import Optional
from loguru import logger

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from config import MODEL_CONFIGS, EXPERIMENT_CONFIG, HARDWARE_CONFIG


class ModelHandler:
    """
    Classe para manipular modelos de linguagem.
    
    Attributes:
        model_name: Nome do modelo no Hugging Face
        model: Modelo carregado
        tokenizer: Tokenizador do modelo
        pipe: Pipeline de geração de texto
    """
    
    def __init__(self, model_name: str):
        """
        Inicializa o handler e carrega o modelo.
        
        Args:
            model_name: Nome do modelo no Hugging Face
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo e tokenizador."""
        logger.info(f"Carregando modelo: {self.model_name}")
        
        # Configuração de quantização
        quantization_config = None
        if HARDWARE_CONFIG["use_quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # 1. Detectar melhor device (adicionando MPS para Mac)
        if HARDWARE_CONFIG["use_cuda"] and torch.cuda.is_available():
            device_map = "auto"
        elif torch.backends.mps.is_available():
            device_map = "auto" # Accelerate lida bem com MPS usando "auto"
        else:
            device_map = "cpu"

        # 2. Determinar dtype seguro
        if device_map == "cpu":
            # Em CPU, float32 é o mais estável. bfloat16 é uma alternativa se a CPU suportar.
            torch_dtype = torch.float32 
        elif torch.backends.mps.is_available():
            # No Mac (MPS), float16 pode causar instabilidade numérica (NaNs) em modelos como Gemma.
            # float32 é mais seguro, embora consuma mais memória.
            torch_dtype = torch.float32
        else:
            # Em GPU, bfloat16 é preferível para Gemma, ou float16
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        
        # Configurações específicas do modelo
        model_config = MODEL_CONFIGS.get(self.model_name, {})
        
        # Carregar tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Definir pad_token se não existir
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Tentar usar Flash Attention 2
        if HARDWARE_CONFIG["use_flash_attention"]:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass  # Flash Attention não disponível
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Criar pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device_map
        )
        
        logger.info(f"Modelo carregado com sucesso!")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto a partir de um prompt.
        
        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais de geração
        
        Returns:
            Texto gerado pelo modelo
        """
        # Mesclar parâmetros padrão com kwargs
        gen_params = EXPERIMENT_CONFIG["generation_params"].copy()
        gen_params.update(kwargs)
        
        # Gerar resposta
        outputs = self.pipe(
            prompt,
            max_new_tokens=gen_params.get("max_new_tokens", 256),
            temperature=gen_params.get("temperature", 0.1),
            top_p=gen_params.get("top_p", 0.9),
            do_sample=gen_params.get("do_sample", True),
            repetition_penalty=gen_params.get("repetition_penalty", 1.1),
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Extrair texto gerado
        generated_text = outputs[0]["generated_text"]
        
        return generated_text.strip()
    
    def generate_batch(self, prompts: list, **kwargs) -> list:
        """
        Gera texto para múltiplos prompts.
        
        Args:
            prompts: Lista de prompts
            **kwargs: Parâmetros de geração
        
        Returns:
            Lista de textos gerados
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def get_vram_usage(self) -> float:
        """
        Retorna o uso atual de VRAM em GB.
        
        Returns:
            Uso de VRAM em GB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo.
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            "model_name": self.model_name,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.model.dtype),
            "device": str(next(self.model.parameters()).device),
            "vram_usage_gb": self.get_vram_usage(),
        }
    
    def unload(self):
        """Descarrega o modelo da memória."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        # Forçar coleta de lixo
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Modelo descarregado da memória.")


class ModelHandlerOllama:
    """
    Handler alternativo usando Ollama para modelos locais.
    Útil para modelos no formato GGUF.
    """
    
    def __init__(self, model_name: str, ollama_model: Optional[str] = None):
        """
        Inicializa o handler Ollama.
        
        Args:
            model_name: Nome do modelo (para referência)
            ollama_model: Nome do modelo no Ollama (ex: 'llama3:8b')
        """
        self.model_name = model_name
        self.ollama_model = ollama_model or model_name
        
        # Verificar se Ollama está disponível
        self._check_ollama()
    
    def _check_ollama(self):
        """Verifica se Ollama está instalado e rodando."""
        try:
            import ollama
            self.client = ollama
            # Testar conexão
            self.client.list()
            logger.info(f"Ollama conectado. Usando modelo: {self.ollama_model}")
        except ImportError:
            raise ImportError("Ollama não instalado. Execute: pip install ollama")
        except Exception as e:
            raise ConnectionError(f"Erro ao conectar com Ollama: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto usando Ollama.
        
        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais
        
        Returns:
            Texto gerado
        """
        response = self.client.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_new_tokens", 256),
            }
        )
        
        return response["response"].strip()
    
    def get_vram_usage(self) -> float:
        """Retorna uso estimado de VRAM."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def unload(self):
        """Placeholder para compatibilidade."""
        pass
