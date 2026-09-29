"""
Pacote de Módulos de IA e Dados Avançados:
- semantic_search: Busca semântica, vetores e reranking
- demand_forecasting: Previsão de séries temporais e comparação de modelos
- reporting_agent: Agente autônomo analítico com DuckDB e gráficos
- chatbot_evaluator: Avaliação de chatbots com LLM-as-a-Judge, latência e custos
- guardrails: Camada de segurança (detecção de prompt injection e anonimização de PII)
"""
from .guardrails import AIGuardrailService, GuardrailResult
from .semantic_search import SemanticSearchEngine
from .demand_forecasting import SalesDemandForecaster
from .reporting_agent import AutonomousReportingAgent
from .chatbot_evaluator import ChatbotEvaluatorService, ChatbotTestCase, MODEL_PRICING_TABLE
