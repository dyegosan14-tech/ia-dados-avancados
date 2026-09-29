"""
Módulo 4: Avaliador de Respostas de Chatbots (Framework LLM-as-a-Judge)
Mede Qualidade (Rubrica formal), Custo Financeiro (Tokens In/Out) e Latência (TTFT/Total).
Gera Matriz de Trade-off (Pareto-optimal) e gráficos comparativos.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Tabela Oficial de Precificação de Mercado (USD por 1 Milhão de Tokens)
MODEL_PRICING_TABLE = {
    "gemini-1.5-flash": {
        "provider": "Google",
        "input_per_m": 0.075,
        "output_per_m": 0.30,
        "typical_latency_ms": 320,
        "tokens_per_sec": 85
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "input_per_m": 1.25,
        "output_per_m": 5.00,
        "typical_latency_ms": 1150,
        "tokens_per_sec": 45
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "typical_latency_ms": 410,
        "tokens_per_sec": 70
    },
    "gpt-4o": {
        "provider": "OpenAI",
        "input_per_m": 2.50,
        "output_per_m": 10.00,
        "typical_latency_ms": 1050,
        "tokens_per_sec": 42
    },
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "input_per_m": 3.00,
        "output_per_m": 15.00,
        "typical_latency_ms": 1280,
        "tokens_per_sec": 38
    },
    "llama-3.3-70b": {
        "provider": "Meta / Groq",
        "input_per_m": 0.59,
        "output_per_m": 0.79,
        "typical_latency_ms": 280,
        "tokens_per_sec": 120
    }
}

@dataclass
class ChatbotTestCase:
    test_id: str
    scenario_name: str
    question: str
    reference_context: str
    expected_keywords: List[str]
    prohibited_keywords: List[str]

@dataclass
class ChatbotEvaluationRun:
    test_id: str
    model_name: str
    prompt_name: str
    question: str
    context: str
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Métricas calculadas
    cost_usd: float = 0.0
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    overall_quality_score: float = 0.0
    is_faithful: bool = True
    judge_reasoning: str = ""

from .guardrails import AIGuardrailService

class ChatbotEvaluatorService:
    """
    Serviço central de avaliação e benchmarking de LLMs.
    Combina análise quantitativa de telemetria (custo/latência), qualitativa (LLM-as-a-Judge)
    e checagem de conformidade de segurança (Guardrails de PII e Prompt Injection).
    """
    def __init__(self, pricing_table: Dict[str, Dict[str, Any]] = MODEL_PRICING_TABLE):
        self.pricing_table = pricing_table
        self.guardrail = AIGuardrailService()

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcula o custo exato da inferência em dólares."""
        pricing = self.pricing_table.get(model, {"input_per_m": 0.5, "output_per_m": 1.5})
        cost_in = (input_tokens / 1_000_000) * pricing["input_per_m"]
        cost_out = (output_tokens / 1_000_000) * pricing["output_per_m"]
        return round(cost_in + cost_out, 6)

    def judge_evaluation(self, run: ChatbotEvaluationRun, expected_keys: List[str], prohibited_keys: List[str]) -> Dict[str, Any]:
        """
        Simulador do motor de LLM-as-a-Judge com cadeia de raciocínio (Chain of Thought)
        integrado com inspeção de Guardrails.
        """
        ans_lower = run.answer.lower()
        ctx_lower = run.context.lower()

        # 1. Inspeção de Guardrails (Segurança e PII)
        guard_res = self.guardrail.inspect(run.answer)

        # 2. Avaliação de Fidelidade (Faithfulness) / Detecção de Alucinação
        has_hallucination = False
        hallucination_reason = []
        for bad_key in prohibited_keys:
            if bad_key.lower() in ans_lower and bad_key.lower() not in ctx_lower:
                has_hallucination = True
                hallucination_reason.append(f"termo proibido/alucinado '{bad_key}'")

        if not guard_res.is_safe:
            faithfulness = 1.0
            is_faithful = False
            r_faith = f"BLOQUEADO POR GUARDRAILS: {'; '.join(guard_res.reasons)}"
        elif has_hallucination:
            faithfulness = 1.0
            is_faithful = False
            r_faith = f"REPROVADO POR ALUCINAÇÃO: O modelo introduziu {', '.join(hallucination_reason)} não respaldado pelo contexto oficial."
        else:
            faithfulness = 5.0
            is_faithful = True
            r_faith = "APROVADO: Nenhuma alucinação detectada; resposta ancorada no contexto."

        # 3. Avaliação de Relevância e Completude (Coverage dos requisitos)
        matched_keys = [k for k in expected_keys if k.lower() in ans_lower]
        coverage_pct = len(matched_keys) / max(1, len(expected_keys))

        if coverage_pct >= 0.9:
            completeness = 5.0
            relevance = 5.0
            r_comp = f"Atendeu integralmente aos {len(expected_keys)} pontos obrigatórios ({', '.join(matched_keys)})."
        elif coverage_pct >= 0.5:
            completeness = 3.5
            relevance = 4.0
            r_comp = f"Cobriu parcialmente os requisitos ({len(matched_keys)}/{len(expected_keys)} pontos). Faltou detalhamento."
        else:
            completeness = 2.0
            relevance = 2.5
            r_comp = "Resposta evasiva ou insuficiente para a dúvida apresentada."

        # Se houver alucinação grave ou quebra de guardrail, a nota geral cai drasticamente
        if not is_faithful or not guard_res.is_safe:
            overall = 1.0
        else:
            overall = round((faithfulness * 0.4) + (relevance * 0.3) + (completeness * 0.3), 1)

        full_reasoning = f"[Fidelidade: {faithfulness}/5] {r_faith} | [Completude: {completeness}/5] {r_comp}"

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "overall": overall,
            "is_faithful": is_faithful,
            "guardrail_status": "🚨 Risco Crítico" if not guard_res.is_safe else ("⚠️ Risco Médio (PII)" if guard_res.detected_pii else "✅ Conforme"),
            "reasoning": full_reasoning
        }

    def run_benchmark_suite(self, runs: List[Dict[str, Any]], test_case: ChatbotTestCase) -> pd.DataFrame:
        """Processa a suíte completa de execuções e gera a tabela consolidada."""
        processed_records = []

        for item in runs:
            model = item["model_name"]
            in_tok = item["input_tokens"]
            out_tok = item["output_tokens"]
            lat = item["latency_ms"]
            ans = item["answer"]
            prompt_name = item.get("prompt_name", "Padrão")

            cost = self.compute_cost(model, in_tok, out_tok)
            
            run_obj = ChatbotEvaluationRun(
                test_id=test_case.test_id,
                model_name=model,
                prompt_name=prompt_name,
                question=test_case.question,
                context=test_case.reference_context,
                answer=ans,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=lat,
                cost_usd=cost
            )

            judge_result = self.judge_evaluation(run_obj, test_case.expected_keywords, test_case.prohibited_keywords)
            
            processed_records.append({
                "Modelo": model,
                "Prompt": prompt_name,
                "Qualidade (1-5)": judge_result["overall"],
                "Fidelidade (1-5)": judge_result["faithfulness"],
                "Relevância (1-5)": judge_result["relevance"],
                "Completude (1-5)": judge_result["completeness"],
                "Sem Alucinação?": "Sim" if judge_result["is_faithful"] else "Não",
                "Segurança / Guardrails": judge_result["guardrail_status"],
                "Latência (ms)": lat,
                "Tokens (In/Out)": f"{in_tok} / {out_tok}",
                "Custo Unitário (USD)": cost,
                "Custo / 100k Req (USD)": round(cost * 100_000, 2),
                "Diagnóstico do Juiz": judge_result["reasoning"]
            })

        return pd.DataFrame(processed_records)

    def create_tradeoff_scatter_plot(self, df: pd.DataFrame):
        """
        Gera a Matriz de Trade-off (Qualidade vs Custo com tamanho = Latência).
        Permite identificar a fronteira de eficiência de Pareto.
        """
        fig = px.scatter(
            df,
            x="Custo / 100k Req (USD)",
            y="Qualidade (1-5)",
            size="Latência (ms)",
            color="Modelo",
            hover_name="Modelo",
            hover_data=["Prompt", "Sem Alucinação?", "Latência (ms)"],
            text="Modelo",
            title="Matriz de Trade-off: Qualidade vs Custo (Tamanho da Bolha = Latência)",
            range_y=[0, 5.5]
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        
        # Linha de corte de qualidade aceitável
        fig.add_hline(y=4.0, line_dash="dash", line_color="#10b981", annotation_text="Corte Mínimo de Qualidade (4.0)")
        return fig

    def create_radar_quality_plot(self, df: pd.DataFrame):
        """Gráfico de Radar comparando dimensões de qualidade de cada modelo."""
        categories = ["Fidelidade", "Relevância", "Completude", "Score Geral"]
        fig = go.Figure()

        for idx, row in df.iterrows():
            values = [
                row["Fidelidade (1-5)"],
                row["Relevância (1-5)"],
                row["Completude (1-5)"],
                row["Qualidade (1-5)"]
            ]
            # Fechamento do polígono do radar
            values.append(values[0])
            cat_closed = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cat_closed,
                fill='toself',
                name=f"{row['Modelo']} ({row['Prompt']})"
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            title="Radar de Competências Qualitativas (Avaliador)",
            font=dict(family="Inter, sans-serif")
        )
        return fig

    def get_standard_test_cases(self) -> List[ChatbotTestCase]:
        """Conjunto pré-configurado de cenários corporativos para testes imediatos."""
        return [
            ChatbotTestCase(
                test_id="TC-FIN-01",
                scenario_name="Financeiro: Reembolso de Despesas",
                question="Qual é o limite diário de refeição e qual o prazo para pedir reembolso?",
                reference_context="O limite diário para despesas de alimentação em viagens corporativas é de R$ 85,00 por colaborador em capitais e R$ 65,00 em cidades do interior. O prazo máximo para envio das notas fiscais é de 30 dias corridos após o término da viagem.",
                expected_keywords=["85", "30 dias", "nota fiscal"],
                prohibited_keywords=["ilimitado", "sem prazo", "180 dias"]
            ),
            ChatbotTestCase(
                test_id="TC-RH-02",
                scenario_name="RH: Fracionamento de Férias",
                question="Posso dividir minhas férias em quantos períodos e qual o período mínimo?",
                reference_context="As férias podem ser divididas em até 3 períodos, desde que um não seja menor que 14 dias e os demais não sejam menores que 5 dias corridos cada. A solicitação deve ser feita 45 dias antes.",
                expected_keywords=["3 períodos", "14 dias", "5 dias"],
                prohibited_keywords=["5 períodos", "2 dias", "quando quiser"]
            ),
            ChatbotTestCase(
                test_id="TC-TI-03",
                scenario_name="Segurança: Acesso Remoto",
                question="Como faço para acessar a rede corporativa de casa?",
                reference_context="O acesso remoto deve ser efetuado exclusivamente através da VPN homologada com autenticação em dois fatores (MFA). É expressamente vedado o uso de computadores pessoais não cadastrados.",
                expected_keywords=["vpn", "mfa"],
                prohibited_keywords=["qualquer computador", "sem senha", "direto pelo navegador"]
            )
        ]
