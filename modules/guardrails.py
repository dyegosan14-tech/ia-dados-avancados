"""
Módulo de Guardrails & Segurança para Sistemas de IA Generativa.
Funcionalidades:
1. Detecção e Anonimização de Dados Sensíveis (PII: CPF, CNPJ, Cartões de Crédito, E-mails, Telefones e API Keys).
2. Detecção de Tentativas de Injeção de Prompt (Prompt Injection, Jailbreak, Delimiter Hijacking).
3. Sanitização e Bloqueio Preventivo de Respostas Não Conformes.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class GuardrailResult:
    is_safe: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detected_pii: List[Dict[str, str]] = field(default_factory=list)
    detected_injections: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    reasons: List[str] = field(default_factory=list)

class AIGuardrailService:
    """
    Motor de segurança em tempo de execução para avaliar entradas e saídas de LLMs.
    """
    def __init__(self):
        # Padrões Regex de PII (com foco em conformidade LGPD e segurança bancária)
        self.pii_patterns = {
            "CPF": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
            "CNPJ": r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
            "Cartão de Crédito": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "E-mail": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
            "Telefone": r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4}[-\s]?\d{4}\b",
            "API Key / Secret Token": r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9-_]{20,}\.[A-Za-z0-9-_]{20,}\.[A-Za-z0-9-_]{20,})\b"
        }

        # Padrões de Injeção de Prompt, Jailbreak e Exfiltração de Instruções
        self.injection_signatures = [
            (r"ignore\s+(all\s+)?(previous|prior)\s+instructions", "Ignorar instruções anteriores"),
            (r"disregard\s+(everything|instructions|rules)", "Desconsideração de regras do sistema"),
            (r"you\s+are\s+now\s+(in\s+developer\s+mode|dan|unrestricted)", "Ativação de Modo DAN / Developer Mode"),
            (r"reveal\s+(your\s+)?(system\s+prompt|secret\s+instructions|instructions)", "Tentativa de exfiltração de System Prompt"),
            (r"repetir\s+(o\s+)?(prompt\s+de\s+sistema|as\s+instruções\s+iniciais)", "Exfiltração de instruções em português"),
            (r"esqueça\s+(tudo|todas\s+as\s+regras|o\s+contexto)", "Tentativa de bypass de contexto em português"),
            (r"<\s*system\s*>|\[\s*INST\s*\]|```system", "Delimiter Injection / Escape de tags especiais"),
            (r"bypass\s+(safety|content\s+filter|guardrails)", "Tentativa de evasão de guardrails")
        ]

    def scan_for_pii(self, text: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Escaneia e identifica dados sensíveis (PII), gerando versão anonimizada.
        """
        detected = []
        sanitized = text

        for pii_type, pattern in self.pii_patterns.items():
            matches = list(re.finditer(pattern, sanitized, re.IGNORECASE))
            for match in reversed(matches):
                val = match.group(0)
                # Validação adicional simples para evitar falsos positivos de telefone em números comuns
                if pii_type == "Telefone" and len(re.sub(r"\D", "", val)) < 10:
                    continue

                detected.append({
                    "type": pii_type,
                    "value": val,
                    "start": match.start(),
                    "end": match.end()
                })
                # Mascara o dado no texto anonimizado
                mask = f"[{pii_type.upper().replace(' ', '_')}_PROTEGIDO]"
                sanitized = sanitized[:match.start()] + mask + sanitized[match.end():]

        return detected, sanitized

    def scan_for_prompt_injection(self, text: str) -> List[str]:
        """
        Analisa se o texto contém assinaturas típicas de prompt injection e jailbreak.
        """
        detected_attacks = []
        text_lower = text.lower()

        for pattern, description in self.injection_signatures:
            if re.search(pattern, text_lower):
                detected_attacks.append(description)

        return detected_attacks

    def inspect(self, text: str, block_on_injection: bool = True) -> GuardrailResult:
        """
        Executa a inspeção completa de segurança (PII + Injection).
        """
        detected_pii, sanitized = self.scan_for_pii(text)
        detected_injections = self.scan_for_prompt_injection(text)

        reasons = []
        risk_level = "LOW"
        is_safe = True

        # Avaliação de Risco de Injection
        if detected_injections:
            risk_level = "CRITICAL"
            is_safe = False
            reasons.append(f"Ataque de Injeção Detectado: {', '.join(detected_injections)}")

        # Avaliação de Risco de PII
        if detected_pii:
            pii_types = list(set(item["type"] for item in detected_pii))
            reasons.append(f"Dados Sensíveis (PII) Detectados: {', '.join(pii_types)}")
            if risk_level != "CRITICAL":
                risk_level = "HIGH" if any(t in ["CPF", "CNPJ", "Cartão de Crédito", "API Key / Secret Token"] for t in pii_types) else "MEDIUM"

        return GuardrailResult(
            is_safe=is_safe if block_on_injection else True,
            risk_level=risk_level,
            detected_pii=detected_pii,
            detected_injections=detected_injections,
            sanitized_text=sanitized,
            reasons=reasons
        )
