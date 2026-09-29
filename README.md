# 🚀 IA & Dados Avançados: Plataforma de Soluções Inteligentes

[![CI - Testes Automatizados](https://github.com/dyegosan14-tech/ia-dados-avancados/actions/workflows/ci.yml/badge.svg)](https://github.com/dyegosan14-tech/ia-dados-avancados/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/Engine-DuckDB-FFF000?logo=duckdb&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Uma plataforma integrada, modular e pronta para produção desenvolvida em Python que reúne os pilares mais avançados de IA Generativa e Ciência de Dados moderna:

1. ⚖️ **Avaliador de Respostas de Chatbots**: Framework de benchmarking (*LLM-as-a-Judge*), métricas de qualidade (fidelidade/relevância), monitoramento de latência e cálculo dinâmico de custos.
2. 🛡️ **Camada de Guardrails & Segurança**: Inspeção em tempo de execução para detecção/bloqueio de *Prompt Injection* e mascaramento automático de dados sensíveis (*PII: CPF, CNPJ, Cartões, API Keys*).
3. 🔍 **Busca Semântica em Documentos**: Pipeline híbrido (vetorial denso + léxico esparso) com fusão *RRF* (*Reciprocal Rank Fusion*) e *Cross-Encoder Reranking*.
4. 📈 **Previsão de Demanda de Vendas**: Modelagem de séries temporais comparando baselines estatísticos (Naive, Média Móvel) com Machine Learning (Gradient Boosting e Random Forest) via métrica *WAPE*.
5. 🤖 **Agente Autônomo de Relatórios**: Agente analítico *Text-to-Data* que traduz linguagem natural para SQL seguro em DuckDB, plota gráficos interativos e redige diagnósticos executivos.

---

## 🛠️ Tecnologias & Arquitetura

- **Linguagem**: Python 3.9+
- **Interface e Dashboard**: [Streamlit](https://streamlit.io/) com tema escuro premium e CSS customizado
- **Visualização Interativa**: [Plotly](https://plotly.com/) (gráficos de dispersão de Pareto, radar de competências, séries temporais e barras)
- **Banco Analítico In-Memory**: [DuckDB](https://duckdb.org/)
- **Machine Learning & NLP**: [Scikit-Learn](https://scikit-learn.org/) (SVD, TF-IDF, Random Forest, Gradient Boosting)
- **Segurança & Guardrails**: Sanitização de PII e motor de assinaturas anti-jailbreak
- **CI/CD**: GitHub Actions com matriz de teste automatizada em Python 3.9, 3.10 e 3.11

---

## 📂 Estrutura do Repositório

```text
ia-dados-avancados/
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline de CI/CD automatizado no GitHub Actions
├── app.py                      # Dashboard Web Unificado interativo (Streamlit)
├── requirements.txt            # Dependências declaradas
├── README.md                   # Documentação detalhada do projeto
├── LICENSE                     # Licença MIT
├── .gitignore                  # Arquivos e pastas ignorados pelo Git
├── data/
│   └── mock_docs.json          # Base de documentos corporativos para busca semântica
├── modules/
│   ├── __init__.py
│   ├── chatbot_evaluator.py    # Módulo: Avaliação de LLMs, LLM-as-a-Judge, Custo e Latência
│   ├── guardrails.py           # Módulo: Detecção de PII e bloqueio de Prompt Injection
│   ├── semantic_search.py      # Módulo: Embeddings, busca vetorial, BM25 e Reranker
│   ├── demand_forecasting.py   # Módulo: Séries temporais, Lags, WAPE e modelos de ML
│   └── reporting_agent.py      # Módulo: Agente analítico com DuckDB e síntese de negócios
└── tests/
    ├── __init__.py
    ├── test_guardrails.py      # Testes de segurança e anonimização
    ├── test_chatbot_evaluator.py # Testes de avaliação, precificação e juiz
    └── test_all_modules.py     # Testes integrados de busca, previsão e DuckDB
```

---

## 🚀 Como Executar Localmente

### 1. Clonar o Repositório
```bash
git clone https://github.com/dyegosan14-tech/ia-dados-avancados.git
cd ia-dados-avancados
```

### 2. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv venv
source venv/bin/activate   # No Linux/macOS
# ou: venv\Scripts\activate no Windows
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Iniciar a Aplicação Web
```bash
streamlit run app.py
```
O painel estará disponível automaticamente em seu navegador em `http://localhost:8501`.

---

## 🧪 Suíte de Testes Automatizados

Para executar todos os testes unitários e de integração localmente:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Os testes são executados automaticamente a cada commit ou Pull Request através do GitHub Actions.

---

## 🧠 Detalhamento dos Módulos

### 1. ⚖️ Avaliador de Respostas de Chatbots (*LLM-as-a-Judge*)
- **Tríade de Decisão**: Avalia simultaneamente **Qualidade**, **Custo** e **Latência**.
- **Rubricas com *Reasoning-First***: O avaliador produz uma análise detalhada antes de conceder a nota (1 a 5), eliminando notas arbitrárias.
- **Detecção de Alucinações**: Penalização direta com reprovação (Score 1) para respostas que extrapolem o contexto oficial de suporte (*Faithfulness*).
- **Matriz de Trade-off de Pareto**: Dispersão interativa entre Custo por 100k requisições versus Score Qualitativo, ponderado pela latência em milissegundos.

### 2. 🛡️ Camada de Guardrails & Segurança
- **Anonimização de PII**: Identificação e mascaramento automático de CPF, CNPJ, Cartões de Crédito, E-mails e Chaves de API no padrão `[TIPO_PROTEGIDO]`.
- **Proteção contra Jailbreak / Injeção**: Bloqueio preventivo contra ataques de delimiter injection, bypass de instruções de sistema e tentativas de revelação de prompt.

### 3. 🔍 Busca Semântica em Documentos & Reranking
- **Dense Retrieval**: Embeddings semânticos para captura conceitual.
- **Sparse Retrieval**: BM25 / TF-IDF para casamento exato de termos e códigos.
- **RRF (Reciprocal Rank Fusion)**: Combina rankings de forma robusta e independente da escala.
- **Cross-Encoder Reranker**: Modelo de refinamento nos Top candidatos que analisa a interação cruzada entre cada par query-documento.

### 4. 📈 Previsão de Demanda de Vendas
- **Divisão Temporal Estrita**: Sem vazamento de dados (*data leakage*).
- **Engenharia de Atributos**: Geração de *lags* temporais (1, 2, 4 e 8 semanas), médias e desvios móveis, e variáveis sazonais de calendário.
- **Métrica WAPE**: $\frac{\sum |y - \hat{y}|}{\sum y}$ — padrão da indústria de varejo e supply chain que não quebra em vendas intermitentes (diferente do MAPE).

### 5. 🤖 Agente Autônomo de Relatórios
- **Text-to-SQL Seguro**: Mapeamento de intenção em linguagem natural para queries SQL otimizadas com trava para operações estritamente de leitura (`SELECT`).
- **Engine DuckDB**: Processamento colunar ultrarrápido em memória.
- **Síntese Executiva**: Redige automaticamente diagnósticos com destaques de receita, margem de lucro e recomendações estratégicas para tomada de decisão.

---

## 👨‍💻 Autor

Desenvolvido e mantido por **Dyego Assis** ([@dyegosan14-tech](https://github.com/dyegosan14-tech)).

[![GitHub Profile](https://img.shields.io/badge/GitHub-dyegosan14--tech-181717?style=for-the-badge&logo=github)](https://github.com/dyegosan14-tech)

---

## 📄 Licença
Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.
