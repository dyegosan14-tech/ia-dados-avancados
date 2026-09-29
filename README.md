# 🚀 IA & Dados Avançados: Plataforma de Soluções Inteligentes

Uma plataforma integrada e moderna desenvolvida em Python que reúne quatro dos pilares mais demandados em engenharia de IA e ciência de dados moderna:

1. ⚖️ **Avaliador de Respostas de Chatbots**: Framework de benchmarking (*LLM-as-a-Judge*), métricas de qualidade (fidelidade/relevância), monitoramento de latência e cálculo dinâmico de custos.
2. 🔍 **Busca Semântica em Documentos**: Pipeline híbrido (vetorial denso + léxico esparso) com fusão RRF (*Reciprocal Rank Fusion*) e *Cross-Encoder Reranking*.
3. 📈 **Previsão de Demanda de Vendas**: Modelagem de séries temporais comparando baselines estatísticos (Naive, Média Móvel) com Machine Learning (Gradient Boosting e Random Forest) via métrica WAPE.
4. 🤖 **Agente Autônomo de Relatórios**: Agente analítico Text-to-Data que traduz linguagem natural para SQL seguro em DuckDB, plota gráficos interativos e redige sínteses executivas.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.9+
- **Interface e Dashboard**: [Streamlit](https://streamlit.io/) com tema dark e componentes customizados
- **Visualização Interativa**: [Plotly](https://plotly.com/) (gráficos de dispersão de Pareto, radar qualitativo, séries temporais e barras)
- **Banco Analítico In-Memory**: [DuckDB](https://duckdb.org/)
- **Machine Learning & NLP**: [Scikit-Learn](https://scikit-learn.org/) (SVD, TF-IDF, Random Forest, Gradient Boosting)
- **Engenharia de Features**: Pandas & NumPy

---

## 📂 Estrutura do Repositório

```text
ia-dados-avancados/
├── app.py                      # Dashboard Web Unificado interativo (Streamlit)
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação completa
├── .gitignore                  # Arquivos ignorados no versionamento Git
├── data/
│   └── mock_docs.json          # Base de documentos corporativos para busca semântica
└── modules/
    ├── __init__.py
    ├── chatbot_evaluator.py    # Módulo 4: Avaliação de LLMs, LLM-as-a-Judge, Custo e Latência
    ├── semantic_search.py      # Módulo 1: Embeddings, busca vetorial, BM25 e Reranker
    ├── demand_forecasting.py   # Módulo 2: Séries temporais, Lags, WAPE e modelos de ML
    └── reporting_agent.py      # Módulo 3: Agente analítico com DuckDB e síntese de negócios
```

---

## 🚀 Como Executar Localmente

### 1. Clonar ou Acessar o Diretório
```bash
cd ia-dados-avancados
```

### 2. Criar e Ativar Ambiente Virtual (Recomendado)
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

## 🧠 Detalhamento dos Módulos

### 1. ⚖️ Avaliador de Respostas de Chatbots (*LLM-as-a-Judge*)
- **Tríade de Decisão**: Avalia simultaneamente **Qualidade**, **Custo** e **Latência**.
- **Rubricas com *Reasoning-First***: O avaliador obrigatoriamente produz uma análise detalhada antes de conceder a nota (1 a 5), evitando notas aleatórias.
- **Detecção de Alucinações**: Penalização direta com reprovação (Score 1) para respostas que extrapolem o contexto oficial de suporte (*Faithfulness*).
- **Matriz de Trade-off de Pareto**: Dispersão interativa entre Custo por 100k requisições versus Score Qualitativo, ponderado pela latência em milissegundos.

### 2. 🔍 Busca Semântica em Documentos & Reranking
- **Dense Retrieval**: Embeddings semânticos para captura conceitual.
- **Sparse Retrieval**: BM25 / TF-IDF para casamento exato de termos e códigos.
- **RRF (Reciprocal Rank Fusion)**: Combina rankings de forma robusta e independente da escala.
- **Cross-Encoder Reranker**: Modelo de refinamento nos Top candidatos que analisa a interação cruzada entre cada par query-documento.

### 3. 📈 Previsão de Demanda de Vendas
- **Divisão Temporal Estrita**: Sem vazamento de dados (*data leakage*).
- **Engenharia de Atributos**: Geração de *lags* temporais (1, 2, 4 e 8 semanas), médias e desvios móveis, e variáveis sazonais de calendário.
- **Métrica WAPE**: $\frac{\sum |y - \hat{y}|}{\sum y}$ — padrão da indústria de varejo e supply chain que não quebra em vendas intermitentes (diferente do MAPE).

### 4. 🤖 Agente Autônomo de Relatórios
- **Text-to-SQL Seguro**: Mapeamento de intenção em linguagem natural para queries SQL otimizadas com trava para operações estritamente de leitura (`SELECT`).
- **Engine DuckDB**: Processamento colunar ultrarrápido em memória.
- **Síntese Executiva**: Redige automaticamente diagnósticos com destaques de receita, margem de lucro e recomendações estratégicas para tomada de decisão.

---

## 📦 Como Subir no GitHub

Para publicar este projeto no seu GitHub pessoal ou corporativo:

```bash
# 1. Verifique se o git está inicializado e com os arquivos comitados
git status

# 2. Crie um novo repositório vazio no GitHub (ex: ia-dados-avancados)

# 3. Vincule a branch remota do seu repositório:
git remote add origin https://github.com/<SEU-USUARIO>/ia-dados-avancados.git

# 4. Envie o código:
git branch -M main
git push -u origin main
```

---

## 📄 Licença
Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.
