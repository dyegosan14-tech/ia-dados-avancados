"""
Módulo 3: Agente Autônomo que Gera Relatórios Sozinho
Recebe pergunta em linguagem natural -> Traduz para SQL analítico (DuckDB) -> 
Executa com segurança -> Monta gráficos interativos -> Redige resumo executivo.
"""
from typing import Dict, Any, List
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class AutonomousReportingAgent:
    """
    Agente de Inteligência Analítica que traduz perguntas em consultas SQL,
    processa dados no DuckDB, renderiza visualizações e redige relatórios de negócio.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(db_path)
        self._init_database()

    def _init_database(self):
        """Popula o banco DuckDB com dados analíticos ricos."""
        np.random.seed(42)
        n_rows = 500
        
        canais = ["Google Ads", "Meta Ads", "Orgânico (SEO)", "E-mail Marketing", "Parcerias/Afiliados"]
        regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
        categorias = ["Software B2B", "Hardware", "Consultoria Especializada", "Treinamentos"]
        produtos = {
            "Software B2B": ["Licença Cloud Pro", "Add-on Segurança", "Plano Enterprise"],
            "Hardware": ["Roteador Mesh Corp", "Servidor Edge", "Token Físico MFA"],
            "Consultoria Especializada": ["Diagnóstico de Dados", "Migração Cloud", "Auditoria LGPD"],
            "Treinamentos": ["Bootcamp Engenharia de IA", "Workshop Product Ops"]
        }

        dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
        
        records = []
        for i in range(n_rows):
            cat = np.random.choice(categorias)
            prod = np.random.choice(produtos[cat])
            canal = np.random.choice(canais)
            regiao = np.random.choice(regioes)
            random_date = dates[int(np.random.randint(0, len(dates)))]
            data_str = random_date.strftime("%Y-%m-%d")
            
            qtd = int(np.random.randint(1, 15))
            preco_unitario = float(np.random.choice([150, 450, 1200, 3500, 7800]))
            faturamento = qtd * preco_unitario
            custo_aquisicao = float(round(faturamento * np.random.uniform(0.15, 0.35), 2))
            custo_operacional = float(round(faturamento * np.random.uniform(0.20, 0.40), 2))
            lucro_liquido = faturamento - (custo_aquisicao + custo_operacional)

            records.append((
                i + 1000, data_str, canal, regiao, cat, prod, qtd, 
                faturamento, custo_aquisicao, custo_operacional, lucro_liquido
            ))

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS vendas_corporativas (
                id_pedido INTEGER,
                data_pedido DATE,
                canal VARCHAR,
                regiao VARCHAR,
                categoria VARCHAR,
                produto VARCHAR,
                quantidade INTEGER,
                faturamento DOUBLE,
                cac DOUBLE,
                custo_op DOUBLE,
                lucro_liquido DOUBLE
            );
        """)
        
        self.con.executemany("""
            INSERT INTO vendas_corporativas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, records)

    def get_database_schema(self) -> str:
        """Retorna o esquema da tabela para instruir o motor de Text-to-SQL."""
        return """
        Tabela: vendas_corporativas
        Colunas:
        - id_pedido (INTEGER): Identificador do pedido
        - data_pedido (DATE): Data da transação
        - canal (VARCHAR): Canal de aquisição (Google Ads, Meta Ads, Orgânico (SEO), E-mail Marketing, Parcerias/Afiliados)
        - regiao (VARCHAR): Região geográfica (Sudeste, Sul, Nordeste, Centro-Oeste, Norte)
        - categoria (VARCHAR): Categoria do produto/serviço
        - produto (VARCHAR): Nome específico do produto
        - quantidade (INTEGER): Volume de itens
        - faturamento (DOUBLE): Valor bruto transacionado em R$
        - cac (DOUBLE): Custo de aquisição do cliente em R$
        - custo_op (DOUBLE): Custos operacionais em R$
        - lucro_liquido (DOUBLE): Lucro final líquido apurado em R$
        """

    def generate_sql_for_question(self, question: str) -> Dict[str, Any]:
        """
        Interpreta a intenção analítica da pergunta e mapeia para a query SQL ideal.
        """
        q = question.lower()
        
        if "canal" in q or "marketing" in q or "aquisição" in q or "aquisicao" in q:
            sql = """
                SELECT 
                    canal,
                    COUNT(id_pedido) AS total_pedidos,
                    ROUND(SUM(faturamento), 2) AS faturamento_total,
                    ROUND(SUM(cac), 2) AS cac_total,
                    ROUND(SUM(lucro_liquido), 2) AS lucro_total,
                    ROUND((SUM(lucro_liquido) / SUM(faturamento)) * 100, 1) AS margem_liquida_pct
                FROM vendas_corporativas
                GROUP BY canal
                ORDER BY faturamento_total DESC;
            """
            chart_type = "bar"
            x_col = "canal"
            y_col = "faturamento_total"
            chart_title = "Faturamento e Margem Líquida por Canal de Aquisição"

        elif "região" in q or "regiao" in q or "geografia" in q or "estado" in q:
            sql = """
                SELECT 
                    regiao,
                    ROUND(SUM(faturamento), 2) AS faturamento_total,
                    ROUND(SUM(lucro_liquido), 2) AS lucro_total,
                    ROUND(AVG(faturamento / quantidade), 2) AS ticket_medio
                FROM vendas_corporativas
                GROUP BY regiao
                ORDER BY faturamento_total DESC;
            """
            chart_type = "bar"
            x_col = "regiao"
            y_col = "faturamento_total"
            chart_title = "Desempenho Comercial por Região"

        elif "categoria" in q or "produto" in q or "serviço" in q or "servico" in q:
            sql = """
                SELECT 
                    categoria,
                    SUM(quantidade) AS itens_vendidos,
                    ROUND(SUM(faturamento), 2) AS faturamento_total,
                    ROUND(SUM(lucro_liquido), 2) AS lucro_total
                FROM vendas_corporativas
                GROUP BY categoria
                ORDER BY faturamento_total DESC;
            """
            chart_type = "pie"
            x_col = "categoria"
            y_col = "faturamento_total"
            chart_title = "Distribuição de Faturamento por Categoria"

        else: # Padrão: Visão temporal mensal consolidada
            sql = """
                SELECT 
                    STRFTIME(data_pedido, '%Y-%m') AS mes,
                    ROUND(SUM(faturamento), 2) AS faturamento_total,
                    ROUND(SUM(lucro_liquido), 2) AS lucro_total,
                    COUNT(id_pedido) AS pedidos
                FROM vendas_corporativas
                GROUP BY mes
                ORDER BY mes ASC;
            """
            chart_type = "line"
            x_col = "mes"
            y_col = "faturamento_total"
            chart_title = "Evolução Histórica Mensal de Faturamento e Lucro"

        return {
            "sql": sql.strip(),
            "chart_type": chart_type,
            "x_col": x_col,
            "y_col": y_col,
            "chart_title": chart_title
        }

    def execute_safe_query(self, sql_query: str) -> pd.DataFrame:
        """Executa a consulta garantindo permissão estrita de leitura (SELECT)."""
        clean_sql = sql_query.strip().lower()
        if not clean_sql.startswith("select"):
            raise PermissionError("Apenas operações analíticas SELECT são permitidas.")
        return self.con.execute(sql_query).df()

    def build_chart(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, title: str):
        """Constrói uma figura Plotly interativa estilizada."""
        if chart_type == "bar":
            fig = px.bar(
                df, x=x_col, y=y_col, 
                text=y_col,
                title=title,
                color=y_col,
                color_continuous_scale="Blues"
            )
            fig.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
        elif chart_type == "pie":
            fig = px.pie(
                df, names=x_col, values=y_col,
                title=title,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
        elif chart_type == "line":
            fig = px.line(
                df, x=x_col, y=y_col, 
                markers=True,
                title=title,
                line_shape="spline"
            )
            fig.update_traces(line_color="#2563eb", line_width=3)
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Inter, sans-serif")
        )
        return fig

    def synthesize_narrative(self, question: str, df: pd.DataFrame, meta: Dict[str, Any]) -> str:
        """Gera a narrativa analítica com insights de negócio e plano de ação."""
        total_fat = df["faturamento_total"].sum() if "faturamento_total" in df.columns else 0
        total_lucro = df["lucro_total"].sum() if "lucro_total" in df.columns else 0
        margem_global = (total_lucro / total_fat * 100) if total_fat > 0 else 0

        # Identifica líder e oportunidade
        top_row = df.iloc[0]
        dimension_col = meta["x_col"]
        top_dim_name = top_row[dimension_col]
        top_fat = top_row.get("faturamento_total", 0)

        narrative = f"""
### 📊 Diagnóstico Executivo Automatizado

**Contexto da Solicitação**: *"{question}"*

1. **Visão Global dos Resultados**:
   - Volume total apurado: **R$ {total_fat:,.2f}**
   - Lucro líquido consolidado: **R$ {total_lucro:,.2f}** (Margem líquida agregada de **{margem_global:.1f}%**)

2. **Destaque de Liderança**:
   - O principal expoente identificado foi **{top_dim_name}**, gerando **R$ {top_fat:,.2f}** (responsável por **{(top_fat/total_fat*100):.1f}%** do faturamento do recorte analisado).

3. **Recomendações Práticas Acionáveis**:
   - **Alocação de Recursos**: Reforçar investimento de escala em **{top_dim_name}** para capitalizar a tração comprovada.
   - **Eficiência de Custos**: Avaliar os segmentos de menor margem para renegociação de custos de aquisição e revisão de precificação.
   - **Governança Contínua**: Manter monitoramento quinzenal automatizado deste indicador via rotina do agente.
"""
        return narrative.strip()

    def process_request(self, question: str) -> Dict[str, Any]:
        """Orquestração completa: Pergunta -> SQL -> Execução -> Gráfico -> Síntese."""
        plan = self.generate_sql_for_question(question)
        df_result = self.execute_safe_query(plan["sql"])
        fig = self.build_chart(df_result, plan["chart_type"], plan["x_col"], plan["y_col"], plan["chart_title"])
        narrative = self.synthesize_narrative(question, df_result, plan)

        return {
            "sql": plan["sql"],
            "data": df_result,
            "figure": fig,
            "narrative": narrative
        }
