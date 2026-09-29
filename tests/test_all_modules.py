"""
Suíte de Testes Automatizados - Módulos de Busca, Previsão e Agente DuckDB
"""
import unittest
import json
import os
from modules.semantic_search import SemanticSearchEngine
from modules.demand_forecasting import SalesDemandForecaster, calculate_wape
from modules.reporting_agent import AutonomousReportingAgent

class TestCoreModules(unittest.TestCase):
    def test_semantic_search_engine(self):
        engine = SemanticSearchEngine()
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_docs.json")
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        engine.index_documents(docs)
        results = engine.search("reembolso de refeição em viagem", top_k=3)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], "DOC-FIN-002")

    def test_demand_forecasting(self):
        forecaster = SalesDemandForecaster(random_state=42)
        df_sales = forecaster.generate_synthetic_sales_data(n_weeks=52)
        results = forecaster.train_and_evaluate(df_sales, test_weeks=8)
        self.assertIn("metrics", results)
        self.assertEqual(len(results["metrics"]), 4)
        best_wape = results["metrics"].iloc[0]["WAPE (%)"]
        self.assertGreater(best_wape, 0.0)

    def test_reporting_agent(self):
        agent = AutonomousReportingAgent()
        res = agent.process_request("Qual o faturamento por canal de aquisição?")
        self.assertIn("sql", res)
        self.assertIn("faturamento_total", res["data"].columns)
        self.assertTrue("### 📊 Diagnóstico Executivo" in res["narrative"])

if __name__ == "__main__":
    unittest.main()
