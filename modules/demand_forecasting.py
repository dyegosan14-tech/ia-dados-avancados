"""
Módulo 2: Previsão de Demanda de Vendas (Séries Temporais)
Compara modelos Baselines (Naive, Média Móvel) com Machine Learning (Gradient Boosting / Random Forest).
Métricas calculadas: WAPE, RMSE, MAE e R².
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Weighted Absolute Percentage Error (evita divisão por zero)."""
    sum_true = np.sum(np.abs(y_true))
    if sum_true == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_true)

class SalesDemandForecaster:
    """
    Pipeline de previsão de demanda com engenharia de atributos temporais
    e comparação rigorosa entre baselines e modelos avançados.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def generate_synthetic_sales_data(self, n_weeks: int = 104) -> pd.DataFrame:
        """Gera uma série temporal realista com tendência, sazonalidade e ruído."""
        np.random.seed(self.random_state)
        dates = pd.date_range(start="2024-01-01", periods=n_weeks, freq="W")
        
        # Componente de tendência
        trend = np.linspace(80, 180, n_weeks)
        
        # Componente de sazonalidade (ciclos trimestrais e anuais)
        seasonality = 35 * np.sin(np.linspace(0, 8 * np.pi, n_weeks)) + 15 * np.cos(np.linspace(0, 2 * np.pi, n_weeks))
        
        # Picos de promoções (Black Friday, liquidações)
        promo_bumps = np.zeros(n_weeks)
        promo_weeks = [47, 99]  # Semanas de Black Friday
        for pw in promo_weeks:
            if pw < n_weeks:
                promo_bumps[pw] = 80.0
                if pw + 1 < n_weeks:
                    promo_bumps[pw + 1] = 40.0
        
        # Ruído aleatório
        noise = np.random.normal(0, 8, n_weeks)
        
        sales = np.maximum(10, trend + seasonality + promo_bumps + noise)
        
        return pd.DataFrame({
            "ds": dates,
            "y": np.round(sales, 1)
        })

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis defasadas (lags), médias móveis e variáveis de calendário."""
        data = df.copy()
        
        # Lags temporais
        for lag in [1, 2, 4, 8]:
            data[f"lag_{lag}"] = data["y"].shift(lag)
            
        # Médias móveis e volatilidade
        data["rolling_mean_4"] = data["y"].shift(1).rolling(window=4).mean()
        data["rolling_mean_8"] = data["y"].shift(1).rolling(window=8).mean()
        data["rolling_std_4"] = data["y"].shift(1).rolling(window=4).std()
        
        # Variáveis de calendário
        data["month"] = data["ds"].dt.month
        data["quarter"] = data["ds"].dt.quarter
        data["week_of_year"] = data["ds"].dt.isocalendar().week.astype(int)
        
        return data.dropna().reset_index(drop=True)

    def train_and_evaluate(self, raw_df: pd.DataFrame, test_weeks: int = 16) -> Dict[str, Any]:
        """
        Treina modelos e compara o desempenho no conjunto de teste.
        """
        featured_df = self.prepare_features(raw_df)
        
        # Divisão temporal estrita (sem data leakage)
        train_df = featured_df.iloc[:-test_weeks]
        test_df = featured_df.iloc[-test_weeks:]
        
        feature_cols = [c for c in featured_df.columns if c not in ["ds", "y"]]
        
        X_train, y_train = train_df[feature_cols], train_df["y"].values
        X_test, y_test = test_df[feature_cols], test_df["y"].values
        
        # 1. Baseline 1: Naive (Repete a última semana observada)
        pred_naive = test_df["lag_1"].values
        
        # 2. Baseline 2: Média Móvel (Rolling 4)
        pred_moving_avg = test_df["rolling_mean_4"].values
        
        # 3. Modelo Avançado: Gradient Boosting com Lags
        gbr = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.08, 
            max_depth=4, 
            random_state=self.random_state
        )
        gbr.fit(X_train, y_train)
        pred_gbr = gbr.predict(X_test)
        
        # 4. Modelo Avançado 2: Random Forest
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=6, 
            random_state=self.random_state
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        
        # Cálculo de Métricas
        models_preds = {
            "Baseline: Naive": pred_naive,
            "Baseline: Média Móvel (4 sem)": pred_moving_avg,
            "ML: Random Forest": pred_rf,
            "ML: Gradient Boosting": pred_gbr
        }
        
        metrics_list = []
        for name, preds in models_preds.items():
            wape = calculate_wape(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            metrics_list.append({
                "Modelo": name,
                "WAPE (%)": round(wape * 100, 2),
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "R² Score": round(r2, 3)
            })
            
        metrics_df = pd.DataFrame(metrics_list).sort_values(by="WAPE (%)", ascending=True)
        
        # DataFrame para plotagem comparativa
        comparison_plot_df = pd.DataFrame({
            "Data": test_df["ds"],
            "Vendas Reais": y_test,
            "Baseline (Média Móvel)": pred_moving_avg,
            "Gradient Boosting": pred_gbr,
            "Random Forest": pred_rf
        })
        
        # Importância de atributos do melhor modelo
        feature_importance = pd.DataFrame({
            "Atributo": feature_cols,
            "Importância": gbr.feature_importances_
        }).sort_values(by="Importância", ascending=False)
        
        return {
            "metrics": metrics_df,
            "plot_data": comparison_plot_df,
            "feature_importance": feature_importance,
            "train_size": len(train_df),
            "test_size": len(test_df)
        }
