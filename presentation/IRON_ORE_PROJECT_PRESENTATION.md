# Iron Ore Forecasting Project Presentation

This deck is designed for a 12-slide presentation. The figure paths below point to generated assets in `presentation/figures/`.

## Slide 1 - Title
Title:
- End-to-End Iron Ore Price Forecasting and MLOps Workflow

Subtitle:
- From preprocessing and EDA to model registry, deployment, and research-aligned experiments

Key points:
- Problem: forecast iron ore prices under regime changes and market turbulence
- Scope: production ML pipeline plus paper-aligned research benchmarking
- Stack: DVC, MLflow, FastAPI, Docker, Jupyter notebooks

Visual:
- `presentation/figures/09_mlop_flow.png`

Speaker note:
- Open with the claim that this is both a forecasting project and a full MLOps system.

## Slide 2 - Iron Ore Market Dynamics
Title:
- Iron Ore Market Dynamics

Key points:
- Dataset contains 2,264 trading-day records from 2016-01-04 to 2024-12-31
- Core variables: Price, Open, High, Low, Volume, Change %
- The price series has long trends, abrupt reversals, and volatility bursts
- This makes static forecasting assumptions fragile

Visual:
- `presentation/figures/01_market_dynamics.png`

Speaker note:
- Use the trend chart to show that the market does not behave like a simple stationary series.

## Slide 3 - Data Ingestion, Cleaning, and EDA
Title:
- Data Ingestion, Cleaning, and EDA

Key points:
- Ingestion copies the raw CSV into a DVC-tracked raw data artifact
- Cleaning standardizes schema, parses dates, converts numeric fields, and normalizes `Vol.` and `Change %`
- Duplicates are removed, weekdays retained, and missing numeric values are forward/back filled
- EDA then checks trend, seasonality, volatility, and feature relationships

Visual:
- `presentation/figures/03_seasonality.png`

Speaker note:
- Emphasize that reliable preprocessing is what makes every later stage reproducible.

## Slide 4 - Volatility and Conditional Variance
Title:
- Volatility and Conditional Variance

Key points:
- Returns are not stable across time; volatility clusters in turbulent windows
- The research-aligned workflow tracks a 14-day rolling volatility proxy
- A 75th percentile threshold is used to split high-volatility vs low-volatility regimes
- EWMA variance error is recorded to quantify conditional-variance mismatch

Visual:
- `presentation/figures/02_volatility_regime.png`

Speaker note:
- This slide connects the paper theme of conditional variance to the diagnostics now added in the repo.

## Slide 5 - Linear Models Under Turbulence
Title:
- Linear Models Under Turbulence

Key points:
- After correcting the benchmark to rolling one-step evaluation, ARIMA and Holt-Winters remain strong baselines
- ARIMA RMSE is 1.862 and Holt-Winters RMSE is 1.860, but both still weaken in high-volatility windows
- Their high-volatility RMSE stays above 3.18, showing where linear structure alone becomes fragile
- This supports residual hybridization rather than discarding statistical models entirely

Visual:
- `presentation/figures/06_research_benchmark.png`

Speaker note:
- The point is not that linear models are useless; the point is that turbulence is where residual nonlinear correction starts to matter.

## Slide 6 - Machine Learning Approaches
Title:
- Machine Learning Approaches

Key points:
- Support Vector Regression formulates nonlinear forecasting as a convex optimization problem and builds a predictor in a kernel-induced high-dimensional feature space
- Forecast quality in SVR depends on the regularized epsilon-insensitive loss, kernel choice, hyperparameter calibration, and careful lag or rolling-feature design
- Random Forest and Gradient Boosting reduce variance through feature subsampling, bootstrap aggregation, and sequential tree refinement
- Core limitation: these methods partition feature space from contemporaneous predictors and do not natively preserve ordered temporal memory without explicit lag engineering
- In the benchmark, SVR is the strongest pure ML model at RMSE 1.9082, but the hybrid ARIMA + LSTM improves further to 1.8576

Key metrics:
- Best pure ML model in research benchmark: SVR
- SVR RMSE: 1.9082
- Hybrid ARIMA + LSTM RMSE: 1.8576

Visual:
- `presentation/figures/13_ml_model_comparison.png`

Speaker note:
- Explain that the hybrid ARIMA + LSTM edges out the pure ML models because it combines a statistical baseline with explicit sequential residual modeling.

## Slide 7 - Deep Learning for Sequence Modeling
Title:
- Deep Learning for Sequence Modeling

Key points:
- LSTM-style recurrent models differ from static ML methods because they internalize temporal state dynamics directly
- In this project, the residual LSTM stopped after 26 epochs, with the best validation loss reached at epoch 14
- The recurrent block models ARIMA residuals instead of raw price directly
- This converts the paper's ARIMA-LSTM idea into a real benchmarked experiment in the repo

Visual:
- `presentation/figures/14_arima_lstm_history.png`

Speaker note:
- Use the training curve to show that the LSTM component was actually trained and early-stopped, not just described conceptually.

## Slide 8 - The Hybrid Ensemble Philosophy
Title:
- The Hybrid Ensemble Philosophy

Key points:
- Hybrid models assume linear and nonlinear components should be learned separately
- Here, ARIMA provides the rolling one-step linear forecast and LSTM models the nonlinear residual structure
- Hybrid ARIMA + LSTM is the best research model with RMSE 1.8576 and R2 0.9760
- High-volatility RMSE falls to 3.1817, slightly better than ARIMA, Holt-Winters, and SVR
- The gain is small but consistent, which is typical of residual-correction hybrids

Visual:
- `presentation/figures/07_research_prediction_traces.png`

Speaker note:
- Use the trace to show how the hybrid stays close to the ARIMA baseline while correcting some residual misses.

## Slide 9 - Sequential Decomposition (ARIMA-LSTM)
Title:
- Sequential Decomposition (ARIMA-LSTM) as the Next Step

Key points:
- Step 1: fit ARIMA on the raw series to extract the linear component
- Step 2: compute residuals from ARIMA forecasts
- Step 3: train LSTM on residual sequences to capture nonlinear dependencies
- Step 4: combine ARIMA forecast and residual forecast into the final prediction

Project positioning:
- Current repo now benchmarks ARIMA-LSTM in the research stage
- What remains open is productionizing it and extending it with richer variance and break detection

Visual:
- `presentation/figures/08_sequential_decomposition.png`

Speaker note:
- This slide is your bridge between what the paper proposes and what your current project is ready to evolve into.

## Slide 10 - Performance Evaluation
Title:
- Performance Evaluation

Key points:
- Evaluation is not limited to RMSE
- Production stage reports RMSE, MAE, MAPE, and R2
- Research stage adds regime RMSE, rolling RMSE, procurement-cost proxy, and EWMA variance error
- Best research result is Hybrid ARIMA + LSTM with RMSE 1.8576 and procurement-cost proxy 2.22M
- This makes the evaluation closer to business and risk reality

Visual:
- `presentation/figures/10_production_fit.png`

Speaker note:
- Mention that procurement cost is a practical proxy for forecasting error impact, not just a technical metric.

## Slide 11 - Identified Research Gaps
Title:
- Identified Research Gaps

Key points:
- Conditional variance is diagnosed but not yet modeled explicitly with GARCH-family methods
- Structural break detection is not yet part of the pipeline
- ARIMA-LSTM is benchmarked, but it is not yet wired into production training, registry, or deployment
- Event or macroeconomic exogenous variables are not yet included
- Registry promotion is threshold-based, not yet risk-aware

Visual:
- `presentation/figures/06_research_benchmark.png`

Speaker note:
- This is where you show that the project is technically honest and understands its own limits.

## Slide 12 - MLOps Workflow and Deployment
Title:
- MLOps Workflow and Deployment

Key points:
- DVC stages: ingestion, cleaning, feature engineering, feature selection, training, registry, promotion, deployment
- MLflow tracks experiments and manages model versions
- Promotion gates move models across testing, staging, and production
- FastAPI serves predictions to the frontend; Docker packages the stack

Visual:
- `presentation/figures/09_mlop_flow.png`

Speaker note:
- Stress that the project does not stop at the notebook stage; it is deployable and reproducible.

## Slide 13 - Conclusion
Title:
- Conclusion

Key points:
- The project combines classical forecasting, machine learning, and MLOps in one workflow
- Production pipeline delivers a strong deployable model with Ridge on selected engineered features
- Research pipeline now includes a real ARIMA + LSTM residual hybrid with volatility-aware diagnostics
- The strongest next step is richer variance modeling and structural-break detection around that hybrid core

Closing line:
- The project is already operational, and the paper-guided extensions define a credible research roadmap.

Visual:
- `presentation/figures/01_market_dynamics.png`
