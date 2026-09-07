# Métricas de Test Adicionales (Trade-DQN y Predictive-DQN)

Estas métricas se calcularon sobre el subconjunto `split=test` (720 filas) usando:
- `data/processed/integrated_base.csv`
- `data/processed/x1_trade_recommendations.csv`
- `data/processed/x2_price_predictions.csv`

## Tamaño test
- Filas test: `720`

## Trade-DQN (x1) en test
- `unique_actions`: `2`
- `action_std`: `0.9963`
- `action_counts`: `{-1: 329, 1: 391}`
- `action_change_rate`: `0.0111`

Lectura rápida:
- No está colapsado a una sola acción (hay 2 acciones).
- Cambia poco de acción entre pasos (change rate bajo).

## Predictive-DQN (x2) en test
- `x2_nunique`: `3`
- `x2_std`: `16.2152`
- `MAE` (% retorno): `55.8163`
- `RMSE` (% retorno): `57.2331`
- `directional_accuracy`: `0.4979`

Lectura rápida:
- No está colapsado a un único valor, pero su calidad predictiva en test es baja.
- La precisión direccional es aproximadamente aleatoria (~50%).
