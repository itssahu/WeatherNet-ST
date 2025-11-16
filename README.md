# WeatherNet-ST
Spatiotemporal Short-Term Weather Forecasting  using XGBoost and ConvLSTM (Odisha).

Data: ERA5 2m temperature, Total precipitation
Train: 2020-2021 | Validation: 2023 | Test: 2024

In May, XGBoost perform better than ConvLSTM for forecasting temperature(2m) across grids as can be seen in following figure:
<img width="6478" height="1765" alt="t2m_rmse_xgb_vs_convlstm_2024_05_01_relative" src="https://github.com/user-attachments/assets/5e49596d-7c1f-444b-bab5-ec5af4f83fc6" />

In August, ConvLSTM perform better than XGBoost for forecasting total precipitation across grids as can be seen in following figure:
<img width="6478" height="1765" alt="tp_rmse_xgb_vs_convlstm_2024_08_01_relative" src="https://github.com/user-attachments/assets/548c0899-1116-4a96-89c2-b6ae52a0085c" />

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Forecasting for 1 Jan 2025

<img width="6920" height="1632" alt="t2m_forecast_xgb_vs_convlstm_2025_01_01" src="https://github.com/user-attachments/assets/d05eef3c-7ec8-4113-bb79-e9cfc24520be" />

<img width="7212" height="1632" alt="tp_forecast_xgb_vs_convlstm_2025_01_01" src="https://github.com/user-attachments/assets/7f5cf9ca-d96f-46b6-9e1e-bdfec75a140c" />




