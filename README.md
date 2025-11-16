# WeatherNet-ST
Spatiotemporal Short-Term Weather Forecasting  using XGBoost and ConvLSTM (Odisha).

Data: ERA5 2m temperature, Total precipitation
Train: 2020-2021 | Validation: 2023 | Test: 2024

In May, XGBoost perform better than ConvLSTM for forecasting temperature(2m) across grids as can be seen in following figure:
<img width="6478" height="1765" alt="t2m_rmse_xgb_vs_convlstm_2024_05_01_relative" src="https://github.com/user-attachments/assets/5e49596d-7c1f-444b-bab5-ec5af4f83fc6" />

In August, ConvLSTM perform better than XGBoost for forecasting total precipitation across grids as can be seen in following figure:
<img width="6478" height="1765" alt="tp_rmse_xgb_vs_convlstm_2024_08_01_relative" src="https://github.com/user-attachments/assets/548c0899-1116-4a96-89c2-b6ae52a0085c" />

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecasting for 1 Jan 2025

$$
AE\\_XGB = |t2m\\_pred - t2m\\_actual|, \quad
AE\\_ConvLSTM = |t2m\\_pred\\_conv - t2m\\_actual|, \quad
AE\\_diff = AE\\_ConvLSTM - AE\\_XGB
$$


<img width="2300" height="600" alt="t2m_forecast_xgb_vs_convlstm_2025_01_01" src="https://github.com/user-attachments/assets/3a07497d-7591-409b-a77f-015b3da963ab" />

<img width="2300" height="600" alt="tp_forecast_xgb_vs_convlstm_mm_2025_01_01" src="https://github.com/user-attachments/assets/78baa017-63c8-46de-8fe6-0e972e9709be" />





