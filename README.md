# WeatherNet-ST
Spatiotemporal Short-Term Weather Forecasting  using XGBoost and ConvLSTM (Odisha).

Data: ERA5 2m temperature, Total precipitation
Train: 2020-2021 | Validation: 2023 | Test: 2024

### ConvLSTM Forecasting Architecture (Schematic) 

┌───────────────────────────────────────────────────────────────┐
│                      
                 INPUT SEQUENCE (7 DAYS)                         │
│  
     X ∈ ℝ^(7 × 1 × H × W)  → 7 grids of t2m/tp (lat×lon)        │
└───────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                  
                    CONVLSTM RECURRENT BLOCK                    │
│                                                               │
│        For each timestep t:                                        │
│                                                               │
│     [x_t , h_(t−1)]  ──►  3×3 Conv2D  ──►  tanh  ──►  h_t     │
│    

│
│   - Input channels: 1                                         │
│   - Hidden channels: 16                                       │
│   - Kernel size: 3×3 (padding=1)                              │
│   - h_0 = zeros                                               │
│       

│
│   Output after 7 steps: hidden state h_7 ∈ ℝ^(16 × H × W)     │
└───────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                
             OUTPUT PROJECTION (1×1 Conv)                  │
│                                                               │
│    ŷ_norm = Conv2D(h_7, out_channels=1, kernel=1×1)          │
│                                                               │
│    Output: 1 predicted grid (H × W) for next day              │
└───────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                 
             UNNORMALIZATION STEP     
                                                                 │
│      ŷ = ŷ_norm × σ_train + μ_train                            │
│                                                               │
│     Final output = Forecast for date (e.g., 2025-01-01)       │
└───────────────────────────────────────────────────────────────┘


| Stage                      | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| **1. Input Construction**  | Use 7 previous days of normalized climate grids.                           |
| **2. ConvLSTM Recurrence** | Each day updates a hidden state using convolution + recurrence.            |
| **3. Temporal Encoding**   | After 7 timesteps, the final hidden state encodes all spatiotemporal info. |
| **4. Output Conv Layer**   | A 1×1 convolution converts hidden state → next-day grid.                   |
| **5. Unnormalize**         | Convert normalized prediction back to °C or mm/day.                        |

“The ConvLSTM uses 16 hidden channels, which act as 16 learned spatial feature maps.
In practice, these can capture patterns like large-scale temperature gradients, local anomalies, or propagating weather systems, but the model discovers these automatically; we do not manually assign them.”



Relative Performance: 

<img width="420" height="71" alt="image" src="https://github.com/user-attachments/assets/dbffd264-434f-41aa-9bdb-e67c8fd71719" />

(blue = XGB better, red = ConvLSTM better, white = similar performance)

In May, XGBoost perform better than ConvLSTM for forecasting temperature(2m) across most of the grids as can be seen in following figure:

<img width="6478" height="1765" alt="t2m_rmse_xgb_vs_convlstm_2024_05_01_relative" src="https://github.com/user-attachments/assets/5e49596d-7c1f-444b-bab5-ec5af4f83fc6" />

In August, ConvLSTM perform better than XGBoost for forecasting total precipitation across most of the grids as can be seen in following figure:

<img width="6478" height="1765" alt="tp_rmse_xgb_vs_convlstm_2024_08_01_relative" src="https://github.com/user-attachments/assets/548c0899-1116-4a96-89c2-b6ae52a0085c" />

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Forecasting for 1 Jan 2025


<img width="387" height="39" alt="image" src="https://github.com/user-attachments/assets/93bb0445-d81c-4480-8577-a3eafa59299b" />

(red = XGB better, blue = ConvLSTM better, white = similar performance)

<img width="2300" height="600" alt="t2m_forecast_xgb_vs_convlstm_2025_01_01" src="https://github.com/user-attachments/assets/3a07497d-7591-409b-a77f-015b3da963ab" />

<img width="2300" height="600" alt="tp_forecast_xgb_vs_convlstm_mm_2025_01_01" src="https://github.com/user-attachments/assets/78baa017-63c8-46de-8fe6-0e972e9709be" />





