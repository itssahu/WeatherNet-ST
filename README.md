# WeatherNet-ST
Spatiotemporal Short-Term Weather Forecasting  using XGBoost and ConvLSTM (Odisha).

This project uses XGBoost and ConvLSTM independently to forecast next-day temperature and precipitation over Odisha.
Both models were applied independently, allowing a comparison between a feature-based statistical learner (XGBoost) and a grid-based deep spatiotemporal model (ConvLSTM) for next-day t2m and tp forecasting.

### Data : 
ERA5 2m temperature, Total precipitation

Train: 2020-2021 | Validation: 2023 | Test: 2024

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### XGBoost Forecasting Architecture

               ┌─────────────────────────────────────────┐
               │        1. Build Features (x_t)           │
               │  lags + rolling + spatial + calendar     │
               │  → Convert raw ERA5 into feature vectors │
               │    (computed from full 2020–2024 data)   │
               └─────────────────────────────────────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────────────────────────┐
               │            2. XGBoost Model Architecture             │
               │  f(x) = Σ_{m=1}^K η·T_m(x)                           │
               │  Each tree T_m fits residuals: r_m = y − ŷ_{m−1}     │
               │  Regularization: Ω(T)=γ·(#leaves)+λ‖w‖²              │
               │  → Learns non-linear ERA5 patterns from 2020–2024    │
               │    using iterative boosting of decision trees        │
               └─────────────────────────────────────────────────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────────────┐
               │            3. Training (once)            │
               │  init → residual → tree → update         │
               │  → Trees iteratively fit 5-year targets  │
               │    (2020–2024 next-day t2m / tp)         │
               └─────────────────────────────────────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────────────┐
               │          4. Forecast 1 Jan 2025          │
               │  x_2025-01-01 → f(x) → prediction        │
               │  → Uses features built from last          │
               │    available day (31 Dec 2024)           │
               └─────────────────────────────────────────┘

          



XGBoost leverages five years of ERA5 history (2020–2024) to learn powerful nonlinear relationships between past weather patterns and next-day outcomes.
Instead of modeling sequences explicitly, it converts the climate signal into high-information features (lags, rolling means, spatial encodings, seasonality) and learns how they jointly drive tomorrow’s temperature/precipitation.

### Intuition 

XGBoost works by sequentially building small decision trees that learn only what the previous trees could not capture, i.e., the residual errors 
rm=y−y^m−1
rm=y−y^m−1
This forces each new tree Tm to correct specific local mistakes in space–time temperature patterns rather than relearn the full signal. Because the final model is a weighted sum of many such targeted trees 
f(x)=∑η Tm(x)
f(x)=∑ηTm(x), it becomes extremely good at modeling non-linear climate relationships, sharp gradients, localized effects, and interactions between predictors. The regularization term
Ω(T)=γ(#leaves)+λ∥w∥2 keeps individual trees simple, preventing overfitting even when training on high-dimensional ERA5 features. This makes XGBoost ideal for downscaling and error-correction tasks where the goal is to learn the fine-scale structure hidden inside coarse-grid climate variables.


The model builds an ensemble of gradient-boosted regression trees, where each new tree aggressively corrects the residual mistakes of the previous ones.
This boosting mechanism allows XGBoost to capture complex climate dependencies—threshold effects, nonlinear interactions, spatial gradients, far more efficiently than a single model.
Once trained, it becomes an extremely fast, stable, and high-accuracy forecaster, requiring only the latest feature row (from 31 Dec 2024) to produce a reliable next-day forecast for 1 Jan 2025.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ConvLSTM Forecasting Architecture  

           ┌───────────────────────────────────────────────┐
           │              1. Prepare Input Sequence         │
           │  7 days of ERA5 grids (2020–2024)             │
           │  X ∈ ℝ^(7 × 1 × H × W), normalized            │
           │  → Provides short-term temporal context        │
           └───────────────────────────────────────────────┘
                             │
                             ▼
           ┌───────────────────────────────────────────────┐
           │            2. ConvLSTM Architecture            │
           │  For each timestep t:                          │
           │    [x_t , h_(t−1)] → 3×3 Conv → tanh → h_t     │
           │                                                 │
           │  - Input channels: 1                            │
           │  - Hidden channels: 16                          │
           │  - Kernel size: 3×3 (padding=1)                 │
           │  - Recurrence captures space + time             │
           │  Output after 7 steps: h_7 ∈ ℝ^(16 × H × W)     │
           └───────────────────────────────────────────────┘
                             │
                             ▼
           ┌───────────────────────────────────────────────┐
           │               3. Output Projection              │
           │  1×1 Conv applied to h_7                       │
           │  ŷ_norm = Conv2D(h_7, out_channels=1)         │
           │  → Produces next-day grid (H × W)              │
           └───────────────────────────────────────────────┘
                             │
                             ▼
           ┌───────────────────────────────────────────────┐
           │           4. Unnormalization Step              │
           │  ŷ = ŷ_norm × σ_train + μ_train              │
           │  → Final forecast for target date              │
           │    (e.g., 2025-01-01)                          │
           └───────────────────────────────────────────────┘




| Stage                      | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| **1. Input Construction**  | Use 7 previous days of normalized climate grids.                           |
| **2. ConvLSTM Recurrence** | Each day updates a hidden state using convolution + recurrence.            |
| **3. Temporal Encoding**   | After 7 timesteps, the final hidden state encodes all spatiotemporal info. |
| **4. Output Conv Layer**   | A 1×1 convolution converts hidden state → next-day grid.                   |
| **5. Unnormalize**         | Convert normalized prediction back to °C or mm/day.                        |

#### ConvLSTM Recurrent Block — Intuition 

At each day t, the ConvLSTM updates its spatial memory using four steps. 

Step 1: It concatenates the current input grid with the previous hidden state along the channel dimension ( [x_t , h_{t−1}] ), letting the model see both “today’s weather” and “yesterday’s memory.” 

Step 2: A 3×3 convolution ( W * [x_t , h_{t−1}] ) extracts local spatial patterns such as gradients, anomalies, and moving systems. 

Step 3: A tanh nonlinearity ( tanh(·) ) compresses values to a stable range, allowing the model to learn nonlinear spatiotemporal interactions. 

Step 4: The output becomes the new hidden state ( h_t ), acting as an updated spatial memory map that carries information forward. 

Overall intuition: the ConvLSTM learns how weather structures evolve across space and time by repeatedly applying a convolutional memory update that blends today’s grid with accumulated past information.

Note: “The ConvLSTM uses 16 hidden channels, which act as 16 learned spatial feature maps.
In practice, these can capture patterns like large-scale temperature gradients, local anomalies, or propagating weather systems, but the model discovers these automatically; we do not manually assign them.”

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Relative Performance: 

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

Both model performs really well on forecasting temperature for next day  



