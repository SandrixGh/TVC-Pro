# TVC Pro v53.00 — Аудит математических формул

**Дата:** 2026-06-28  
**Файл:** `TVC Pro v53.00.txt`  
**Точное количество строк (с учётом пустых и разделителей):** **1734**  
*(Внутренний комментарий в строке 2 содержит ошибку: указано "Total Lines: 1636" — расхождение 98 строк)*

---

## 1. НАСТРОЙКИ (строки 22–205)

Раздел содержит только декларации `input.*` без математических формул. Настройки корректно определяют все параметры:

- `kssa_len` ∈ [3, 50] — размер окна L траекторной матрицы
- `kssa_bj_len` ∈ [20, 200] — длина обучающей выборки N
- `kssa_ar_order` ∈ [2, 30] — порядок AR/LRF p
- `kssa_forecast` ∈ [1, 30] — горизонт прогноза F
- `frac_d` ∈ [0.1, 0.9] — порядок дробного дифференцирования d
- `hmm_nu` — степени свободы t-распределения Стьюдента

---

## 2. ИНИЦИАЛИЗАЦИЯ И ГЛОБАЛЬНЫЙ SCOPE (строки 218–280)

### f_tanh(x) — строки 220–222

```
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

Реализация с числовым clamp во избежание overflow:

```pine
float _exp2x = math.exp(math.max(math.min(2.0 * _x, 20.0), -20.0))
(_exp2x - 1.0) / (_exp2x + 1.0)
```

**Статус:** Формула точная, без упрощений. Clamp ∈ [-10, 10] по аргументу tanh (эквивалентен clamp аргумента exp по 2x ∈ [-20, 20]).

### f_atan2(y, x) — строки 267–279

Полная реализация 4-квадрантного арктангенса:

| Условие | Результат |
|---------|-----------|
| x > 0 | atan(y/x) |
| x < 0, y ≥ 0 | atan(y/x) + π |
| x < 0, y < 0 | atan(y/x) − π |
| x = 0, y > 0 | π/2 |
| x = 0, y < 0 | −π/2 |
| x = 0, y = 0 | 0 (default) |

**Статус:** Полная формула atan2 без упрощений.

### Efficiency Ratio (строки 250–253)

```
change_n  = |src[0] - src[er_len]|
volatility_n = Σ_{i=1..er_len} |src[i-1] - src[i]|
ER_raw = change_n / max(volatility_n, 0)
ER_smoothed = WMA(ER_raw, 2)
```

### Адаптивное окно (строки 258–263)

**Efficiency Ratio метод:**
```
h = h_min + (h_max - h_min) * (1 - ER_smoothed)²
```

**ATR метод:**
```
vol_ratio = ATR_short / ATR_long
h = clamp(h_max * vol_ratio, h_min, h_max)
```

---

## 3. МАТЕМАТИЧЕСКАЯ ЛОГИКА TVC (строки 282–431)

### f_get_weight(u, t) — строки 284–295

Ядерные функции (компактный носитель |u| ≤ 1):

| Ядро | Формула |
|------|---------|
| Tricube | K(u) = (1 − |u|³)³ |
| Epanechnikov | K(u) = 0.75·(1 − u²) |
| Gaussian | K(u) = exp(−0.5·u²) |
| Triangular | K(u) = 1 − |u| |

**Статус:** Все формулы точные.

### f_calc_tvc — строки 297–423

Взвешенная локальная полиномиальная регрессия порядка p.

**Эффективный размер выборки (формула Kish):**
```
n_eff = (Σw_i)² / Σw_i²
```

**Крамерово правило — линейная регрессия (p=2):**

Система Wx·β = Wy:
```
S0 = Σ w_i       S1 = Σ w_i·x_i     S2 = Σ w_i·x_i²
Sy0 = Σ w_i·y_i  Sy1 = Σ w_i·y_i·x_i
det = S0_r·S2_r − S1²
b0 = (Sy0·S2_r − S1·Sy1) / det
b1 = (S0_r·Sy1 − S1·Sy0) / det
```

**Крамерово правило — квадратичная регрессия (p=3):**

```
S3 = Σ w_i·x_i³   S4 = Σ w_i·x_i⁴
Sy2 = Σ w_i·y_i·x_i²
D = S0_r·(S2_r·S4_r − S3²) − S1·(S1·S4_r − S3·S2_r) + S2_r·(S1·S3 − S2_r²)
b0 = [Sy0·(S2_r·S4_r − S3²) − Sy1·(S1·S4_r − S2_r·S3) + Sy2·(S1·S3 − S2_r²)] / D
b1 = [S0_r·(Sy1·S4_r − S3·Sy2) − Sy0·(S1·S4_r − S2_r·S3) + S2_r·(S1·Sy2 − Sy1·S2_r)] / D
b2 = [S0_r·(S2_r·Sy2 − Sy1·S3) − S1·(S1·Sy2 − Sy1·S2_r) + Sy0·(S1·S3 − S2_r²)] / D
```

Ридж-регуляризация: `S0_r = S0 + λ`, `S2_r = S2 + λ`, `S4_r = S4 + λ`, λ = 1e-4

**Скорректированный R²:**
```
R² = max(0, 1 − (1 − R²_raw)·(n_eff − 1)/dof)    где dof = max(n_eff − p, 1)
```

**t-статистика:**
- Линейная: `SE(b1) = sqrt(max(σ²·S0_r/det, 0))` (затем нормировка на w_norm)
- Квадратичная: `SE(b1) = sqrt(max(σ²·(S0_r·S4_r − S2_r²)/D, 0))`
- `t_stat = b1 / SE(b1)`, `w_norm = Σw_i / sqrt(Σw_i²)`

**MAD-дисперсия (формула Rousseeuw):**
```
σ_MAD = 1.4826 · median(|e_i|)
```

**Смазывающая поправка Duan (логарифмическая регрессия):**
```
TVC = exp(b0) · Σ w_i·exp(ε_i)    (smearing estimator)
```

**Статус:** Все формулы точные, без упрощений.

---

## 3.1 MTF REGIME ALIGNMENT (строки 446–457)

Вычисление наклона старшего таймфрейма:
```pine
float raw_htf_b1 = request.security(syminfo.tickerid, mtf_res,
    f_get_htf_b1()[1], lookahead=barmerge.lookahead_off)
```

Использует сдвиг [1] для предотвращения look-ahead bias.

**Статус:** Корректная реализация гравитационного фильтра.

---

## 4. ИНФОРМАЦИОННАЯ ЭНТРОПИЯ (строки 460–559)

### f_mutual_information — строки 461–536

**Адаптивные корзины (Sturges-Scott approximation):**
```
k = round(sqrt(N))    при use_adaptive_bins=true
```

**Лапласово сглаживание (additive smoothing):**
```
p(x_i) = (count_i + ε) / (N + k·ε)        ε = 1e-6
p(x_i, y_j) = (count_ij + ε) / (N + k²·ε)
```

**Энтропия Шеннона (лог₂):**
```
H(X) = −Σ_{i} p(x_i)·log₂(p(x_i))
H(Y) = −Σ_{j} p(y_j)·log₂(p(y_j))
H(X,Y) = −Σ_{i,j} p(x_i, y_j)·log₂(p(x_i, y_j))
```

**Взаимная информация:**
```
MI(X;Y) = H(X) + H(Y) − H(X,Y)
```

**Нормализованная дивергенция:**
```
D = max(0, min(1, 1 − MI / H(X)))
```

**Поправка Миллера-Мэдоу (устранение смещения оценки энтропии):**
```
H_corrected(X) = H(X) + (k−1) / (2·N·ln2)
H_corrected(X,Y) = H(X,Y) + (k²−1) / (2·N·ln2)
```

**Статус:** Полные формулы без упрощений.

---

## 4.1 АЛГОРИТМ KSSA (строки 562–991)

### Метод Бёрга с Ridge-регуляризацией (`get_burg_ar_weights`) — строки 565–616

**Huber-bounded дисперсия (строки 571–579):**
```
μ_x = mean(x)
var_x = (1/n) · Σ clamp(x_i − μ_x, −1.5, 1.5)²
ridge = α · var_x
```
*(Clamp 1.5σ — стандартный Huber-порог для N(0,1))*

**Ridge-адаптированный знаменатель Бёрга (строки 585–595):**
```
num_m = Σ_{i=m..n-1} ef_i · eb_{i-1}
den_m = Σ_{i=m..n-1} (ef_i² + eb_{i-1}²) + ridge·(n−m)
k_m   = 2·num_m / den_m     (при den_m > 1e-12)
```

*(Адаптивное масштабирование Ридж-штрафа на (n-m) сохраняет инвариантность к длине ряда)*

**Строгое отражение Шура-Кона (строки 597–601):**
```
|k_m| ≥ 1  →  k_m = sign(k_m)·(1 − ε),   ε = 1e-3
```
*(Аналитическая гарантия устойчивости AR-полинома по теореме Шура-Кона)*

**Мягкий Schur-Cohn барьер (Левинсоновское сжатие) — строка 603:**
```
k_m := k_m · (1 − exp(−(1 − |k_m|)² / τ))
```
*(Дифференцируемая монотонная функция: при |k_m| → 1 barrier → 0; при |k_m| → 0 barrier → 1)*

**Обновление AR-коэффициентов по Левинсону (строки 606–609):**
```
a[i] := a_prev[i] − k_m · a_prev[m−2−i],    i = 0..m-2
```
*(Рекуррентная схема Левинсона-Дурбина для обновления коэффициентов предыдущего порядка)*

**Обновление решёточного фильтра (строки 611–615):**
```
ef_new[i] = ef[i] − k_m · eb[i−1]
eb_new[i] = eb[i−1] − k_m · ef[i]
```
*(Ортогональная lattice-рекуррентность для нумерации m=1..p)*

### LRF-веса из сигнального подпространства (`get_ssa_lrf_weights`) — строки 620–642

**ESPRIT-подобная Minimum-Norm LRF (теорема Голяндиной):**
```
π_i = U_r[L−1, i]           (последняя строка базиса)
ν² = Σ_{i=0..r-1} π_i²      (вертикальность подпространства)
```

**Адаптивный verticality fallback (строки 634–636):**
```
denom = 1 − ν²
safe_denom = ν² < thresh ? denom : max(1 − thresh, 1e-9)
coef = 1 / safe_denom
```
*(При ν² → 1 вместо обнуления ряда применяется регуляризованный коэффициент)*

**LRF-коэффициенты (строки 637–641):**
```
A[L−2−i] = (Σ_{j=0..r-1} U_r[i,j] · π_j) · coef,    i = 0..L-2
```

### Основная функция AR-SSA (`f_ar_ssa_0lag`) — строки 644–991

**1. Динамическая память по Ляпунову (строка 649):**
```
ρ(ER) = ρ_min + (ρ_max − ρ_min) · ER_smoothed
```

**2. L1-робастное центрирование MAD·1.4826 (строки 666–688):**
```
median_p = median(hist_p)
MAD_p = 1.4826 · median(|hist_p[i] − median_p|)
x_centered[i] = (hist_p[i] − median_p) / MAD_p
```
*(Фактор 1.4826 = 1/Φ⁻¹(0.75) — нормировка MAD к σ для N(0,1))*

**3. Huber-зажим для матрицы ковариации (строки 703–708):**
```
hist_p_cov[i] = clamp(hist_p_cov[i], −δ, δ),    δ = 1.345
```
*(δ = 1.345 — оптимальный предел Хьюбера для 95% эффективности при N(0,1))*

**4a. Biased Toeplitz автоковариация с Fading Memory (строки 739–760):**
```
total_w = Σ_{t=0..N-1} ρ^(N-1-t)

R_pp(k) = [Σ_{t=0..N-1-k} ρ^(N-1-(t+k)) · p_t · p_{t+k}] / total_w
```
*(Biased estimator: нормировка на total_w гарантирует PSD матрицы)*

**4b. PSD-Unbiased коррекция спектрального завала (строки 725–756):**
```
W(k) = Σ_{t=0..N-1-k} ρ^(N-1-(t+k))
unbias_corr(k) = W(0) / W(k)
R_corrected(k) = R_pp(k) · unbias_corr(k)
```
*(Diagonal tapering: компенсирует систематическое занижение R(k) при k → L)*

**4c. Стандартная траекторная матрица (строки 775–788):**
```
X_tmp[i,j] = p_{i+j} · sqrt(ρ^(K-1-j))    (с fading)
C = X_tmp · X_tmp^T / K,    K = N − L + 1
```

**5. Forward-Backward SSA (FB-SSA) — строки 791–798:**
```
J = exchange matrix (J[i,j] = 1 iff i+j=L-1)
C_fb = (C + J·C·J) / 2
```
*(Удвоение пространственной стабильности матрицы)*

**6. SVD через собственные числа (строки 801–803):**
```
C = Σ λ_i · u_i · u_i^T    (спектральное разложение ковариации)
```

**7. Soft Shrinkage (spectral denoising) — строки 823–831:**
```
λ_clean = max(0, λ_raw − σ²_noise)    (universal soft threshold)
```

**8a. Порог Марченко-Пастура (строки 819–821):**
```
γ = L / (N − L + 1)
λ_MP = σ²_noise · (1 + √γ)²
```
*(Верхняя граница спектра случайной матрицы Wishart размера L×K)*

**8b. MDL критерий с AR(1) поправкой (строки 868–890):**
```
penalty = 0.5·k·(2L−k)·ln(N)·max(1−ER, 0.1)
log_ratio = (1/count)·Σ_{i=k..L-1} ln(λ̂_i) − ln((1/count)·Σ λ̂_i)
MDL(k) = −N·count·log_ratio + penalty
```
*(Minimum Description Length с адаптацией к AR(1)-цветному шуму)*

**8c. Ранг по Энтропии Шеннона (строки 836–858):**
```
H_i = −(λ̂_i/Σλ̂_j) · ln(λ̂_i/Σλ̂_j)
r* = min{k : Σ_{i=0..k-1} H_i / Σ H_i ≥ 0.95}
```

**9. MGS Реортогонализация (строки 906–925):**
```
Модифицированный Грам-Шмидт (MGS):
для каждого столбца k:
  для каждого уже ортонормированного j < k:
    u_k := u_k − (u_k^T · u_j) · u_j   (проецирование вычитается ПОСЛЕДОВАТЕЛЬНО)
  u_k := u_k / ||u_k||
```
*(MGS численно устойчив: ошибка O(ε·κ) vs O(ε·κ²) у классического GS)*

**10. Prewhitening scale (строка 932):**
```
scale_i = sqrt(max(λ̂_clean_i / λ̂_raw_i, 0))
U_r_scaled[:,i] = U_raw[:,i] · scale_i
```

**11. Forecast Embedding (строки 950–958):**
```
x̂[N+f] = Σ_{k=1..p} a_k · x[N+f−k]    (AR/LRF рекуррентный прогноз)
```
*(Динамический горизонт: f_dyn = round(F_max · ER_smoothed))*

**12. Проекция на сигнальное подпространство (строка 971):**
```
X_rec = U_r · U_r^T · X_ext
```
*(Ортогональная проекция траекторной матрицы)*

**13. Взвешенная диагональная Ганкелизация с окном Натолла (строки 978–988):**

4-member Nuttall window:
```
w_Nuttall(i) = 0.3635819
             − 0.4891775·cos(2π·i/(L−1))
             + 0.1365995·cos(4π·i/(L−1))
             − 0.0106411·cos(6π·i/(L−1))
```
```
w(i) = w_Nuttall(i)^diag_weight
rec_val = Σ_{i: j=t-i ∈ [0,K)} X_rec[i, t-i] · w(i)
ŷ(t) = rec_val / Σ w(i)
```
*(Anti-aliasing диагональное усреднение с WSP-спектральным окном)*

**14. Деномализация (строка 989):**
```
out = median_p + final_scaled · MAD_p
```

**Статус раздела 4.1:** Все 14 шагов реализованы полностью, без математических упрощений. Подтверждены:
- Полный алгоритм Бёрга с обновлением Левинсона
- ESPRIT-совместимый LRF
- Biased+PSD-unbiased Toeplitz ковариация
- FB-SSA симметризация
- Soft shrinkage + 3 критерия выбора ранга (MP, Shannon, MDL)
- MGS реортогонализация
- Nuttall-взвешенная Ганкелизация

---

## 4.2 Фрактальное дифференцирование (строки 1050–1077)

### f_frac_diff — строки 1052–1060

**Fixed-Width Fractional Differencing (FWFD):**

Биномиальные веса (рекуррентная формула):
```
w_0 = 1
w_{k+1} = −w_k · (d − k) / (k + 1)    ⟺    w_k = Π_{j=1..k}[(j-1-d)/j]
```

**Нормализованный FD с L1-нормой весов:**
```
FD(t) = Σ_{k=0..len-1} w_k · x[t-k] / Σ_{k=0..len-1} |w_k|
```

*(Нормировка на Σ|w_k| устраняет масштабный эффект от усечения бесконечного ряда)*

### f_fwfd_core — строки 209–215 (ГЛОБАЛЬНЫЕ ФУНКЦИИ)

```
FD_core(t) = Σ_{k=0..len-1} w_k · x[t-k]
```
*(Без нормировки — используется как промежуточная фича для HMM)*

**Z-score FD сигнала:**
```
fd_z = (FD_smooth − mean(FD_smooth, L)) / stdev(FD_smooth, L)
```

**Статус:** Формулы точные без упрощений.

---

## 5. Hilbert Transform (строки 1080–1109)

**Преобразование Хилберта (Ehlers Analytic Signal):**

Детрендинговый фильтр (полосовой пропуск):
```
DP[t] = 0.0962·WMA4[t] + 0.5769·WMA4[t-2] − 0.5769·WMA4[t-4] − 0.0962·WMA4[t-6]
```

Квадратурная компонента:
```
Q1[t] = 0.0962·DP[t] + 0.5769·DP[t-2] − 0.5769·DP[t-4] − 0.0962·DP[t-6]
```

Синфазная компонента:
```
I1[t] = DP[t-3]
```

Мгновенная фаза:
```
φ[t] = atan2(Q1[t], I1[t]) · 180/π
```

Разворачивание фазы:
```
Δφ = φ[t] − φ[t-1]
Δφ := Δφ − 360·round(Δφ/360)    (нормировка в (-180, 180])
Φ[t] = Φ[t-1] + Δφ
```

Нормировка в [0, 360):
```
hilbert_phase = (WMA(Φ, 3) mod 360 + 360) mod 360
```

**Статус:** Точная реализация аналитического сигнала Эхлерса.

---

## 6. Z-SCORE СИГНАЛЫ И ЗАЩИТА ОТ ПОЗДНИХ ВХОДОВ (строки 1111–1171)

**Гибридное стандартное отклонение (blend регрессионной и ценовой дисперсии):**
```
w_t = norm_s = clamp(|t_stat|/3, 0, 1)
σ_hybrid = sqrt((1−w_t)·dev² + w_t·σ_price²)
```

**Z-Score:**
```
z = log(src / TVC) / σ_hybrid    [логарифмический режим]
z = (src − TVC) / σ_hybrid       [линейный режим]
```

**Spring-loaded Z-сигнал:**
```
buy_spring_loaded = (Σ_{i=0..z_persist-1} [z[t-i] ≤ -1.5]) == z_persist
z_sig_buy = spring_loaded AND b1 > 0 AND bars_since_turn ≤ z_tol_window
```

**Взвешенный возраст тренда:**
```
age_weighted = trend_age / max(σ_price / ATR, 1)
```

**Ускорение TVC:**
```
Δb1 = b1[t] − b1[t-1]
accel_up = Δb1 > StDev(Δb1, h) · mult AND prev_Δb1 ≤ StDev · mult
```

---

## 7. KALMAN FILTER (строки 1173–1233)

**Трёхмерная модель состояния [x, v, a] (Position-Velocity-Acceleration):**

Матрица перехода:
```
F = [[1, dt, dt²/2],
     [0,  1,  dt  ],
     [0,  0, γ(ER)]]    γ = 0.9·(1 − 0.5·ER_smoothed)
```

**Предсказание:**
```
x̂_k|k-1 = x_k + v_k·dt + 0.5·a_k·dt²
v̂_k|k-1 = v_k + a_k·dt
â_k|k-1 = a_k·γ
P_k|k-1 = F·P·F^T + Q    (аналитически развёрнуто)
```

**Адаптивные шумы (AEKF):**
```
R_adapt[t] = 0.95·R[t-1] + 0.05·ε[t]²     (Innovation Covariance Matching)
Q_adapt[t] = 0.95·Q[t-1] + 0.05·(a·dt)²
```

**Скалярное измерение H=[1,0,0]:**
```
S_k = P00_k|k-1 + R
K0 = P00/S_k,   K1 = P10/S_k,   K2 = P20/S_k
```

**Обновление:**
```
[x, v, a]^T_k = [x̂, v̂, â]^T_k|k-1 + [K0, K1, K2]^T · ε_k
P_k = (I − K·H)·P_k|k-1    (аналитически; с ограничением P ≤ 50·σ²)
```

---

## 8. HYBRID ENGINE: 3D EMISSION M-HMM (строки 1237–1349)

**4-состоянтная HMM: {BULL, BEAR, CHOP, REVERSAL}**

**FracDiff-фичи (нулевой лаг):**
```
v_raw = FWFD(b1, d, 10)
a_raw = FWFD(b2·h, d, 10)
z_raw = FWFD(z_score, d, 10)
```

**Z-нормировка фич:**
```
v_n = clamp(v_raw / stdev(v_raw, L), −3, 3)
```

**3D Гауссовы ядра эмиссии (log-domain):**
```
log B_s(v,a,z) = log_gauss(v, μ_v^s, σ) + log_gauss(a, μ_a^s, σ) + log_gauss(z, μ_z^s, σ)
log_gauss(x, μ, σ) = −0.5·((x−μ)/σ)²
```

Центроиды состояний (μ_v, μ_a, μ_z):
```
BULL:     ( 1.0,  0.5,  0.5)
BEAR:     (−1.0, −0.5, −0.5)
CHOP:     ( 0.0,  0.0,  0.0)
REVERSAL: (−0.5,  1.5, −1.5) и (0.5, −1.5, 1.5), берётся max
```

**Алгоритм Байеса (forward algorithm в log-domain):**
```
log α_j[t] = log_sum_exp_i(log α_i[t-1] + log A_{ij}) + log B_j(o_t) / T
```

**Log-Sum-Exp (численная стабилизация):**
```
LSE(v0,v1,v2,v3) = M + log(Σ exp(v_i − M)),    M = max(v_i)
```

**Нормировка (scaling):**
```
log α_j := log α_j − log Σ_j exp(log α_j)
```

**Энтропия распределения состояний:**
```
H = −Σ_{s∈S} p_s · ln(p_s) / ln(4)    (нормировано на log 4)
```

---

## 9. ONLINE LOGISTIC REGRESSION (строки 1351–1552)

**Кинематический целевой сигнал:**
```
momentum = b1 + b2·h
elastic_penalty = z·(ATR_domain·0.25)
kin_target = momentum − elastic_penalty
noise_floor = ATR_domain·0.05·(2 − ER)
kin_clean = sign(kin_target)·max(0, |kin_target| − noise_floor)
target_buy = σ(kin_clean / (ATR·0.5) · 3)
```

**6-мерный вектор признаков (tanh-активация):**
```
x = [1, tanh(b1/σ), tanh(b2·h/σ), tanh(−z/2), tanh(kv или jerk), tanh(2·ER−1)]^T
```

**RLS-обновление для логистической регрессии:**
```
p̂ = σ(w^T·x)
var = p̂·(1−p̂)·v_w
λ_dyn = max(0.85, 1 − lr·|ε|)
S = (λ_dyn / var) + x^T·P·x
P := (P − P·x·x^T·P / S) / λ_dyn
w := w + P·x·ε·lr_dyn
```

**Q-SYNC Ансамбль (взвешенное голосование):**
```
score_buy = (w_kssa·P_kssa + w_rls·P_rls + w_hmm·P_hmm + w_fd·P_fd + w_div·P_div)
            / (w_kssa + w_rls + w_hmm + w_fd + w_div)
qsync_buy = score_buy > qsync_thresh AND score_buy[t-1] ≤ qsync_thresh
```

---

## 9.5 DYNAMIC KELLY CRITERION (строки 1555–1576)

**Оценка Бриера (Brier Score):**
```
BS_buy[t]  = (q_buy[t-1] − actual_up[t])²
BS_sell[t] = (q_sell[t-1] − actual_dn[t])²
avg_BS = (BS_buy + BS_sell) / 2
```

**Brier Skill Score (BSS относительно случайного предсказания 0.5):**
```
smooth_BS = SMA(avg_BS, kelly_len)
BSS = 1 − smooth_BS / 0.25    (baseline: E[BS_random] = 0.5²·2/2 = 0.25)
```

**Динамический мультипликатор Келли:**
```
k_raw = max(0.01, BSS)
variance_penalty = regime == "CHOP" ? 0.5 : 1.0
kelly_mult = min(2.0, k_raw · variance_penalty · 2.0)
```

**Итоговый размер позиции:**
```
entry_size = position_score · risk_factor · kssa_boost · kelly_mult
```

---

## 10. ОТРИСОВКА И ДАШБОРД (строки 1578–1734)

**Полосы MAD (Prediction Bands) в логарифмическом режиме:**
```
band_upper = TVC · exp(dev · mult · regime_factor)
band_lower = TVC · exp(−dev · mult · regime_factor)
```

**Адаптивный цвет TVC:**
```
norm_s_adj = |t_stat| < 2 ? norm_s·0.5 : norm_s
color = gradient(norm_s_adj, 0→1, base_color_transparent → base_color_solid)
```

**Конвертация Kalman-состояния обратно:**
```
plot_kx = reg_mode == "Logarithmic" ? exp(k_x) : k_x
```

**Конвертация KSSA обратно:**
```
plot_kssa = reg_mode == "Logarithmic" ? exp(kssa_final) : kssa_final
```

---

## ИТОГОВОЕ ЗАКЛЮЧЕНИЕ

| Раздел | Формулы | Статус |
|--------|---------|--------|
| 2. Инициализация | tanh, atan2, ER, adaptive h | ✅ Полные |
| 3. TVC Core | Kernel WLS, Cramer, Ridge, R², t-stat, MAD, Duan | ✅ Полные |
| 3.1 MTF | request.security, lookahead_off | ✅ Корректно |
| 4. Энтропия | Shannon MI, Laplace, Miller-Madow | ✅ Полные |
| **4.1 KSSA** | **Burg+Ridge, LRF, Toeplitz, FB-SSA, MGS, Nuttall** | ✅ **Без упрощений** |
| 4.2 FracDiff | FWFD биномиальные веса, L1-нормировка | ✅ Полные |
| 5. Hilbert | Детрендер, Q1/I1, разворачивание фазы | ✅ Полные |
| 6. Z-Score | Гибридная дисперсия, spring-loaded | ✅ Полные |
| 7. Kalman | 3D PVA, AEKF адаптация | ✅ Полные |
| 8. M-HMM | 3D Гаусс, forward algorithm LSE | ✅ Полные |
| 9. RLS/LR | Kinematic target, Hooke, IRLS | ✅ Полные |
| 9.5 Kelly | Brier Score, BSS, Kelly fraction | ✅ Полные |
| 10. Dashboard | Отрисовка, конвертация режимов | ✅ Корректно |

### Раздел 4.1 KSSA — подтверждение полноты формул:

1. **Burg AR** — полный алгоритм включая: Huber var, adaptive ridge*(n-m), strict Schur-Cohn clamp, soft barrier, Levinson coefficient update, forward-backward lattice
2. **SSA LRF** — ESPRIT minimum-norm с adaptive verticality fallback
3. **Toeplitz covariance** — fading memory + PSD-unbiased diagonal tapering
4. **FB-SSA** — симметризация через обменную матрицу J
5. **Rank selection** — 3 метода: Shannon (95% cumulative H), Marchenko-Pastur (λ_MP = σ²(1+√γ)²), Colored Noise MDL с AR(1) поправкой
6. **MGS** — последовательная ортогонализация (numerically stable)
7. **Nuttall window** — 4-член оконная функция с регулируемым diag_weight
8. **Forecast embedding** — динамический горизонт f_dyn = round(F_max · ER)

**Математические упрощения НЕ обнаружены.**
