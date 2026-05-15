# Reporte Ejecutivo — Temporada 2025/26
### Arsenal & Tottenham Predictive Intelligence Platform
*Generado por el Dixon-Coles Multi-Liga Engine con Contextual Adjustment Layer*

---

## Resultados Principales

| Objetivo | Probabilidad |
|---|:---:|
| Arsenal — Premier League Title | **83.71%** |
| Arsenal — CL Final Win *(incl. ET/Penales)* | **47.08%** |
| Arsenal — Doblete Histórico *(PL + CL)* | **39.41%** |
| Tottenham — Riesgo de Descenso | **16.92%** |

> [!IMPORTANT]
> El doblete es el evento más difícil de cuantificar en el fútbol moderno. Una probabilidad del **39.41%** representa uno de los resultados más extraordinarios que un modelo estadístico puede proyectar para un club en la era moderna — comparable a los dobles del Manchester City (2018/19) y Chelsea (2009/10).

---

## CL Final: Arsenal vs Paris Saint-Germain
### Puskas Arena, Budapest — Sede Neutral

| Escenario | Arsenal | Empate | PSG |
|---|:---:|:---:|:---:|
| **90 Minutos** | 29.3% | 35.6% | 35.1% |
| **Incluyendo ET/Penales** | **47.1%** | — | **52.9%** |

> [!NOTE]
> La alta probabilidad de empate a 90 minutos (35.6%) es estadísticamente coherente con finales de alta presión donde ambos equipos juegan con cautela táctica. La proyección de ET/Penales asume un reparto 50/50, consistente con la literatura de sports analytics sobre penales en finales (Bar-Eli et al., 2007).

---

## Motor Contextual — Variables Aplicadas

El modelo va más allá de los datos históricos brutos. Para cada equipo en cada competición, se aplican multiplicadores calibrados sobre las tasas esperadas de gol (λ y μ del modelo Dixon-Coles) basados en fatiga, lesiones y motivación.

| Factor Contextual | Arsenal (PL) | Arsenal (CL) | Tottenham (PL) | PSG (CL) |
|---|:---:|:---:|:---:|:---:|
| Escala Ataque | 0.931 | 0.989 | 0.920 | 0.989 |
| Escala Defensa | 0.960 | 0.970 | 0.900 | 0.931 |
| Bonus Motivación *(log-space)* | +0.040 | +0.060 | +0.080 | +0.070 |
| Penalización Fatiga | 4% | 3% | 0% | 4% |

### Justificación por Equipo

**Arsenal — Premier League**
- Fatiga de campaña europea (13 partidos CL) → penalización 4% en sprint metrics (Mohr et al., 2005)
- Motivación de título compensa parcialmente → bonus +4.1% en expected goals
- Rotaciones de Arteta mitigan fatiga aguda (Havertz/Trossard cubriendo minutos)

**Arsenal — CL Final**
- Preparación máxima: 3+ semanas dedicadas post-PL para la final
- Primera final histórica del club → motivación sin precedentes (+6.1%)
- Fatiga residual de semifinal gestionada con descanso planificado

**Tottenham Hotspur**
- Sin competición europea → 0% de fatiga por fixture congestion
- Batalla de supervivencia en descenso → bonus de desesperación +8.3%
- Organización defensiva débil toda la temporada (GD = -9) → escala defensa 0.90
- Moral del plantel en mínimos históricos → mayor riesgo de errores bajo presión

**Paris Saint-Germain**
- Desesperados por primer título CL tras finales perdidas (2020 vs Bayern)
- Tridente Dembélé/Barcola/Ramos en forma → escala ataque 0.989
- Ligue 1 asegurada con antelación → concentración 100% en CL
- Vulnerabilidad defensiva bajo presión alta confirmada por xGA stats de la temporada
- High-pressing style (Luis Enrique) aumenta la carga física del plantel

---

## Arquitectura del Pipeline

```
FBref (Premier League)  -+
FBref (Ligue 1)         -+-> [Pandera Validation] -> [Combined Dataset: 657 matches]
                          |
                          v
                [Time-Decay Hyperparameter Tuning]
                  Grid Search: alpha in {0.001, 0.003, 0.0065, 0.010, 0.015}
                  Optimal alpha = 0.0100 (half-life ~69 days)
                          |
                          v
                [Dixon-Coles Model Fit -- Multi-Liga]
                  Parameters: attack_i, defense_i, home_adv, rho
                  Constraint: sum(attack_i) = 0 (identifiability)
                          |
                          v
              [Contextual Adjustment Engine]
                  TeamContext: attack_scale x defense_scale x exp(motivation)
                  MatchContext: neutral_venue flag for CL Final
                          |
                    +-----+------+
                    |            |
                    v            v
           [Monte Carlo]   [CL Final Prediction]
           100,000 iter.   Arsenal vs PSG @ Puskas Arena
           PL Season Sim.  (home_adv = 0.0)
                    |            |
                    +-----+------+
                          |
                          v
               [Executive Summary Output]
                  PL Title + CL Win + Double + Relegation
```

---

## Formulación Matemática

La verosimilitud del modelo Dixon-Coles con decaimiento temporal:

$$\mathcal{L} = \prod_{k} w_k \cdot \tau_{\rho}(x_k, y_k) \cdot \text{Poisson}(x_k|\lambda_k) \cdot \text{Poisson}(y_k|\mu_k)$$

Donde:
- $\lambda_k = e^{\alpha_i + \delta_j + \gamma}$ — goles esperados del equipo local
- $\mu_k = e^{\alpha_j + \delta_i}$ — goles esperados del equipo visitante
- $w_k = e^{-\alpha^* \cdot t_k}$ — peso de decaimiento temporal ($\alpha^* = 0.0100$)
- $\tau_\rho$ — corrección para marcadores bajos (0-0, 1-0, 0-1, 1-1)

Con ajuste contextual:
$$\lambda_{adj} = \lambda \cdot s_{att}^{home} \cdot s_{def}^{away} \cdot e^{m^{home}}$$

---

## Outputs para Power BI

| Archivo | Descripción |
|---|---|
| `data/processed/current_standings.csv` | Tabla de posiciones PL actual |
| `data/processed/remaining_fixtures_probs.csv` | Probabilidades 1X2 por partido restante |
| `data/processed/cl_final_probs.csv` | Probabilidades CL Final (90min + ET) |
| `data/processed/simulation_probabilities.csv` | Distribución completa del Monte Carlo (100K iter.) |
| `data/processed/executive_summary.csv` | 4 KPIs principales para tarjetas de métricas |

---

## Visualizaciones Generadas

| Chart | Descripción |
|---|---|
| `images/title_probabilities.png` | Top 5 candidatos al título PL (barras horizontales) |
| `images/cl_final_probabilities.png` | Donut chart CL Final (90 minutos) |
| `images/executive_summary.png` | Dashboard ejecutivo con 4 KPIs |

---

## Referencias

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Applied Statistics, 46(2), 265–280.
- Mohr, M., Krustrup, P. & Bangsbo, J. (2005). *Fatigue in soccer: A brief review.* Journal of Sports Sciences, 23(6), 593–599.
- Grinsztajn, L. et al. (2022). *Why tree-based models still outperform deep learning on tabular data.* NeurIPS 2022.
- Bar-Eli, M. et al. (2007). *Action bias among elite soccer goalkeepers: The case of penalty kicks.* Journal of Economic Psychology, 28(5), 606–621.
- Lago-Ballesteros, J. & Lago-Peñas, C. (2010). *Performance in team sports: Identifying the keys to success in soccer.* Journal of Human Kinetics, 25, 85–91.

---

*Modelo entrenado con datos hasta el 15 de Mayo de 2026.*
