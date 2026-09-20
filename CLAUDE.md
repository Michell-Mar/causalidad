# causalidad

Estudio de inferencia causal: **efecto de la sequía sobre el rendimiento agrícola municipal en México** (maíz), 2016–2024. Unidad de observación: municipio × año agrícola × ciclo (OI/PV) × modalidad hídrica (Riego/Temporal).

## Entorno

Los notebooks corren con el **Python de Anaconda** (`/opt/anaconda3/bin/python`, 3.12.2), **no** con el venv `causalidad_env` del repo — a ese le faltan `openpyxl`, `statsmodels` y `scikit-learn`. Anaconda trae todo lo necesario (geopandas, statsmodels, sklearn, openpyxl, scipy).

Para ejecutar/validar un notebook desde la terminal:
```
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

## Pipeline (correr en este orden)

| Notebook | Qué hace | Output principal |
|---|---|---|
| `01_limpieza_datos_lluvia` | Limpieza de datos de lluvia | `data/02_processed/datos_lluvia.csv` |
| `01_limpieza_datos_sequia` | Limpieza del Monitor de Sequía (D1–D4) | `data/02_processed/datos_sequia.csv` |
| `01_limpieza_datos_temp_min` | Limpieza de temperatura mínima | `data/02_processed/datos_temp_min.csv` |
| `01_limpieza_datos_agricolas` | Limpieza de datos agrícolas SIAP (siembra/cosecha/producción) | `data/02_processed/datos_agricolas_limpios.csv` |
| `01_confusores` | Confusores térmicos vía Daymet V4 (API single-pixel). Incluye anomalías (`tmean_ciclo_anom`, `gdd_anom`, `dias_helada`) y variables antecedentes (`tmean_prev90*`, `tmean_preseason_anom`). Cachea en `data/01_raw/daymet_cache/` — borrar la caché si cambia `FETCH_INI`/`FETCH_FIN` | `data/02_processed/confusores_temp.csv` |
| `01_altura_promedio` | Altitud media/desviación por municipio (OpenTopoData SRTM). Cachea en `data/01_raw/altitud_cache/` | `data/02_processed/altura_municipios.csv` |
| `01_cuencas_acuifero` | Cruce **administrativo** (no espacial) de disponibilidad de acuíferos y cuencas por municipio. Ver detalle abajo | `data/02_processed/cuencas_acuifero_municipio.csv` |
| `02_join_data` | Une todo lo anterior en la tabla de modelado (grano municipio×año×ciclo×modalidad) | `data/02_processed/tabla_modelo.csv` |
| `03_EDA` | Análisis exploratorio de `tabla_modelo.csv`: completitud, balance del panel, mecanismo tasa de siniestro por modalidad, correlaciones/redundancia, descomposición de varianza between/within | — (solo gráficos/hallazgos) |
| `04_inferencia_causal` | Estimación del efecto causal (ver más abajo) | — (resultados en el notebook) |

### `01_cuencas_acuifero` — multi-año

Los xlsx de CONAGUA (acuíferos: 2015/2018/2020/2023; cuencas: 2016/2020/2023) no cubren el panel 2016–2024 ni coinciden entre sí. Cada año del panel usa el año-fuente CONAGUA **más próximo ≤ ese año** (`anio_mas_proximo`, sin mirar al futuro; mapeo en `cuencas_acuifero_mapa_anios.csv`). Grano resultante: municipio × año (2478×9 = 22 302 filas).

**Caveat de datos importante**: CONAGUA no publicó la columna `Condición` (SOBREEXPLOTADO/SUBEXPLOTADO) en los xlsx 2015 y 2018 → `tiene_acuifero_sobreexplotado` sale en 0 por **dato faltante**, no por ausencia real, en los años 2016‑2019 del panel. La flag `acuifero_condicion_ok` (0 en 2016‑2019, 1 en 2020‑2024) marca esto. No se incluye como covariable en `04` (es un escalón determinista de `anio`, colineal con los EF de año); en su lugar se usa como corte de una prueba de sensibilidad (ver abajo).

Nota técnica: usar el helper `_s(col) = col.fillna('').astype(str)` en cualquier cadena `.str...`, no `col.astype(str)` — en pandas ≥3 una columna float64 totalmente NaN no se convierte a string con `.astype(str)` y rompe el accessor `.str`.

### `02_join_data` — tabla de modelado

`tabla_modelo.csv`: 40 267 filas × 56 columnas. Columnas clave:
- **Outcome**: `rendimiento_real` = `volumenproduccion / cosechada` (definición SIAP correcta; **no** dividir entre `cosechada − siniestrada`, eso doble-resta el siniestro).
- **Calidad del outcome**: `tasa_siniestro` = `siniestrada/sembrada`; `siniestro_ok` = tasa_siniestro < 50 % (corte **fijo**, no percentil — decisión explícita del usuario).
- **Tratamiento**: `nivel_sequia_max` (D1–D4 del Monitor de Sequía en el ciclo).
- **Confusores**: `nivel_sequia_prev90_max` (sequía antecedente), variables Daymet (`tmean_ciclo_anom`, `gdd_anom`, `dias_helada`, `tmean_prev90_anom`, `tmean_preseason_anom`, `tmean_normal`), `altitud_media_m`/`altitud_std_m`, `acuifero_disp`/`cuencas_disp`/`tiene_acuifero_sobreexplotado`/`tiene_cuenca_sin_disp` (ahora varían por año) + `acuifero_condicion_ok`.
- **Flags de completitud por fila**: `clima_ok`, `prev90_ok`, `altitud_ok`, `hidro_ok`, `siniestro_ok` → combinadas en `apto`.

## `04_inferencia_causal` — diseño

- **Tratamiento** `trat` (binario): 1 si `nivel_sequia_max ∈ {1,2,3,4}`, 0 si `{0,-1,no clasificado}`.
- **Outcome**: `log(rendimiento_real)` (primario), también en niveles (t/ha).
- **Moderador**: `nommodalidad` (Riego/Temporal) — CATE por estrato.
- **Colisionador** (excluido del ajuste): `superficie_cosechada` — es el denominador de `Y`.
- **Confusores ajustados**: sequía antecedente, térmicos, topográficos, estrés hídrico estructural (`acuifero_disp`, `cuencas_disp`, `tiene_acuifero_sobreexplotado`, `tiene_cuenca_sin_disp`), `log(sembrada)`, EF de año/municipio, `nomcicloproductivo`, `nommodalidad`.
- **Estimadores**: diferencia cruda, regresión/g-cómputo, IPW estabilizado, AIPW doblemente robusto con cross-fitting (GBM), TWFE (municipio+año), tratamiento ordinal, E-value, placebo (tratamiento aleatorizado), sensibilidad a `acuifero_condicion_ok` (restringiendo a 2020‑2024).

### Hallazgo actual (última ejecución)

Muestra: 39 155 filas, 2 379 municipios, 57.6 % tratados. Asociación cruda +7.7 % (confundida).

- **AIPW doblemente robusto**: −0.4 % (IC −1.3 %, +0.5 %) — **no significativo**.
- **TWFE (municipio+año)**: **−1.6 %** (p<0.001) — significativo y estable en todas las variantes probadas.
- **Ordinal (por nivel 0‑4)**: **−0.72 %/nivel** (IC excluye 0) — dosis‑respuesta.
- **AIPW en niveles**, **AIPW soporte común** y **AIPW restringido a 2020‑2024** dan signos/magnitudes distintos entre sí (+3.5 %, −1.2 %, +2.1 %, todos "significativos") → el AIPW es **inestable** ante estas covariables hídricas (que tienen poca varianza *within*-municipio, ver `03_EDA`), mientras que el TWFE no las usa y por eso no cambia.
- **Conclusión vigente**: la evidencia más creíble es **TWFE + ordinal** (efecto negativo, ~1 %, con dosis‑respuesta); el AIPW transversal no aporta evidencia consistente en esta versión de los datos. No hay evidencia robusta de moderación por modalidad de riego. E‑value ≈ 1.07 (robustez limitada a confusión no medida).

## Convenciones de trabajo

- Editar notebooks vía manipulación directa del JSON (`json.load`/`json.dump` con `indent=1, ensure_ascii=False`) y luego re-ejecutar completo con `jupyter nbconvert --execute --inplace` — no editar celdas de salida a mano.
- Siempre releer el notebook desde disco antes de editarlo (el usuario puede haberlo corrido/editado en su propio kernel entre turnos).
- Antes de recomendar cambios en `01_cuencas_acuifero`/`02_join_data`, recordar que el merge de `cuencas_acuifero_municipio.csv` debe hacerse por `['idestado','idmunicipio','anio']`, no solo por municipio (si se omite `anio` colapsa silenciosamente a un año arbitrario).
