# Inteligencia de Enjambre — Tres ejemplos de optimización aplicada

Repositorio de los ejemplos desarrollados para el curso de Inteligencia de Enjambre.
Cada cuaderno resuelve un problema de optimización distinto con un algoritmo bioinspirado,
implementado **desde cero** (sin librerías de metaheurísticas), y recorre completo el ciclo
de seis etapas del algoritmo de enjambre.

| # | Cuaderno | Algoritmo | Qué se optimiza | Dimensión |
|---|---|---|---|---|
| 1 | `1_Gruber_Feature_Selection_ABC.ipynb` | ABC (Artificial Bee Colony) | Subconjunto de características de un dataset | 30 (binaria) |
| 2 | `2_Gruber_PSO_Hyperparameter_Tuning.ipynb` | PSO | Hiperparámetros `C` y `gamma` de una SVM-RBF | 2 (continua) |
| 3 | `3_Gruber_PSO_NN_sin_backpropagation.ipynb` | PSO | Todos los pesos y bias de un MLP de 3 capas | 85 / 137 (continua) |

Todos corren en **Google Colab** sin instalar nada: `Abrir en Colab → Entorno de ejecución → Ejecutar todo`.
Dependencias: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (ya incluidas en Colab).

---

## 1. ¿Qué se entiende aquí por "optimización"?

En los tres casos el problema se plantea de la misma forma:

```
encontrar  x*  ∈  Espacio de búsqueda
tal que    f(x*)  sea óptimo
```

donde `f` es una **función de caja negra**: no se conoce su fórmula analítica, no tiene
gradiente disponible y cada evaluación es cara (implica entrenar y validar un modelo).
Por eso no sirven los métodos basados en derivadas ni la búsqueda exhaustiva, y se recurre
a una **población de soluciones candidatas que comparten información** entre sí.

### Las seis etapas, aplicadas a cada ejemplo

| Etapa | Ejemplo 1 (ABC) | Ejemplo 2 (PSO) | Ejemplo 3 (PSO) |
|---|---|---|---|
| **1. Representación** | Vector real en `[0,1]^30` binarizado con umbral 0.5 → máscara de características | Vector `[log₁₀C, log₁₀γ]` | Vector plano con todos los pesos y bias de la red |
| **2. Inicialización** | 15 fuentes uniformes en `[0,1]^30` | 12 partículas uniformes dentro de los límites | 40 partículas uniformes en `[-2.5, 2.5]^D`, velocidades pequeñas |
| **3. Aptitud** | `α(1-acc_CV) + (1-α)·n_sel/D` (minimizar) | Exactitud media en CV de 5 pliegues (maximizar) | Entropía cruzada + regularización L2 (minimizar) |
| **4. Comportamiento** | Abejas empleadas / observadoras / exploradoras | Inercia + componente cognitiva + componente social | Igual que 2, más reinicio de partículas estancadas |
| **5. Evolución** | Ciclos: empleadas → observadoras → memorizar → exploradoras | Iteraciones con inercia decreciente 0.9→0.4 | Iteraciones con inercia decreciente y registro de diversidad |
| **6. Finalización** | Máximo de ciclos (40) | Máximo de iteraciones o estancamiento (`patience`) | Máximo de iteraciones, tolerancia alcanzada o estancamiento |

---

## 2. Cómo está optimizado cada programa

Además de *resolver* problemas de optimización, los cuadernos aplican decisiones concretas
para que la búsqueda sea **más barata, más estable y más honesta**. Esto es lo que hace que
los programas funcionen en Colab en minutos y no en horas.

### 2.1 Reducir el costo de la función de aptitud

La aptitud es el cuello de botella: cada llamada entrena un modelo completo.

- **Caché de evaluaciones (Ejemplo 1).** Muchas partículas distintas producen la *misma*
  máscara binaria. Se guarda un diccionario `{máscara → (costo, exactitud)}` y solo se
  reentrena ante un subconjunto nuevo. El contador `n_real_evals` separa las evaluaciones
  reales de los aciertos de caché.
- **Partición de CV fija.** El `StratifiedKFold` se crea una sola vez con semilla fija, de
  modo que la aptitud es **determinista**: dos partículas iguales dan el mismo valor y las
  comparaciones entre ellas no están contaminadas por el ruido del remuestreo.
- **Paralelismo en la CV (Ejemplo 2).** `cross_val_score(..., n_jobs=-1)` reparte los
  pliegues entre los núcleos disponibles.
- **Pesos fuera del objeto (Ejemplo 3).** El `MLP` guarda solo la *forma* de la red; los
  pesos se inyectan desde el vector de la partícula en cada `forward`. Así un único objeto
  evalúa a todo el enjambre sin duplicar memoria.
- **Forward pass vectorizado.** La red procesa todo el lote de datos con productos
  matriciales de NumPy (`A @ W + b`), sin bucles por muestra.

### 2.2 Diseñar bien el espacio de búsqueda

- **Escala logarítmica (Ejemplo 2).** `C` y `gamma` se buscan como `log₁₀`, porque varían en
  órdenes de magnitud. Buscar en escala lineal desperdiciaría casi todas las partículas en
  la zona de valores grandes.
- **Codificación continua + umbral (Ejemplo 1).** ABC es continuo y la selección de
  características es binaria; el umbral `x_j > 0.5` permite usar el operador de vecindad
  original sin modificarlo.
- **Aplanado de la red (Ejemplo 3).** Convertir la arquitectura en un vector transforma el
  entrenamiento en optimización continua estándar: el mismo código sirve para el caso
  binario y el multiclase cambiando solo la activación de salida y la aptitud.
- **Límites y recorte.** Todas las posiciones se recortan (`clip`) a su dominio válido
  después de cada movimiento, evitando configuraciones imposibles.

### 2.3 Equilibrar exploración y explotación

- **Inercia decreciente** `w: 0.9 → 0.4` (Ejemplos 2 y 3): el enjambre explora ampliamente
  al inicio y refina al final.
- **Limitación de velocidad** (`v_max` como fracción del rango): impide que las partículas
  "salten" fuera de las regiones prometedoras.
- **Abejas exploradoras (Ejemplo 1):** una fuente que no mejora tras `LIMIT` ciclos se
  reemplaza por una aleatoria; es el mecanismo anti-estancamiento de ABC.
- **Reinicio de partículas estancadas (Ejemplo 3):** cumple el mismo papel dentro de PSO.
- **Tasa de modificación `MR` (Ejemplo 1):** solo un 30 % de las dimensiones se perturba por
  vecindad, lo que produce cambios graduales en lugar de saltos que destruyen la solución.

### 2.4 Gastar menos iteraciones

- **Parada temprana.** Los tres criterios de finalización (máximo de iteraciones, tolerancia
  y estancamiento) se combinan; el cuaderno 3 además reporta `stop_reason_`, de modo que se
  sabe *por qué* terminó cada ejecución.
- **Selección voraz (Ejemplo 1).** Una solución solo se reemplaza si el vecino es mejor, así
  que el mejor valor nunca retrocede.
- **Memoria del enjambre.** `pbest`/`gbest` (PSO) y la mejor fuente memorizada (ABC)
  conservan el óptimo aunque la población se degrade.

### 2.5 Evaluación honesta de los resultados

Esta parte es tan importante como la búsqueda: sin ella, los números reportados serían
optimistas.

- **Conjunto de prueba aislado.** Se reserva un 25–30 % de los datos que el optimizador
  **nunca ve**. La búsqueda solo usa validación cruzada sobre el entrenamiento.
- **Presupuesto igualado.** ABC se compara contra búsqueda aleatoria, y PSO contra Grid y
  Random Search, con **el mismo número de evaluaciones de la aptitud**. Comparar por tiempo
  o por iteraciones sería engañoso.
- **Repetición con varias semillas.** Los algoritmos son estocásticos: se reportan media y
  desviación estándar sobre 5 ejecuciones, no una corrida afortunada.
- **Conclusiones calibradas.** Cuando PSO solo empata con Random Search, el cuaderno lo dice
  y explica por qué (espacio de 2 dimensiones con una meseta amplia de buenas soluciones).
- **Diagnóstico del enjambre (Ejemplo 3).** Se grafican convergencia, diversidad frente a
  inercia y exactitud en test del `gbest`, para verificar que el enjambre efectivamente
  transita de exploración a explotación y no está memorizando.

---

## 3. Resultados resumidos

| Ejemplo | Línea base | Resultado del enjambre | Lectura |
|---|---|---|---|
| 1 · ABC | 30 características, acc. CV ≈ 0.967 | ≈13 características, acc. CV ≈ 0.977 | Misma exactitud en test con **~57 % menos variables** |
| 2 · PSO | SVM por defecto | `C` y `gamma` ajustados, mejor exactitud CV y test | Empata con Random Search en 2D; la ventaja crece con la dimensión |
| 3 · PSO | Backpropagation (SGD + momentum) | ≈95–96 % de exactitud en test, 85 parámetros | Se entrena una MLP **sin calcular una sola derivada** |

---

## 4. Limitaciones conocidas

- El costo crece con la dimensión `D`: para redes de miles de parámetros, backpropagation
  sigue siendo muy superior porque aprovecha la estructura diferenciable del problema.
- Los enjambres tienen sus propios hiperparámetros (`w`, `c1`, `c2`, tamaño de población),
  que también habría que ajustar.
- Con 5 repeticiones las comparaciones son orientativas, no estadísticamente concluyentes.
- Los métodos *wrapper* (Ejemplo 1) sobreajustan a los pliegues de CV; de ahí la insistencia
  en el conjunto de prueba independiente.

---

## 5. Estructura del repositorio

```
.
├── README.md
├── informe/
│   └── Informe_Inteligencia_de_Enjambre.pdf
└── notebooks/
    ├── 1_Gruber_Feature_Selection_ABC.ipynb
    ├── 2_Gruber_PSO_Hyperparameter_Tuning.ipynb
    └── 3_Gruber_PSO_NN_sin_backpropagation.ipynb
```

## 6. Reproducibilidad

Todas las semillas están fijadas (`SEED = 42` y semillas 0–4 / 100–104 para los estudios de
robustez). Las cifras pueden variar ligeramente entre versiones de `scikit-learn`.

---

**Autor:** _(completar)_ · **Curso:** Inteligencia de Enjambre · **Año:** 2026
