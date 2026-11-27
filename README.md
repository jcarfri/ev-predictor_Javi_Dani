# 🛡️ CHURN GUARD

**Predicción del abandono de clientes en telecomunicaciones**

Proyecto de aprendizaje automático (*Machine Learning*) orientado a estimar la probabilidad de que un cliente abandone el servicio y a analizar los factores asociados a dicho abandono (*churn*).

El proyecto abarca el análisis exploratorio de datos (EDA), la ingeniería de variables, el entrenamiento y la evaluación de modelos de clasificación, y su puesta en funcionamiento mediante una aplicación web desarrollada con Streamlit y preparada para ejecutarse con Docker.

> 📄 **Informe técnico:** desarrolla el análisis, los gráficos, la evaluación del modelo y las limitaciones del proyecto.

---

## 🎯 Objetivos

- Identificar los perfiles de cliente con mayor tasa de abandono.
- Analizar la relación del abandono con la antigüedad, el contrato, los servicios y la facturación.
- Comparar distintos algoritmos de clasificación.
- Alcanzar un recall mínimo del 75 % en validación.
- Mantener el sobreajuste por debajo del 5 %.
- Ofrecer las predicciones mediante una aplicación interactiva.
- Recoger resultados reales para futuros reentrenamientos.

## ❓ Preguntas de investigación

1. ¿Qué proporción de clientes abandona el servicio?
2. ¿Influye la antigüedad en la tasa de abandono?
3. ¿Qué tipos de contrato presentan más bajas?
4. ¿Se asocian determinados servicios o formas de pago con el abandono?
5. ¿Con qué fiabilidad se pueden anticipar las bajas?

---

## 📊 Datos utilizados

- **Conjunto de datos:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), ejemplo ficticio publicado por IBM.
- **Archivo de trabajo:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Indicador | Valor |
|---|---|
| Clientes | 7.043 |
| Columnas originales | 21 |
| Variables predictoras originales | 19 |
| Clientes que abandonan | 1.869 (26,5 %) |
| Clientes que permanecen | 5.174 (73,5 %) |
| Filas duplicadas | 0 |
| Valores vacíos en `TotalCharges` | 11 |

**Unidad de análisis:** cada fila representa un cliente.

### Variables disponibles

| Categoría | Variables |
|---|---|
| Perfil | Género, mayor de 65 años, pareja y personas a cargo |
| Servicios | Teléfono, líneas, internet, seguridad, copia de seguridad, protección, soporte y *streaming* |
| Contrato | Tipo de contrato, factura electrónica y forma de pago |
| Economía y antigüedad | Cuota mensual, cargos totales y meses como cliente |
| Objetivo | `Churn` (abandona o permanece) |

---

## 🔎 Metodología

1. **Inspección:** revisión de la estructura, los tipos de datos, los valores vacíos y los duplicados.
2. **Análisis exploratorio:** distribuciones, correlaciones y tasas de abandono por categoría.
3. **Limpieza:** conversión de `TotalCharges` a formato numérico y asignación de 0 a los valores vacíos, correspondientes a clientes sin antigüedad.
4. **Ingeniería de variables:** creación de nuevos indicadores.
5. **Preprocesamiento:** escalado de variables numéricas y codificación *One-Hot* de las categóricas.
6. **División de datos:** 80 % para entrenamiento y validación cruzada y 20 % para test, de forma estratificada.
7. **Modelado:** comparación y optimización de algoritmos.
8. **Evaluación:** métricas, sobreajuste, importancia de variables y análisis de errores.
9. **Puesta en producción:** aplicación Streamlit y contenedor Docker.

### Ingeniería de variables

| Variable | Descripción |
|---|---|
| `NumServices` | Número de servicios telefónicos y adicionales contratados |
| `AvgMonthlyCharge` | Cargo medio mensual |
| `ChargeDiff` | Diferencia entre la cuota actual y el cargo medio |
| `FiberNoSupport` | Cliente con fibra y sin soporte técnico |
| `NewMonthly` | Contrato mensual y antigüedad máxima de 12 meses |
| `AutoPay` | Pago automático |
| `LogTenure` | Antigüedad en escala logarítmica |

---

## 📈 Principales resultados del análisis

Los resultados describen exclusivamente el conjunto analizado y **no demuestran causalidad**.

### 1. Antigüedad

| Antigüedad | Tasa de abandono |
|---|---|
| 0 a 12 meses | 47,4 % |
| 13 a 24 meses | 28,7 % |
| 25 a 48 meses | 20,4 % |
| 49 a 72 meses | 9,5 % |

### 2. Tipo de contrato

| Contrato | Tasa de abandono |
|---|---|
| Mensual | 42,7 % |
| Un año | 11,3 % |
| Dos años | 2,8 % |

### 3. Servicios y forma de pago

| Característica | Tasa de abandono |
|---|---|
| Fibra óptica | 41,9 % |
| DSL | 19,0 % |
| Sin soporte técnico (con internet) | 41,6 % |
| Con soporte técnico | 15,2 % |
| Cheque electrónico | 45,3 % |
| Tarjeta de crédito automática | 15,2 % |

Los clientes con contrato mensual y un máximo de un año de antigüedad representan el 28 % del total y concentran el 55 % de las bajas.

---

## 🤖 Modelado y evaluación

Los modelos se optimizaron con `RandomizedSearchCV` y validación cruzada estratificada de 5 particiones.

| Modelo | ROC-AUC en validación | Diferencia de AUC con entrenamiento (puntos) |
|---|---|---|
| Regresión logística | 0,8495 | 0,5 |
| Random Forest | 0,8479 | 2,9 |
| **Gradient Boosting** | **0,8500** | **2,1** |

Se seleccionó **Gradient Boosting** por obtener el mayor ROC-AUC, si bien su rendimiento es prácticamente equivalente al de la regresión logística. El umbral de decisión se fijó en **0,307**.

### Resultados en test

| Métrica | Valor | Interpretación |
|---|---|---|
| Recall | 76,5 % | Proporción de bajas detectadas |
| Precisión | 54,7 % | Proporción de avisos correctos |
| Accuracy | 76,9 % | Proporción total de aciertos |
| F1 | 0,638 | Equilibrio entre precisión y recall |
| ROC-AUC | 0,846 | Capacidad para ordenar clientes por riesgo |

### Matriz de confusión

| | Predicción: permanece | Predicción: abandona |
|---|---|---|
| **Real: permanece** | 798 | 237 |
| **Real: abandona** | 88 | 286 |

- ✅ **Sobreajuste:** la diferencia máxima entre las métricas de clasificación de entrenamiento y validación es de **2,8 puntos porcentuales**.
- 🔑 **Variables más influyentes:** tipo de contrato y antigüedad, según la importancia por permutación.

---

## 💻 Aplicación interactiva con Streamlit

| Sección | Funcionalidades |
|---|---|
| 🎯 Predicción | Probabilidad, nivel de riesgo, señales de alerta y simulador de retención |
| 📂 Predicción por lotes | Carga de un CSV y descarga de resultados |
| 📊 Rendimiento | Métricas y gráficas del modelo |
| 💬 Feedback | Registro de resultados reales, métricas de seguimiento y exportación de datos |

Las predicciones que el usuario decide guardar se almacenan en una base de datos SQLite.

---

## 🧰 Herramientas y librerías

### Programas y plataformas

| Herramienta | Uso |
|---|---|
| Python | Lenguaje principal del proyecto |
| Google Colab / Jupyter Notebook | Análisis exploratorio y experimentación |
| Visual Studio Code | Desarrollo del código |
| Git y GitHub | Control de versiones y colaboración |
| Docker Desktop | Contenerización de la aplicación |
| Kaggle | Fuente del conjunto de datos |

### Librerías

| Librería | Versión | Uso |
|---|---|---|
| Pandas | 3.1.1 | Manipulación de datos |
| NumPy | 2.5.5 | Cálculo numérico |
| Scikit-learn | 1.9.1 | Preprocesamiento, modelos y evaluación |
| Matplotlib | 3.11.2 | Visualización e informes |
| Seaborn | 0.13.2 | Gráficos del análisis exploratorio |
| Streamlit | 1.67.0 | Aplicación web |
| Joblib | 1.6.0 | Almacenamiento y carga del modelo |
| SQLite (`sqlite3`) | Incluida en Python | Base de datos local |

---

## 📁 Estructura del proyecto

```text
clasificacion_Javi_Dani/
├── .streamlit/
│   └── config.toml
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl
├── NoteBooks/
│   └── churn (1).ipynb
├── reports/                  # Métricas y gráficas
├── src/
│   ├── app.py                # Aplicación
│   ├── features.py           # Ingeniería de variables
│   └── train.py              # Entrenamiento y evaluación
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación y ejecución

**Requisitos:** Git, Python 3.14 y, opcionalmente, Docker Desktop.

### Ejecución local (Windows PowerShell)

```powershell
git clone https://github.com/Bootcamp-DA-P3/clasificacion_Javi_Dani.git
cd clasificacion_Javi_Dani
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run src/app.py
```

### Reentrenamiento del modelo

```powershell
python src/train.py
```

### Ejecución con Docker

```powershell
docker build -t churn-guard .
docker run --rm -p 8501:8501 churn-guard
```

La aplicación estará disponible en `http://localhost:8501`.

---

## ⚠️ Limitaciones

- Los datos son ficticios y corresponden a un único periodo.
- El modelo no indica cuándo se producirá una baja.
- El conjunto de test se consultó en iteraciones anteriores, por lo que las métricas pueden ser algo optimistas.
- Las asociaciones observadas y las simulaciones no demuestran causalidad.
- La base de datos no persiste al eliminar el contenedor.
- El notebook está pendiente de alinearse con el entrenamiento actual.

## 🔮 Líneas de trabajo futuras

- Pruebas unitarias.
- Persistencia de datos en Docker.
- Despliegue en la nube.
- Reentrenamiento automático y comparación de modelos.
- Monitorización de cambios en los datos.

---

## ✅ Conclusiones

El abandono se concentra especialmente en clientes recientes y con contrato mensual. El modelo detecta aproximadamente tres de cada cuatro bajas en el conjunto de test, a costa de un volumen relevante de falsas alarmas. Resulta adecuado como apoyo para priorizar acciones de retención, cuya eficacia debe comprobarse antes de aplicarlas de forma generalizada.

## 👥 Autoría

Proyecto realizado en el contexto del Bootcamp de Data Analytics.

- Javier Carnero Frías
- Daniel Gámez Soto