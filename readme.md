# CityLearn-RL-Project 🇪🇸🏙️⚡️

**Gestión energética inteligente de edificios con Deep Reinforcement Learning**

Proyecto personal para demostrar (y aprender) cómo diseñar, entrenar y evaluar agentes de RL capaces de reducir consumo, coste y picos de demanda manteniendo el confort térmico en un distrito virtual de edificios (🛠️ **CityLearn**).

---

## ✨ Objetivos

1.  **Minimizar energía total** consumida por edificio.
2.  **Minimizar la factura** eléctrica aplicando tarifas horarias dinámicas.
3.  **Peak-shaving** → reducir picos máximos de potencia.
4.  **Balancear confort vs. consumo** penalizando salidas del rango 21–24 °C.
5.  Explorar **baterías y fotovoltaica** (carga/descarga óptima).
6.  Comparar **múltiples agentes** (DQN, PPO, SAC, reglas, random).
7.  Visualizar resultados en un **dashboard interactivo** (Streamlit).

*Cada objetivo será un experimento reproducible con métrica(s) clara(s).*

---

## 📁 Estructura

```
CityLearn-RL-Project/
│
├── README.md               ← este archivo
├── requirements.txt        ← dependencias Python
├── .gitignore
│
├── data/                   ← escenarios CityLearn (climas, configs…)
│
├── notebooks/              ← exploración, demo y análisis visual
│   ├── 00_exploracion_datos.ipynb
│   ├── 01_baselines_reglas.ipynb
│   └── 02_experimentos_RL.ipynb
│
├── src/
│   ├── main.py             ← lanza entrenamiento según config YAML
│   ├── train.py            ← loop de entrenamiento RL
│   ├── evaluate.py         ← evalúa políticas y guarda métricas
│   ├── agents/             ← DQN, PPO, SAC, Random, ReglaFija…
│   ├── env/                ← wrappers CityLearn + funciones reward
│   ├── utils/              ← métricas, visualización, parsing config
│   └── monitoring/         ← panel Streamlit (en vivo / post-hoc)
│
├── configs/                ← *.yaml con parámetros de experimento
│
├── results/                ← logs, checkpoints, gráficos, CSV métricas
└── tests/                  ← tests unitarios (PyTest)
```

---

## 🚀 Instalación rápida

```bash
# 1. Clona el repo
git clone [https://github.com/tu-usuario/CityLearn-RL-Project.git](https://github.com/tu-usuario/CityLearn-RL-Project.git)
cd CityLearn-RL-Project

# 2. Crea entorno virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Prueba un entorno CityLearn
python - <<'PY'
from citylearn import CityLearn
env = CityLearn(data_path='data/Climate_Zone_1/')
print('Estado inicial OK:', env.reset()[:5])
PY
```

### 🏃‍♂️ Cómo lanzar un experimento

```bash
python src/main.py --config configs/exp_min_energia.yaml
```
`main.py` lee el YAML: selecciona agente, reward, horizonte temporal y semilla; guarda todo en `results/`.

### 📊 Monitor en vivo

```bash
streamlit run src/monitoring/dashboard.py
```
Verás:
* Curva de reward por episodio
* Evolución de temperatura interior/exterior
* Consumo vs. baseline
* Picos de potencia, coste acumulado, CO₂, etc.

---

## 📈 Métricas clave

| Métrica                | Descripción                             |
| :--------------------- | :-------------------------------------- |
| **`energy_kWh`** | Energía total consumida                 |
| **`cost_eur`** | Factura (€) con tarifa TOU              |
| **`peak_kw`** | Potencia máxima horaria                 |
| **`comfort_violations_h`** | Horas fuera de 21–24 °C                 |
| **`co2_kg`** | Emisiones (si el escenario lo permite)  |

*Todas se guardan en `results/metrics/experiment_id.csv`.*

---

## 📓 Notebooks recomendados
* `00_exploracion_datos` – qué variables hay en CityLearn y cómo se comportan.
* `01_baselines_reglas` – regla de control sencilla y agente aleatorio.
* `02_experimentos_RL` – carga checkpoints, compara agentes y genera plots listos para papers.

---

## 🛠️ Añadir un nuevo objetivo / reward
1.  Implementa tu función en `src/env/reward_functions.py`.
2.  Añade la entrada `reward_type:` en un YAML de `configs/`.
3.  Lanza `main.py` con la nueva config.

*La métrica aparecerá automáticamente en el dashboard y en los CSV.*

---

## 🧪 Tests

```bash
pytest -q
```
Nos aseguramos de que:
* El wrapper del entorno devuelva estados con el `shape` correcto.
* Las métricas calculen bien ahorro y confort.
* Los agentes produzcan acciones dentro de los límites.

---

## 📜 Créditos y licencias
* **CityLearn** © Intelligent Environments Lab - MIT License
* **Este proyecto**: Apache 2.0 (ver LICENSE).

---

## 🤝 Contacto
José Arbeláez (UAM) – jose.ancizar.667@gmail.com

¡Feliz optimización energética! ⚡️🏢