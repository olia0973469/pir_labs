# Прогнозування відтоку клієнтів (Customer Churn): MLOps-проєкт

## Мета проєкту

Побудувати наскрізний MLOps-проєкт: перевірити якість вхідних даних, зафіксувати версію датасету, навчити моделі з відстеженням експериментів у MLflow, а потім розгорнути навчену модель як REST API (FastAPI) і запакувати сервіс у Docker-образ.

Цільова змінна `Churn` має два значення: клієнт залишився (`No`, 0) або пішов (`Yes`, 1).

## Джерело даних

Датасет Telco Customer Churn (IBM / Kaggle): інформація про клієнтів телекомунікаційної компанії, їхні послуги, контракти, платежі та факт відтоку.

- Файл: `data/raw/churn.csv`
- Цільова колонка: `Churn`
- Посилання: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> Колонка `TotalCharges` містить порожні або пробільні значення. Вони трактуються як пропущені й обробляються на етапі версіонування даних.

## Структура директорій

```text
customer-churn-mlops/
├── app/
│   ├── __init__.py
│   └── main.py                  # FastAPI inference-сервіс
├── configs/
│   └── data_schema.yaml
├── data/
│   ├── raw/churn.csv
│   └── processed/
│       ├── churn_v1.csv
│       └── churn_v1_metadata.json
├── examples/
│   └── predict_request.json     # приклад запиту до /predict
├── models/                      # latest_model.joblib (не в Git)
├── mlruns/                      # локальне сховище MLflow (не в Git)
├── notebooks/
├── reports/
│   ├── latest_metrics.json
│   ├── data_validation_report.json
│   └── model_selection.md
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── validate_data.py
│   ├── version_data.py
│   └── train.py
├── tests/
├── Dockerfile
├── .dockerignore
├── README.md
├── requirements.txt
└── .gitignore
```

## Встановлення залежностей

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Основні залежності: pandas, scikit-learn, joblib, pytest, pyyaml, mlflow, fastapi, uvicorn, pydantic.

## Підготовка даних і навчання

Усі команди виконуються з **кореня проєкту**.

```bash
python src/validate_data.py   # 1. валідація сирих даних
python src/version_data.py    # 2. створення версії датасету v1
python src/train.py           # 3. навчання моделі з логуванням у MLflow
```

- `validate_data.py` перевіряє `data/raw/churn.csv` за схемою `configs/data_schema.yaml`. Статус `passed` означає, що критичних помилок немає, `failed` означає, що знайдено помилки, і скрипт завершується з кодом `1`.
- `version_data.py` створює `data/processed/churn_v1.csv` та метадані з SHA-256 хешами.
- `train.py` навчає модель на версії даних `v1`, логує параметри, метрики, звіт і модель у MLflow та зберігає `models/latest_model.joblib`.

> Файл `models/latest_model.joblib` перезаписується кожним запуском навчання. API використовує саме його, тому перед запуском сервісу переконайтеся, що останньою навчено потрібну модель.

## MLflow: експерименти та Model Registry

Запуск UI (перший термінал, з кореня проєкту):

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Відкрийте http://127.0.0.1:5000.

- Experiment: `customer-churn-baseline`.
- Для кожного run логуються параметри (`model_type`, `data_version`, `test_size`, `random_state`), метрики (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`), artifact `latest_metrics.json` і модель.
- Усі runs зберігаються локально в папці `mlruns/` (у директорії, звідки запущено `mlflow ui` і `train.py`).
- Модель реєструється як **`CustomerChurnClassifier`**; версії видно у вкладці **Models**.
- Вибір найкращої моделі описано в `reports/model_selection.md`.

## Inference API (FastAPI)

Сервіс завантажує `models/latest_model.joblib` під час старту й надає такі endpoints:

| Endpoint | Метод | Призначення |
|---|---|---|
| `/health` | GET | стан сервісу та чи завантажена модель |
| `/predict` | POST | прогноз відтоку для одного клієнта |
| `/docs` | GET | Swagger UI (інтерактивна документація) |

### Запуск локально

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

### Приклад запиту

Файл `examples/predict_request.json`:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.0,
  "TotalCharges": 850.0
}
```

**macOS / Linux (bash):**

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

**Windows (PowerShell):** потрібно викликати `curl.exe`, бо `curl` у PowerShell є псевдонімом `Invoke-WebRequest`. Команду записують в один рядок:

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@examples/predict_request.json"
```

Альтернатива для PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body (Get-Content examples/predict_request.json -Raw)
```

### Приклад відповіді

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.0,
  "risk_label": "low"
}
```

> Підставте сюди реальну відповідь вашого сервісу. `risk_label` дорівнює `high`, якщо `churn_probability >= 0.5`, інакше `low`.

## Docker

Для запуску потрібен Docker Desktop (він має бути запущений, статус **Engine running**).

Збірка образу (з кореня проєкту):

```bash
docker build -t churn-api:latest .
```

Запуск контейнера:

```bash
docker run --rm -p 8000:8000 churn-api:latest
```

Перевірка (в іншому терміналі) виконується тими самими командами `curl`, що й для локального запуску: http://127.0.0.1:8000/docs.

Що входить в образ: код `app/` і `src/`, файл `models/latest_model.joblib` та залежності з `requirements.txt`. Файл `.dockerignore` виключає `.venv`, `.git`, `mlruns`, `notebooks`, `reports` та кеш. Контейнер слухає `0.0.0.0:8000`, порт відкривається через `-p 8000:8000`.

> Перед збіркою має існувати локальний файл `models/latest_model.joblib` (його створює `python src/train.py`). У Git він не комітиться, тому на новій машині спочатку потрібно навчити модель.

## Обмеження поточного inference pipeline

1. **Preprocessing не зберігається разом із моделлю.** У `/predict` кодування категорій (`pd.get_dummies`) дублює логіку з навчання. Якщо вона розійдеться з `prepare_features`, прогнози стануть хибними.
2. **Вирівнювання колонок.** Модель навчена на повному наборі one-hot колонок, а запит одного клієнта містить лише частину ознак. Тому в API колонки вирівнюються через `reindex(columns=model.feature_names_in_, fill_value=0)`, а відсутні ознаки заповнюються нулями.
3. **Неповний набір ознак у запиті.** API приймає 9 полів, а модель навчалась на всіх колонках датасету (`Contract`, `DeviceProtection` та інші), тому прогноз є наближеним.
4. **Невідомі категорії** ігноруються (усі відповідні колонки стають нулями) замість явної помилки.
5. **Розмір образу.** Через `mlflow` у `requirements.txt` збірка довга, а образ великий; для production потрібен окремий мінімальний набір залежностей для inference.

**Можливе покращення:** зберігати разом із моделлю повний `sklearn.pipeline.Pipeline` (`ColumnTransformer` з `OneHotEncoder(handle_unknown="ignore")` + модель). Тоді API передає сирі дані, а кодування відбувається всередині моделі, і розбіжність між навчанням та inference неможлива.

## Версія даних, яку використовує модель

| Параметр | Значення |
|---|---|
| Версія даних | `v1` |
| Файл | `data/processed/churn_v1.csv` |
| Метадані | `data/processed/churn_v1_metadata.json` |
| Звіт валідації | `reports/data_validation_report.json` |

## Результати експериментів

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | … | … | … | … | … |
| random_forest | … | … | … | … | … |

Основна метрика для вибору моделі: **…** (див. `reports/model_selection.md`). Модель, що обслуговується API: **…**.

## Висновок

Навчену модель перетворено на inference-сервіс: FastAPI надає endpoints `/health` і `/predict`, а Docker дозволяє запускати його в ізольованому середовищі з усіма залежностями. Основне обмеження полягає в тому, що preprocessing у API дублює логіку навчання й не зберігається разом із моделлю. Тому наступним кроком має стати збереження повного `Pipeline`, щоб навчання та inference використовували однакову обробку даних.