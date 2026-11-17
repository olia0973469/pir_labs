# Прогнозування відтоку клієнтів (Customer Churn): MLOps-проєкт

## Мета проєкту

Сформувати структуру MLOps-проєкту, перевірити якість вхідних даних, зафіксувати версію датасету, навчити базові моделі бінарної класифікації та відстежувати експерименти за допомогою MLflow з реєстрацією найкращої моделі в Model Registry.

Цільова змінна `Churn` має два значення: клієнт залишився (`No`, 0) або пішов (`Yes`, 1).

## Джерело даних

Датасет Telco Customer Churn (IBM / Kaggle): інформація про клієнтів телекомунікаційної компанії, їхні послуги, контракти, платежі та факт відтоку.

- Файл: `data/raw/churn.csv`
- Цільова колонка: `Churn`
- Посилання: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> Особливість датасету: колонка `TotalCharges` містить порожні або пробільні значення. Вони трактуються як пропущені і обробляються на етапі версіонування даних.

## Структура директорій

```text
customer-churn-mlops/
├── configs/
│   └── data_schema.yaml
├── data/
│   ├── raw/
│   │   └── churn.csv
│   └── processed/
│       ├── churn_v1.csv
│       └── churn_v1_metadata.json
├── models/
├── mlruns/                  # локальне сховище MLflow (не в Git)
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

Основні залежності: pandas, scikit-learn, joblib, pytest, pyyaml, mlflow.

## Запуск пайплайна

Усі команди виконуються з **кореня проєкту**.

```bash
python src/validate_data.py   # 1. валідація сирих даних
python src/version_data.py    # 2. створення версії датасету v1
python src/train.py           # 3. навчання моделі з логуванням у MLflow
```

- `validate_data.py` перевіряє `data/raw/churn.csv` за схемою `configs/data_schema.yaml`. Статус `passed` означає, що критичних помилок немає, `failed` означає, що знайдено помилки, і скрипт завершується з кодом `1`.
- `version_data.py` створює `data/processed/churn_v1.csv` та метадані з SHA-256 хешами.
- `train.py` навчає модель на версії даних `v1`.

## MLflow: відстеження експериментів і реєстр моделей

### Запуск MLflow UI

У **першому** терміналі (з кореня проєкту):

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Відкрийте в браузері: http://127.0.0.1:5000

### Запуск експериментів

У **другому** терміналі (з активованим `.venv`, також з кореня проєкту):

```bash
python src/train.py
```

Тип моделі обирається в кінці `src/train.py` у виклику `main(model_type=...)`:

- `logistic_regression`: `StandardScaler` + `LogisticRegression`;
- `random_forest`: `RandomForestClassifier`.

Щоб порівняти моделі, потрібно запустити навчання щонайменше двічі з різними значеннями `model_type`.

Кожен запуск у MLflow логує:

| Що | Деталі |
|---|---|
| Параметри | `model_type`, `data_version`, `test_size`, `random_state` |
| Метрики | `accuracy`, `precision`, `recall`, `f1`, `roc_auc` |
| Artifact | `reports/latest_metrics.json` |
| Модель | `mlflow.sklearn.log_model` (artifact `model`) |

Назва experiment: `customer-churn-baseline`.

### Де зберігаються MLflow runs

Усі runs, метрики, параметри й артефакти зберігаються **локально в папці `mlruns/`** у тій директорії, звідки запущено `mlflow ui` і `train.py`. Тому обидві команди потрібно запускати з кореня проєкту: тоді UI та скрипт працюють з однією папкою `mlruns/`.

Папка `mlruns/` створюється автоматично і додана в `.gitignore`. Файли `models/latest_model.joblib` також не комітяться.

### Як переглянути runs

1. Відкрийте http://127.0.0.1:5000.
2. Перейдіть до experiment `customer-churn-baseline`.
3. У списку runs можна порівняти параметри та метрики. Для порівняння виберіть кілька runs і натисніть **Compare**.
4. Відкрийте конкретний run, щоб побачити вкладку **Artifacts** (звіт `latest_metrics.json` і папка `model`).

### Як переглянути Model Registry

1. У верхньому меню MLflow UI відкрийте вкладку **Models** (або перейдіть за адресою http://127.0.0.1:5000/#/models).
2. Оберіть модель **`CustomerChurnClassifier`**.
3. Тут видно всі її версії (Version 1, 2, ...). Кожна версія посилається на run, з якого вона створена, тобто пов'язана з параметрами, метриками та версією даних.

Модель реєструється автоматично під час навчання через параметр `registered_model_name="CustomerChurnClassifier"`. Кожен запуск створює нову версію. Найкращу версію обирають за основною метрикою, обґрунтування наведено в `reports/model_selection.md`.

> Для `random_forest` у `log_model` передається `skops_trusted_types=["sklearn.tree._tree.Tree"]`: MLflow зберігає sklearn-моделі у форматі skops і вимагає явно довірити цей тип. Модель створена в цьому ж проєкті, тож їй можна довіряти.

## Версія даних, яку використовує модель

| Параметр | Значення |
|---|---|
| Версія даних | `v1` |
| Файл | `data/processed/churn_v1.csv` |
| Метадані | `data/processed/churn_v1_metadata.json` |
| Звіт валідації | `reports/data_validation_report.json` |

Версія даних логується в кожному MLflow run як параметр `data_version`, тож кожен результат можна однозначно пов'язати з датасетом, на якому навчена модель.

## Результати експериментів

Результати на тестовій вибірці (дані `v1`):

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | … | … | … | … | … |
| random_forest | … | … | … | … | … |

Основна метрика для вибору моделі: **…** (див. `reports/model_selection.md`).

## Висновок

Додано відстеження експериментів у MLflow: параметри, версія даних, метрики, звіт і модель логуються для кожного запуску, що дозволяє порівнювати моделі та відтворювати результати. Найкращу модель за основною метрикою зареєстровано в Model Registry як `CustomerChurnClassifier`. Обрана модель: … (обґрунтування в `reports/model_selection.md`).