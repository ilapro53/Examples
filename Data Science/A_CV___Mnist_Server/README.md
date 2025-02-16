# Распознавание рукописных символов EMNIST

## 1. Описание решения

- Задача: обучить модель, которая классифицирует рукописные символы
- Входные данные:
    - Одноканальные (монохромные) изображения размера 28 на 28 px.
    - Писели кодируются цветом от 0 до 255.
    - Модель получает данные в формате numpy.ndarray размера (28, 28)
    - Пары «лейбл — ASCII-код символа» хранятся в файле `emnist-balanced-mapping.txt`
- Выходные данные:
    - Строка, содержащая 1 символ из списка: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`, `a`, `b`, `d`, `e`, `f`, `g`, `h`, `n`, `q`, `r`, `t`
- Используемая модель: `RandomForestClassifier`
- Гиперпарметры:
```
{
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': None,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'random_state': 42,
    'splitter': 'best'
}
```
- Accuracy на тестовых данных: 0.8115425531914894
- Средний cross val score на 5 фолдах: 0.7421276595744681


## 2. Установка и запуск сервиса

Установка сервиса:

```bash
git clone https://gitlab.skillbox.ru/pirozhenko_ilia/ml-advanced-HiEd.git
cd CV
docker build -t numbers .
```

Запуск сервиса:
```bash
docker run -p 8000:8000 numbers
```