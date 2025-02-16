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
- Используемая модель: 
  
  ```python
  MLP(
  (layer1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (flat): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=1024, out_features=1024, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (relu3): ReLU()
  (fc2): Linear(in_features=1024, out_features=47, bias=True)
  )
  ```
- Гиперпарметры: Стандартные
- Accuracy на тестовых данных: 0.8847340425531914

## 2. Установка и запуск сервиса

Установка сервиса:

```bash
git clone https://gitlab.skillbox.ru/pirozhenko_ilia/ml-advanced-HiEd.git
cd CV-NeuralNetworks
docker build -t numbers_nn .
```

Запуск сервиса:

```bash
docker run -p 8000:8000 numbers_nn
```
