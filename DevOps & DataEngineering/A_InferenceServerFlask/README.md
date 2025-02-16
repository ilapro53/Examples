
## Рабочая версия

## Ход работы

#### 1. Подготовка модели:
  - Ход работы: [./local/prepare_model.ipynb](local/prepare_model.ipynb)
  - Выходные файлы:
    - [./local/server_config.json](local/server_config.json)
    - [./local/my_model.keras](local/my_model.keras)
    - `/tmp/tfsrver/models/my_model/*`
    - [./local/test_images/*](local/test_images)

#### 2. Подготовка сервера Tensorflow Serving локально (wsl):
  - Ход работы: [./local/prepare_server.ipynb](local/prepare_server.ipynb)
  - Входные файлы:
    - [./local/server_config.json](local/server_config.json)
  - Выходные файлы:
    - [./local/install_server.sh](local/install_server.sh)
    - [./local/requirements.txt](local/requirements.txt)
    - [./local/start_server.sh](local/start_server.sh)
    - [./local/server.log](local/server.log)
   
#### 3. Подготовка клиента:
  - Ход работы: [./local/build_client.ipynb](local/build_client.ipynb)
  - Входные файлы:
    - [./local/test_images/*](local/test_images)
    - `/tmp/tfsrver/models/my_model/*`

> :warning: При выполнении запроса к серверу возникает ошибка (последняя ячейка [build_client.ipynb](local/build_client.ipynb))
> ```
> Could not find variable dense/bias. This could mean that the variable has been deleted. In TF1, it can also mean the variable is uninitialized. Debug info: container=localhost, status error message=Resource localhost/dense/bias/N10tensorflow3VarE does not exist.\n\t [[{{function_node __inference_serving_default_577}}{{node sequential_1/dense_1/Add/ReadVariableOp}}]]
> ```

#### 4. Подготовка сервера Tensorflow Serving через Docker:
  - Ход работы: [./docker/prepare_server.ipynb](docker/prepare_server.ipynb)

> :warning: При выполнения запроса к серверу возникает та же сошибка, что и при запуске сервера локльно

#### 5. Написание собственного сервера:
  - Ход работы:
    - [./custom_server/build_server.ipynb](custom_server/build_server.ipynb)
    - [./custom_server/build_client.ipynb](custom_server/build_client.ipynb)
  - Входные файлы:
    - [./custom_server/test_images/*](custom_server/test_images)
    - [./custom_server/server_models/my_model/*](custom_server/server_models/my_model)
  - Выходные файлы:
    - [./custom_server/client.py](custom_server/client.py)
    - [./custom_server/server.py](custom_server/server.py)

> ✔️ Клиент и сервер взаимодействуют без ошибок (последняя ячейка [build_client.ipynb](custom_server/build_client.ipynb))

>  ℹ️  Однако точность модели не очень хорошая (возможно это происходит из-за неправильной трансформации изображений при подготовке тестовых данных или при перед инференсом на сервере)
