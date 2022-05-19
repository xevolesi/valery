# Задача

Необходимо построить модель мультилейбл-классификации на 17 возможных классов для прогнозирования типа местности по спутниковым снимкам.

# Данные

Данные можно достать по [ссылке](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data).

# Описание репозитория

 - В директории `dataset` лежит код торчевого датасет-класса;
 - В директории `models` лежит код модели;
 - В директории `test_data` лежит небольшая порция данных для тестов;
 - В директории `tests` лежат сами тесты;
 - В директории `utils` лежат разного рода утилитки:
    1. `utils/augmentations` содержит определения методов аугментаций для обучения и валидации;
    2. `utils/evaluation` содержит определение функционала качества (в данном случае это `F2-score`, я взял именно ее, потому что в соревновании использовалась именно эта метрика);
    3. `utils/general` лежат основные "системные" утилиты;
    4. `utils/options` содержит DTO для настроек проекта;
    5. `utils/splitting` содержит код для сплита на выборки, который я честно одолжил у вас;
    6. `utils/training` содержит утилиты для обучения модели.
- Модуль `train_val_test_split.py` запускает процесс разделения данных на тренировочную, валидационную и тестовую выборки;
- Модуль `train.py` запускает процесс тренировки;
- Модуль `predict.py` содержит код для инференса;
- Модуль `convert_to_jit.py` содержит код для конвертирования модели в TorchScript-модель.

# Ёкарный бабай! Поехали!!!

### Общие настройки

1. Склонируйте репозиторий и перейдите в директорию с ним;
2. Создайте виртуальное окружение:

        python3 -m venv /path/to/new/virtual/environment
        source /path/to/new/virtual/environment/bin/activate

3. Установите зависимости:

        pip install -r requirements.txt

### Для обучения

1. Скачайте данные;
2. Выполните команду:

        python train_val_test_split.py \ 
        --csv_path %путь до исходного .csv файла train_v2.csv% \
        --splitting_folder_path %путь до папки, в которую будут сохранены .csv файлы в результате сплита% \
        --path_to_image_folder %путь до папки с картинками, trian-jpg% \
        --train_fraction %Процент данных, которые будут использованы для тренировочной выборки. Например, 0.8 % 
        
    и получите файлики `train_df.csv`, `val_df.csv`, `test_df.csv`.
3. Заполните файл `config.yml` под свои нужды. В большинстве случаев вам захочется поменять что-нибудь в секциях `path` и `training`;
4. Установите `ClearML`:
   1. [Ссылка на ClearML](https://clear.ml/);
   2. [ClearML get started](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/).
5. Запустите скрипт `train.py`:

        python train.py

### Для инференса

Выполните:

        python predict.py \
        --model_path %Путь до файла модели. Если используете TorchScript, то указывайте путь до скриптованной модели, если не используете, то указывайте путь до обычной модели% \
        --image_path %Путь до картинки% 

# ClearML-ссылки

Ссылки на эксперименты: 

1. [Эксперимент 6](https://app.community.clear.ml/projects/c73a61ffe1f84e0fb6597644adf2b98d/experiments/7ff77d25b3634ab98520455faced79eb/output/execution)
2. [Эксперимент 5](https://app.community.clear.ml/projects/c73a61ffe1f84e0fb6597644adf2b98d/experiments/ac6fde48c0b24a9fb77a47e8be7cc95f/output/execution);
3. [Эксперимент 4](https://app.community.clear.ml/projects/c73a61ffe1f84e0fb6597644adf2b98d/experiments/1563d56b13a0444b8ad4cb06eb77b09d/output/execution).
