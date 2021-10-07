# README #
language available: eng, rus

### About project ###
This project will explore how to creat AutoMl project on table data.

### Directory structure ###
```bash
+-- README.md.                   <- The top-level README for developers using this project.
+-- requirements.txt             <- The requirements file for reproducing the environment, e.g.generated with `pipreq /<path>` 
+-- setup.py.                    <- Make this project with `python setup.py build_ext --inplace`
+-- ml_data.                     <- Include the data for ml models.
+-- reports.                     <- Include automl reports.
+-- notebooks.                   <- Jupyter notebooks.
|   +-- Example.ipynb            <- Example AutoML with kaggle-data .
+-- config.yml                   <- loading... 
+-- edotools.                    <- The main project
|   +-- BaseAutoMlEstimator.py   <- Base AutoML class.
|   +-- AutoMlClassification.py  <- Class for classifictaion data.
|   +-- BaseAutoMlEstimator.py   <- Class for regression data.
|   +-- mytransformers.py        <- Transformers for Base class.
|   +-- mymetrics.py             <- loading...(Custom metrics)
```


### Dependencies ###
The code is compatible with Python 3 or higher. 
The following dependencies are needed to run the tracker:

* matplotlib>=3.1.1
* numpy>=1.17.2
* pandas>=0.25.1
* scikit_learn>=0.24.1


### Installation ###
1. Clone this repository  
`git clone https://github.com/Edvard88/Test_works/tree/master/edotools`

2. Recommend creating a virtual environment, because can be dependencies     
	2.1 Install virtual environment  
	`sudo apt install -y python3-venv`  
	2.2 Creat virtual environment    
	`python -m venv env`  
	2.3 Activate virtual environment  
	`source env/bin/activate`

3. Check all dependencies installed  
`pip install -r requirements.txt`

4. Build the code.    
`python setup.py build_ext --inplace`


### Quick Start ###
You may see the notebooks/Example.ipynb to quick start



### Contact ###
edvard88@inbox.ru

===============================================================================================================================

### О проекте ###
Проект по созданию AutoML на табличных данных 

### Структура директории###

```bash
+-- README.md.                   <- Основной README для разработчиков, использующие этот проект.
+-- requirements.txt             <- Фаил `requirements.txt` с используемые версиями библиотек для воспроизводимости программы. 
+-- setup.py.                     <- Соберите проект с помощью `python setup.py build_ext --inplace`
+-- ml_data.                     <- Включается в себя тестовые данные для проверки AutoML.
+-- reports.                     <- Вклчючает AutoML отчеты.
+-- notebooks.                   <- Jupyter notebooks.
|   +-- Example.ipynb            <- Пример AutoML, запущенного на kaggle данных .
+-- config.yml                   <- loading... (пока не создан)
+-- edotools.                    <- Главнй проект
|   +-- BaseAutoMlEstimator.py   <- Базовый класс AutoML.
|   +-- AutoMlClassification.py  <- Класс для задач классифкации.
|   +-- BaseAutoMlEstimator.py   <- Класс для задач регрессии.
|   +-- mytransformers.py        <- Вспомогательные трансформеры для Базового класса
|   +-- mymetrics.py             <- loading... (Кастомные метрики)
```



### Зависимости ###
Код копилируется на версии Python 3 и выше
Для запуска проекта необходимы следующие зависимости:

* matplotlib>=3.1.1
* numpy>=1.17.2
* pandas>=0.25.1
* scikit_learn>=0.24.1


### Установка ###
1. Скопируйте репозиторий 
`git clone https://github.com/Edvard88/Test_works/tree/master/edotools`

2. Рекомендуем создать виртуальное окружение, потому что могут быть зависимости
      
	2.1 Установка виртуального окружения    
	`sudo apt install -y python3-venv`  
	2.2 Создание виртуального окружения    
	`python -m venv env`.  
	2.3 Активируйте виртуальное окружение    
	`source env/bin/activate`
 

3. Проверьте установлены ли все необходимые зависимости.  
`pip install -r requirements.txt`


4. Запустите код  
`python setup.py build_ext --inplace`   

### Быстрый запуск программы   
Можно посмотреть notebook с примером notebooks/Example.ipynb для быстрого старта


### Контакты ###
edvard88@inbox.ru

