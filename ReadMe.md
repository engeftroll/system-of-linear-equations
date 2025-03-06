# Решение СЛАУ
1. Код написан поздно ночью.
2. Код написан уставшим человеком.
3. Уставший человек убил на это целый день.

## Установка и запуск
0. Открыть терминал (Linux: `ctrl + alt + t`; Windows: `win+R`, ввести `cmd`). Перейти в директорию с кодом, используя `cd path/to/repository`.
1. Создайте виртуальное окружение в : `python3 -m venv system-of-linear-equations`
2. Активируйте среду (Linux: `source ./path/to/venv`; Windows `читай отдельный блок ниже`)

3. Запуск: `python3 app.py`

## Активация среды на Windows
0. В терминале Windows (см 0 пункт)
1. `cd venv\Scripts`
2. `activate.bat`
3. `cd ..\..`


# Устройство под капотом
## [app.py](./app.py)
Это основной код, который программа исполняет. Всё просто.

## [methods_interface.py](./methods_interface.py)
Здесь описываются два основных класс: SolutionMemoryForLinearSystem (хранение решённых СЛАУ, для графика), AnyNumericalMethod (родительский класс для любого численного метода решения СЛАУ).

## [methods/](./methods/ReadMe.md)
Реализация конкретных численных методов.

