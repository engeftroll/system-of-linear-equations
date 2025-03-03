### Решение СЛАУ
1. Код написан поздно ночью.
2. Код написан уставшим человеком.
3. Уставший человек убил на это целый день.

### Установка и запуск
1. Создайте виртуальное окружение: `python3 -m venv system-of-linear-equations`
2. Активируйте среду (Linux: `source ./path/to/venv`)

3. Запуск: `python3 app.py`


### Устройство под капотом
#### [app.py](./app.py)
Это основной код, который программа исполняет. Всё просто.

#### [methods_interface.py](./methods_interface.py)
Здесь описываются два основных класс: SolutionMemoryForLinearSystem (хранение решённых СЛАУ, для графика), AnyNumericalMethod (родительский класс для любого численного метода решения СЛАУ).

#### [methods/](./methods/ReadMe.md)
Реализация конкретных численных методов.
