# Deconvolution
Требoвалось написать программу, реализующую метод обращения свёртки (деконволюции) изображений через минимизацию регуляризирующего функционала.

В качестве регуляризатора использовася функционал обобщенной полной вариации.

Полный вид минимизируемого функционала: \
$F(z) = ||Az - u||^2 + \alpha \cdot TV(z) + \beta \cdot TV2(z) $

Программа поддерживает запуск из командной строки с форматом команд: \
`%programname% (input_image) (kernel) (output_image) (noise_level)`

Аргументы:
Arg | Description
:---|:---
input_image	| 	Имя файла — входное размытое и зашумлённое изображение
kernel	| 	Имя файла — ядро размытия, изображение
output_image	| 	Имя файла — выходное изображение
noise_level	| 	Стандартное отклонение гауссовского шума на входном изображении (в диапазоне $[0, 20]$)
