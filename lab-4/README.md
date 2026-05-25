# Документация (Lab‑4)

## 1) Модули и функции

### Основные точки входа
- [lab-4/src/main.cpp](lab-4/src/main.cpp): старт приложения, вызывает `App::run`.
- [lab-4/src/app.cpp](lab-4/src/app.cpp): парсинг CLI, загрузка сцены, настройка камеры, запуск рендера, запись PPM.

### Трассировка путей и освещение
- [lab-4/src/core/path_tracer.cpp](lab-4/src/core/path_tracer.cpp): ядро трассировки.
  - `PathTracer::render(...)`:
    - генерирует первичные лучи на пиксель;
    - вычисляет пересечения (через Embree);
    - считает прямое освещение от источников;
    - выбирает диффузное или зеркальное событие по `k_d/k_s`;
    - применяет русскую рулетку;
    - накапливает радианс.
  - Внутренние функции:
    - `build_light_distribution(...)` — сбор распределения по площадям и яркости эмиттеров.
    - `sample_light(...)` — выбор источника, выбор точки на треугольнике.
    - `generate_camera_ray(...)` — луч камеры с джиттером.

### Семплинг
- [lab-4/src/core/sampling.cpp](lab-4/src/core/sampling.cpp): вероятностные выборки.
  - `build_onb(...)` — ортонормированный базис вокруг нормали.
  - `sample_cosine_hemisphere(...)` — косинусно‑взвешенный полушар.
  - `reflect_direction(...)` — идеальное зеркальное отражение.
  - `russian_roulette(...)` — терминатор траекторий с весовой поправкой.

### Геометрия и ускорение
- [lab-4/src/accel/embree_scene.cpp](lab-4/src/accel/embree_scene.cpp): создание сцены Embree и пересечения.
  - `EmbreeScene::build(...)` — загрузка треугольников в Embree.
  - `EmbreeScene::intersect(...)` — пересечение луча с геометрией, возвращает `HitInfo`.

### Загрузка сцены и материалы
- [lab-4/src/io/obj_loader.cpp](lab-4/src/io/obj_loader.cpp): парсинг OBJ + MTL.
  - `load_obj(...)` — вершины/треугольники, группы `light_*` → эмиттеры.
  - `load_mtl(...)` — чтение `Kd`/`Ks`.

### Вывод
- [lab-4/src/io/ppm_writer.cpp](lab-4/src/io/ppm_writer.cpp): запись PPM (P6).
  - `write_ppm(...)` — нормализация экспозиции, гамма‑коррекция, запись в файл.
- [lab-4/src/io/bilateral_filter.cpp](lab-4/src/io/bilateral_filter.cpp): билатеральная фильтрация в линейном пространстве.
  - `apply_bilateral_filter(...)` — использует глубину, нормаль и object id первого пересечения.

### Вспомогательные структуры
- [lab-4/include/core/vec3.h](lab-4/include/core/vec3.h): векторные операции, `dot`, `cross`, `normalize`.
- [lab-4/include/core/rng.h](lab-4/include/core/rng.h): генератор `Rng`.
- [lab-4/include/core/scene.h](lab-4/include/core/scene.h): `Scene`, `Triangle`, список эмиттеров.
- [lab-4/include/core/material.h](lab-4/include/core/material.h): `Material` с `k_d`, `k_s`, флагом эмиссии.
- [lab-4/include/core/camera.h](lab-4/include/core/camera.h): параметры камеры.
- [lab-4/include/io/image.h](lab-4/include/io/image.h): буфер изображения.

## 2) Логика и математика

### Камера
Луч через пиксель:
$$
\vec{d} = \text{normalize}(\vec{f} + \vec{r}\,x + \vec{u}\,y),
$$
где $(x, y)$ — координаты пикселя в NDC, масштабированные по FOV и aspect ratio.

### Диффузное отражение (Ламберт)
BRDF:
$$
 f_d = \frac{k_d}{\pi}
$$

Семплинг направления — косинусно‑взвешенное полушарие:
$$
\phi = 2\pi \xi_1,\quad \sin\theta = \sqrt{\xi_2},\quad \cos\theta = \sqrt{1 - \xi_2}
$$
PDF:
$$
 p(\omega) = \frac{\cos\theta}{\pi}
$$

### Зеркальное отражение
$$
\vec{d}_{\text{refl}} = \vec{d} - 2(\vec{d}\cdot\vec{n})\vec{n}
$$

### Прямое освещение от площадного источника
Для выбранной точки на источнике:
$$
L = T \cdot f_d \cdot L_e \cdot \frac{\cos\theta_s \cos\theta_l}{r^2} \cdot \frac{1}{p_A}
$$
где $T$ — текущий путь (throughput), $p_A$ — PDF по площади.

### MIS (балансная эвристика)
Смешивание PDF источника и PDF BRDF:
$$
 w = \frac{p_{light}}{p_{light} + p_{brdf}}
$$
Это снижает вклад редких «вспышек» (fireflies).

### Русская рулетка
Продолжение пути с вероятностью $q$, вес корректируется:
$$
T \leftarrow \frac{T}{q}
$$

## 3) Как использовать

### Сборка
```powershell
cmake -S lab-4 -B lab-4/build -DCMAKE_TOOLCHAIN_FILE=D:/ITMO.kal/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build lab-4/build --config Debug --target path_tracer
```

### Запуск
```powershell
.\lab-4\build\Debug\path_tracer.exe \
  --scene lab-4\scene_simple.obj \
  --out lab-4\output.ppm \
  --width 512 --height 512 \
  --spp 4 --max-depth 3 \
  --seed 1 \
  --cam-pos 0 1.2 3.5 \
  --cam-look 0 1.1 0 \
  --exposure 1.0 --gamma 2.2
```

### Формат результата
Выходной файл — PPM (P6). Его можно открыть в IrfanView, GIMP, XnView или конвертировать в PNG через ImageMagick.

## Сценарий использования (end‑to‑end)

1) Пользователь запускает программу с параметрами сцены и рендера.
  - Модуль: [lab-4/src/main.cpp](lab-4/src/main.cpp)
  - Метод: `main(int argc, char** argv)`
  - Действие: передает аргументы в `App::run`.

2) Парсинг аргументов, подготовка настроек и камеры.
  - Модуль: [lab-4/src/app.cpp](lab-4/src/app.cpp)
  - Методы: `App::run(int argc, char** argv)`, `print_usage()`, `parse_uint32(...)`, `parse_float(...)`
  - Действие: читает `--scene`, `--out`, `--width/height`, `--spp`, `--max-depth`, `--seed`, `--cam-pos`, `--cam-look`, `--exposure`, `--gamma`.
  - Результат: сформированы `RenderSettings`, `ToneMappingSettings`, параметры камеры.

3) Загрузка геометрии и материалов.
  - Модуль: [lab-4/src/io/obj_loader.cpp](lab-4/src/io/obj_loader.cpp)
  - Методы: `load_obj(...)`, `load_mtl(...)`, `parse_face_vertex(...)`, `to_index(...)`
  - Действие: парсит OBJ (вершины, треугольники, группы), читает MTL, назначает `Kd/Ks`.
  - Эмиттеры: группы `light_*` помечаются как источники света.
  - Результат: заполнена структура `Scene`.

4) Построение ускорителя и запуск трассировки.
   - Модуль: [lab-4/src/core/path_tracer.cpp](lab-4/src/core/path_tracer.cpp)
   - Метод: `PathTracer::render(...)`
   - Действие: 
     - `EmbreeScene::build(...)` загружает треугольники в Embree.
     - `generate_camera_ray(...)` формирует первичные лучи.
     - `sample_light(...)` вычисляет прямое освещение.
     - Выбор события: `sample_cosine_hemisphere(...)` или `reflect_direction(...)`.
     - `russian_roulette(...)` ограничивает длину пути.
   - Вспомогательные модули:
     - [lab-4/src/core/sampling.cpp](lab-4/src/core/sampling.cpp) — `build_onb(...)`, `sample_cosine_hemisphere(...)`, `reflect_direction(...)`, `russian_roulette(...)`.
     - [lab-4/src/accel/embree_scene.cpp](lab-4/src/accel/embree_scene.cpp) — `EmbreeScene::intersect(...)`.

5) Постобработка и запись результата.
  - Модуль: [lab-4/src/io/ppm_writer.cpp](lab-4/src/io/ppm_writer.cpp)
  - Модули: [lab-4/src/io/bilateral_filter.cpp](lab-4/src/io/bilateral_filter.cpp), [lab-4/src/io/ppm_writer.cpp](lab-4/src/io/ppm_writer.cpp)
  - Методы: `apply_bilateral_filter(...)`, `write_ppm(...)`
  - Действие: билатеральная фильтрация по depth/normal/object id, затем нормализация экспозиции и запись PPM (P6).
  - Результат: готовый файл изображения по пути `--out`.

### Пример полного запуска
```powershell
.\lab-4\build\Debug\path_tracer.exe \
  --scene lab-4\scene_simple.obj \
  --out lab-4\output_view.ppm \
  --width 512 --height 512 \
  --spp 4 --max-depth 3 \
  --seed 1 \
  --cam-pos 0 1.2 3.5 \
  --cam-look 0 1.1 0 \
  --exposure 1.0 --gamma 2.2
```
