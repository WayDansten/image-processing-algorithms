Цель: Снизить шумы на синтезированном изображении, сохранив его физическую корректность.

Входные данные: Изображение со следующей информацией в точке первого пересечения луча со сценой:
- Прямая яркость
- Вторичная яркость
- Глубина
- Индекс объекта сцены
- Нормаль к поверхности

Задание: Реализовать метод билатеральной фильтрации синтезированного изображения с сохранением границ объектов. Показать корректность результатов фильтрации.

Детали и уточнения:
- Формула билатеральной фильтрации:
$$g(p)=\frac{1}{W_p}\sum\limits_{q\in S}f(q)G_S(||p-q||)G_r(||p-q||)$$
где:
	- $p$ - точка изображения
	- $f(q)$ - интенсивность точки изображения
	- $S$ - окно фильтра
	- $G_S$ - ядро по изображению
	- $G_r$ - пространственное ядро
	- $W_p$ - весовой коэффициент.
- Для фильтрации изображения в качестве веса по поверхности можно использовать как простейшее арифметическое среднее (проще), так и медианный фильтр (сложнее).
- При условии, что $\sum\limits_{q\in S}G_S(||p-q||)=1$ надо $W_p=\sum\limits_{q\in S}G_r(||p-q||)$.
- Для медианного фильтра надо нормировать все изображение после фильтрации так, чтобы для каждого объекта $O$:
$$\sum\limits_{p\in O}g(p)=\sum\limits_{p\in O}f(q)$$

---

## Подробное объяснение реализации и математики

### 1) Какие данные используются
Фильтрация выполняется над итоговой яркостью в линейном пространстве, но с учетом геометрии первого пересечения:
- **Глубина** $d(p)$: длина луча до первой точки пересечения.
- **Нормаль** $n(p)$: нормаль поверхности в этой точке.
- **Индекс объекта** $id(p)$: id треугольника (primID) от Embree.

Эти данные собираются в рендере и сохраняются в буферах `depth`, `normals`, `object_ids`.

### 2) Математическая постановка
Билатеральная фильтрация объединяет сглаживание по изображению и по признакам поверхности:

$$
g(p)=\frac{1}{W_p}\sum\limits_{q\in S} f(q)\,G_S(\|p-q\|)\,G_r(p,q)
$$

Где вес по поверхности задается как произведение:

$$
G_r(p,q)=G_{depth}(d_p-d_q)\cdot G_{normal}(n_p,n_q)\cdot \text{mask}(id_p=id_q)
$$

Глубина:

$$
G_{depth}=\exp\left(-\frac{(d_p-d_q)^2}{2\sigma_d^2}\right)
$$

Нормали:

$$
G_{normal}=\exp\left(-\frac{(1-\langle n_p,n_q\rangle)^2}{2\sigma_n^2}\right)
$$

Маска по объекту жестко запрещает смешивание разных объектов.

### 3) Программная реализация (ключевые шаги)

#### 3.1) Сбор G-buffer (первое пересечение)
Во время трассировки, только на глубине `depth == 0`, аккумулируются глубина и нормаль. `object_id` берется из первого сэмпла:

```cpp
if (depth == 0) {
	primary_hit = true;
	depth_sum += hit.t;
	normal_sum += hit.normal;
	if (!object_id_set) {
		object_id = hit.object_id;
		object_id_set = true;
	}
}
```

После завершения всех сэмплов значения усредняются и записываются в буфер:

```cpp
out_image.object_id_at(x, y) = object_id;
if (primary_hit_count > 0) {
	out_image.depth_at(x, y) = depth_sum / primary_hit_count;
	Vec3 avg_normal = normal_sum / primary_hit_count;
	if (length(avg_normal) > 0.0f) {
		avg_normal = normalize(avg_normal);
	}
	out_image.normal_at(x, y) = avg_normal;
}
```

#### 3.2) Пространственный вес $G_S$
Сначала предвычисляются веса для окна $S$ радиуса `radius`:

```cpp
const float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
const float weight = gaussian_weight(distance, sigma_spatial);
```

#### 3.3) Вес по нормалям
Нормали сравниваются через скалярное произведение, затем применяется гауссово ядро:

```cpp
const float ndot = std::clamp(dot(na, nb), -1.0f, 1.0f);
const float diff = 1.0f - ndot;
return gaussian_weight(diff, sigma_normal);
```

#### 3.4) Основной цикл фильтрации
Для каждого соседа $q$ внутри окна:
1) проверяется совпадение объекта;
2) вычисляются $G_{depth}$ и $G_{normal}$;
3) итоговый вес = $G_S \cdot G_{depth} \cdot G_{normal}$.

```cpp
if (neighbor_object != base_object) {
	continue;
}

const float w_depth = gaussian_weight(base_depth - depth_q, sigma_depth);
const float w_normal = normal_weight(base_normal, normal_q, sigma_normal);
const float weight = offset.weight * w_depth * w_normal;

if (weight > 0.0f) {
	sum += input.at(ux, uy) * weight;
	weight_sum += weight;
}
```

Если сумма весов равна нулю (например, пиксель на резкой границе), берется исходный цвет:

```cpp
output.at(x, y) = (weight_sum > 0.0f) ? (sum / weight_sum) : base_color;
```

### 4) Где фильтр встраивается
Фильтр применяется после рендера и до тонмаппинга, чтобы работать в линейном пространстве:

```cpp
Image filtered{};
BilateralFilterSettings filter_settings{};
apply_bilateral_filter(image, filtered, filter_settings, error_message);
write_ppm(output_path, filtered, tone_map, error_message);
```

### 5) Параметры по умолчанию
Используются значения:
- `radius = 5`
- $\sigma_s = 2.0$
- $\sigma_d = 0.1$
- $\sigma_n = 0.2$

Их можно вынести в CLI, если понадобится подбор параметров для отчетной части.