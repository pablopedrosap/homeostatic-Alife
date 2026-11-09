# Go2 body budget en GPU (Kaggle / Colab): instrucciones exactas

Notebook: [`go2_body_budget_gpu.ipynb`](go2_body_budget_gpu.ipynb) · Código: [`go2_body_budget.py`](go2_body_budget.py)

Miles de robots Unitree Go2 entrenando en paralelo en una GPU: la misma física que los experimentos de
CPU (motores que se calientan, pierden fuerza y se cortan; el robot siente temperatura y esfuerzo), en
JAX/MJX + Brax PPO.

## Una sola vez: preparar la cuenta de Kaggle

1. Crea una cuenta en https://www.kaggle.com (gratis).
2. **Verifica tu teléfono**: *Settings* (tu avatar) → *Phone verification*. Sin esto Kaggle no deja
   usar GPU ni Internet.

## Cada vez: abrir y ejecutar el notebook

1. En Kaggle: **Create → New Notebook**.
2. Menú **File → Import Notebook** → pestaña **GitHub** (o **Link**) → pega
   `https://github.com/pablopedrosap/homeostatic-Alife/blob/main/kaggle/go2_body_budget_gpu.ipynb`
   → *Import*. (Alternativa: descarga el `.ipynb` del repositorio y súbelo con *File → Import Notebook →
   File*.)
3. Panel derecho → **Session options**:
   - **Accelerator → GPU T4 x2**
   - **Internet → On**
4. Ejecuta las celdas **1 a 4** en orden (Shift+Enter). La celda 2 debe imprimir algo como
   `[CudaDevice(id=0), CudaDevice(id=1)]`. Si dice CPU, vuelve al paso 3.
5. **Parte 0 (test de velocidad, unos 5 min):** ejecútala y mira la línea final
   `≈ X steps/s → 50M steps ≈ Y min per robot`. Ese número te dice cuánto tardará todo lo demás.

## Parte 1 (comparación) y Parte 2 (evolución): corridas largas

Las sesiones interactivas se cortan si cierras el navegador o estás inactivo. Para corridas largas:

1. Arriba a la derecha: **Save Version** → **Save & Run All (Commit)** → *Save*.
2. Kaggle ejecuta el notebook entero en segundo plano (hasta ~12 h) aunque cierres la pestaña.
3. Cuando termine: abre la versión guardada → pestaña **Output** → descarga `results/`
   (`compare_drives.csv`, `evolution_history.json`, gráficas `.png`, vídeo `.mp4`).

Tiempos aproximados (usa tu número de la Parte 0):
- **Parte 1:** 4 robots × (tiempo de 50M pasos) + misiones (~minutos).
- **Parte 2:** 2 grupos × 6 robots × 4 generaciones × (tiempo de 20M pasos). Si es demasiado, baja en la
  celda `pop`, `generations` o `num_timesteps`.

Si solo quieres la Parte 1, comenta (#) la celda de `G.evolve(...)` antes de *Save & Run All*.

## Qué cambiar para experimentar (todo en las celdas, sin tocar el código)

| Quieres… | Cambia… |
|---|---|
| un instinto más fuerte o más débil | `weights=(0.0, 0.2)` → p. ej. `(0.0, 0.1, 0.5)` |
| más fiabilidad estadística | `seeds=(0, 1)` → `(0, 1, 2, 3)` |
| robots mejor entrenados | `num_timesteps` |
| una población evolutiva mayor | `pop=6` → `pop=12` |

## Problemas típicos

- **"No GPU"**: Session options → Accelerator → GPU T4 x2, y reinicia la sesión.
- **Error al instalar / `jax` raro**: *Run → Restart & clear outputs* y ejecuta desde la celda 1. No
  cambies las versiones: `jax 0.9.2 + brax 0.14.2 + mujoco 4.0.0` están probadas juntas (versiones más
  nuevas de JAX rompen Brax 0.14.2).
- **El vídeo falla** (error de EGL/OpenGL): no afecta a los resultados; sáltate esa celda.
- **Out of memory**: en `G.train(...)` añade `num_envs=2048`.

## En Google Colab

El mismo notebook funciona: *Runtime → Change runtime type → T4 GPU*, y luego ejecuta las celdas.
Colab corta las sesiones con más facilidad, así que úsalo para la Parte 0 y pruebas cortas, y Kaggle
para las corridas largas.

## Qué está probado y qué no

Todo el pipeline (entrenamiento, misiones, vida con envejecimiento, vídeo, evolución) se ha ejecutado
de principio a fin a escala mínima en CPU con exactamente estas versiones. La escala real en GPU (4096
robots, decenas de millones de pasos) solo se puede comprobar en la GPU: si algo falla allí, copia el
error y lo arreglamos.
