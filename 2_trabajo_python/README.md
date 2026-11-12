# Implementación Python del ejemplo Damper

Esta carpeta contiene una reimplementación documentada del ejemplo numérico del
artículo de Schön et al. (2018). No ejecuta LibBi: implementa directamente en
Python el mismo modelo de estados y los algoritmos Monte Carlo descritos en el
artículo.

Para instalar el entorno desde cero, consulte primero el
[`README principal`](../README.md).

## Objetivo

Estimar la distribución posterior de

```text
theta = (f_c, c_0, k, p)
```

a partir del desplazamiento observado `y` almacenado en `data/obs.nc`.

- `f_c`: fricción de Coulomb.
- `c_0`: amortiguamiento viscoso.
- `k`: rigidez del resorte.
- `p`: exponente no lineal.

## Modelo implementado

La posición y la velocidad evolucionan mediante Euler hacia adelante:

```text
s[t+1]    = s[t] + T_s * velocidad[t]

vel[t+1]  = vel[t]
            + (T_s / m) * (
                -f_c * signo(vel[t])
                -c_0 * vel[t]
                -k * signo(s[t]) * |s[t]|**p
              )
            + v[t]

y[t]      = s[t] + e[t]
```

Constantes utilizadas:

| Cantidad | Valor |
|---|---:|
| `T_s` | 0.1 |
| `m` | 2.0 |
| `sigma_v` | 0.01 |
| `sigma_y` | 0.1 |
| `s[0]` | 0.5 |
| `velocidad[0]` | 0.0 |

La masa sigue el `Damper.bi` del apéndice. El texto del artículo escribe `m=8`,
pero ese valor contradice el código original y no reproduce adecuadamente
`obs.nc`.

## Prioris

```text
f_c ~ Gamma(forma=2, escala=0.01)
c_0 ~ Gamma(forma=2, escala=1)
k   ~ Gamma(forma=4, escala=0.3)
p   ~ Uniforme(0, 1)
```

## Algoritmos

### Filtro de partículas bootstrap

`filtro_particulas()` implementa el Algoritmo 2 y estima
`log p(y | theta)`:

1. Inicializa 256 posibles estados.
2. Evalúa la observación para obtener pesos.
3. Remuestrea los estados según esos pesos.
4. Propaga las partículas mediante la dinámica.
5. Repite el proceso para toda la serie.

Se usa remuestreo sistemático, un método insesgado de baja varianza recomendado
en la sección 6.1 del artículo.

### Particle Metropolis–Hastings

`pmh()` implementa el Algoritmo 3:

1. Propone un nuevo vector `theta` mediante gaussianas truncadas.
2. Ejecuta el filtro de partículas para estimar su verosimilitud.
3. Calcula la probabilidad de aceptación de la ecuación (33), incluido el
   cociente de las densidades de propuesta truncadas.
4. Acepta el candidato o conserva conjuntamente el vector y la estimación de
   verosimilitud anteriores.

## Archivos

| Ruta | Descripción |
|---|---|
| `pmh_damper.py` | Modelo, prioris, propuesta truncada, filtro bootstrap, PMH y guardado NetCDF |
| `ver_resultados.py` | Lectura del posterior, figuras y tabla resumen |
| `requirements.txt` | Versiones de las dependencias utilizadas |
| `data/obs.nc` | Copia de los datos simulados del artículo |
| `resultados_python/posterior_python_corregido.nc` | Última cadena completa de 10 000 iteraciones |
| `resultados_python/posterior_diagnostico.nc` | Corrida corta usada para validar la implementación |
| `resultados_python/posterior_python.nc` | Corrida antigua de 40 iteraciones; no usar para inferencia |
| `figuras/01_datos_observados.png` y `.pdf` | Serie observada en segundos |
| `figuras/02_posteriores.png` y `.pdf` | Posteriores, prioris y valores verdaderos |
| `figuras/03_trazas.png` y `.pdf` | Trazas MCMC completas |
| `figuras/resumen_posterior.csv` | Media, mediana, desviación e intervalo de cada parámetro |

## Ejecutar PMH desde esta carpeta

Active primero el entorno creado según el README principal. Después puede elegir
las iteraciones y las partículas:

```powershell
python pmh_damper.py --iter 10000 --part 256 --semilla 1025884781
```

Valores disponibles:

```text
--iter N              longitud de la cadena de parámetros
--part N              partículas por estimación de verosimilitud
--semilla N           semilla del generador aleatorio
--inicio FC C0 K P    punto inicial opcional
--salida RUTA         ruta del NetCDF producido
```

El programa utiliza 10 000 iteraciones y 256 partículas si se omiten `--iter` y
`--part`:

```powershell
python pmh_damper.py
```

### Última corrida

```powershell
python pmh_damper.py `
  --iter 10000 `
  --part 256 `
  --semilla 1025884781 `
  --inicio 0.1 1 3 0.4
```

Esta ejecución tardó cerca de 8.35 minutos en Windows 11 con Python 3.14.5 y
obtuvo una tasa de aceptación de 33.24 %.

El punto inicial `[0.1, 1, 3, 0.4]` se escogió deliberadamente apartado de los
valores verdaderos (`[0.0114, 0.713, 2.155, 0.577]`) para observar cómo la
cadena se aleja de esa región y converge hacia el modo posterior. Para un uso
ordinario, omita `--inicio`.

## Generar resultados gráficos

Después de ejecutar PMH:

```powershell
python ver_resultados.py
```

Cada gráfico se guarda en dos formatos:

- `.pdf`: salida vectorial recomendada para informes, presentaciones, LaTeX e
  impresión; puede ampliarse sin perder nitidez.
- `.png`: vista previa de 300 dpi, conveniente para GitHub y documentos que no
  acepten gráficos vectoriales.

No se genera EPS porque PDF conserva vectores y transparencias, tiene mejor
soporte actual y suele producir archivos más fáciles de incorporar.

El script prefiere `posterior_python_corregido.nc`. Si ese archivo no existe,
puede leer la corrida antigua para compatibilidad, pero mostrará una advertencia
cuando la cadena sea demasiado corta o tenga pocos valores distintos.

### Figura de datos

`01_datos_observados.png` convierte el índice de `obs.nc` a tiempo físico:

```text
tiempo_en_segundos = indice * 0.1
```

### Figura posterior

`02_posteriores.png` reproduce la estructura de la Figura 4 del artículo:

- histograma posterior azul;
- densidad prior negra;
- línea roja en el valor verdadero.

### Burn-in

Para histogramas y resúmenes se descarta el primer 20 % de la cadena. Con
10 000 iteraciones:

```text
Burn-in:             2 000 iteraciones
Muestras analizadas: 8 000 iteraciones
```

Las muestras no se borran del NetCDF. `03_trazas.png` muestra la cadena completa,
incluido el burn-in, para permitir su inspección.

## Resultado de la última corrida

Después del burn-in de 2 000 iteraciones:

| Parámetro | Verdadero | Media | Desviación | IC 95 % |
|---|---:|---:|---:|---:|
| `f_c` | 0.011414 | 0.012811 | 0.006870 | [0.002162, 0.027222] |
| `c_0` | 0.712997 | 0.713497 | 0.058694 | [0.601190, 0.827582] |
| `k` | 2.155382 | 2.112739 | 0.131920 | [1.833432, 2.320786] |
| `p` | 0.577213 | 0.566176 | 0.030508 | [0.501378, 0.615200] |

## Interpretación y convergencia

Las iteraciones MCMC son correlacionadas. La última cadena tiene una aceptación
de 33.24 %, pero sus tamaños efectivos de muestra son bajos, aproximadamente
entre 2 y 16 según el parámetro. Por ello:

- los histogramas reproducen razonablemente la forma publicada;
- los resúmenes deben considerarse aproximaciones Monte Carlo;
- una tasa de aceptación razonable no demuestra convergencia;
- para un análisis formal conviene ejecutar varias cadenas desde puntos iniciales
  dispersos, calcular `R-hat` y aumentar la longitud hasta obtener un ESS mayor.

El artículo no especifica una semilla ni proporciona una cadena posterior
utilizable. Por eso no se espera igualdad muestra por muestra con LibBi, sino
resultados estadísticamente comparables.
