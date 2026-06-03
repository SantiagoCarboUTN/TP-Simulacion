"""
TP 2.1 - Generadores Pseudoaleatorios
Universidad Tecnológica Nacional - FRRO
Materia: Simulación - 2026

Implementa:
  - Generador de Cuadrados Medios (Middle Squares - Von Neumann)
  - Generador Congruencial Lineal (GCL)
  - Comparación con random.random() de Python (Mersenne Twister)

Tests implementados (sobre GCL y comparados entre generadores):
  1. Test de Uniformidad (Chi-cuadrado)
  2. Test de Kolmogorov-Smirnov
  3. Test de Rachas (Runs Test)
  4. Test de Autocorrelación (Poker / Serial)

Genera todas las figuras necesarias para el informe LaTeX.
"""

import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 0. Reproducibilidad
# ─────────────────────────────────────────────────────────────────────────────
SEED = 12345
N    = 1000        # cantidad de números a generar por corrida


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERADORES
# ─────────────────────────────────────────────────────────────────────────────

class MiddleSquares:
    """
    Método de los Cuadrados Medios (Von Neumann, 1946).
    Toma una semilla de 'digits' dígitos, la eleva al cuadrado y extrae
    los dígitos del centro.  Normaliza a [0,1).

    Limitaciones conocidas: período corto, puede degenerar en ciclos
    triviales (0, o repetición inmediata).  Se incluye a modo de ejemplo
    histórico / comparativo.
    """
    def __init__(self, seed: int = SEED, digits: int = 8):
        self.digits  = digits
        self.modulus = 10 ** digits
        # Asegurar que la semilla tenga exactamente 'digits' dígitos
        self.state   = int(str(seed).zfill(digits)[:digits])

    def next(self) -> float:
        sq   = self.state ** 2
        sq_s = str(sq).zfill(2 * self.digits)
        start = (len(sq_s) - self.digits) // 2
        self.state = int(sq_s[start : start + self.digits])
        return self.state / self.modulus

    def generate(self, n: int) -> list[float]:
        return [self.next() for _ in range(n)]


class GCL:
    """
    Generador Congruencial Lineal (Linear Congruential Generator).
    Recurrencia:  X_{n+1} = (a * X_n + c) mod m
    Normalización: U_n = X_n / m  ∈ [0, 1)

    Parámetros predeterminados: los de glibc / POSIX (usados en rand() de C),
    que satisfacen el Teorema de Hull-Dobell para período completo m.
        m = 2^31  (2 147 483 648)
        a = 1 103 515 245
        c = 12 345
    """
    def __init__(self,
                 seed: int = SEED,
                 a:    int = 1_103_515_245,
                 c:    int = 12_345,
                 m:    int = 2**31):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def next(self) -> float:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def generate(self, n: int) -> list[float]:
        return [self.next() for _ in range(n)]


def python_random(n: int, seed: int = SEED) -> list[float]:
    """Mersenne Twister de la biblioteca estándar de Python."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TESTS ESTADÍSTICOS
# ─────────────────────────────────────────────────────────────────────────────

def test_chi_cuadrado(data: list[float],
                      k:    int   = 10,
                      alpha: float = 0.05) -> dict:
    """
    Test de bondad de ajuste Chi-cuadrado para uniformidad en [0,1).
    H0: la muestra proviene de una distribución U[0,1).

    Divide [0,1) en k intervalos iguales; compara frecuencias observadas
    con la frecuencia esperada n/k.

    Estadístico:  chi2 = sum( (Oi - Ei)^2 / Ei )
    Distribución: chi2(k-1) bajo H0.
    """
    n      = len(data)
    Ei     = n / k
    counts = [0] * k
    for x in data:
        idx = min(int(x * k), k - 1)
        counts[idx] += 1

    chi2_stat = sum((obs - Ei)**2 / Ei for obs in counts)
    # Valor crítico aproximado con fórmula de Wilson-Hilferty
    gl = k - 1
    # Usamos tabla de valores críticos para alpha=0.05 y alpha=0.01
    # chi2_crit(9, 0.05) ≈ 16.919
    CRITICAL = {(9,  0.05): 16.919,
                (9,  0.01): 21.666,
                (19, 0.05): 30.144,
                (19, 0.01): 36.191,
                (49, 0.05): 66.342,
                (49, 0.01): 74.919}
    chi2_crit = CRITICAL.get((gl, alpha), None)
    if chi2_crit is None:
        # Aproximación normal para gl > 30
        z_alpha = 1.645 if alpha == 0.05 else 2.326
        chi2_crit = gl * (1 - 2/(9*gl) + z_alpha * math.sqrt(2/(9*gl)))**3

    aprueba = chi2_stat < chi2_crit
    return {
        "nombre":    "Chi-cuadrado",
        "estadistico": round(chi2_stat,  4),
        "critico":     round(chi2_crit,  4),
        "gl":          gl,
        "alpha":       alpha,
        "aprueba_H0":  aprueba,
        "resultado":   "PASA" if aprueba else "FALLA",
    }


def test_ks(data: list[float], alpha: float = 0.05) -> dict:
    """
    Test de Kolmogorov-Smirnov para uniformidad.
    H0: la muestra proviene de U[0,1).

    Estadístico: D = max_x |F_n(x) - F(x)|
    Valor crítico (Massey 1951):
        D_crit ≈ c(alpha) / sqrt(n)
        c(0.05) = 1.36,  c(0.01) = 1.63
    """
    n     = len(data)
    xs    = sorted(data)
    D_max = 0.0
    for i, x in enumerate(xs):
        Fn_upper = (i + 1) / n
        Fn_lower = i / n
        D_max = max(D_max, abs(Fn_upper - x), abs(Fn_lower - x))

    c_alpha = {0.05: 1.36, 0.01: 1.63}.get(alpha, 1.36)
    D_crit  = c_alpha / math.sqrt(n)
    aprueba = D_max < D_crit
    return {
        "nombre":      "Kolmogorov-Smirnov",
        "estadistico": round(D_max,  6),
        "critico":     round(D_crit, 6),
        "alpha":       alpha,
        "aprueba_H0":  aprueba,
        "resultado":   "PASA" if aprueba else "FALLA",
    }


def test_rachas(data: list[float], alpha: float = 0.05) -> dict:
    """
    Test de Rachas (Runs Test) — independencia.
    H0: la secuencia es independiente (aleatoria).

    Una 'racha' es una subsecuencia de valores todos por encima o todos
    por debajo de la mediana muestral.  Bajo H0 el número de rachas R
    sigue una distribución aproximadamente normal:
        E[R] = 2*n1*n2/(n1+n2) + 1
        Var[R] = 2*n1*n2*(2*n1*n2 - n1 - n2) / ((n1+n2)^2*(n1+n2-1))
        Z = (R - E[R]) / sqrt(Var[R])
    """
    mediana = sorted(data)[len(data) // 2]
    # signos: True si >= mediana, False si <
    signos = [x >= mediana for x in data]
    n1 = sum(signos)
    n2 = len(signos) - n1

    # Contar rachas
    R = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i-1]:
            R += 1

    ER   = 2 * n1 * n2 / (n1 + n2) + 1
    VarR = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
            / ((n1 + n2)**2 * (n1 + n2 - 1)))
    Z    = (R - ER) / math.sqrt(VarR)

    Z_crit = {0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    aprueba = abs(Z) < Z_crit
    return {
        "nombre":      "Rachas",
        "estadistico": round(abs(Z),  4),
        "critico":     Z_crit,
        "rachas":      R,
        "E_rachas":    round(ER, 2),
        "alpha":       alpha,
        "aprueba_H0":  aprueba,
        "resultado":   "PASA" if aprueba else "FALLA",
    }


def test_autocorrelacion(data: list[float],
                         lag:   int   = 1,
                         alpha: float = 0.05) -> dict:
    """
    Test de Autocorrelación (correlación de Pearson con retardo lag).
    H0: no hay correlación entre X_i y X_{i+lag}.

    r = Σ(xi - x̄)(x_{i+lag} - x̄) / [ (n-lag) * s^2 ]
    Bajo H0, r ≈ N(0, 1/n) → Z = r * sqrt(n-lag)
    """
    n    = len(data)
    xbar = sum(data) / n
    s2   = sum((x - xbar)**2 for x in data) / n
    if s2 == 0:
        return {"nombre": "Autocorrelación", "resultado": "FALLA (varianza 0)"}

    num  = sum((data[i] - xbar) * (data[i + lag] - xbar)
               for i in range(n - lag))
    r    = num / ((n - lag) * s2)
    Z    = r * math.sqrt(n - lag)

    Z_crit = {0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    aprueba = abs(Z) < Z_crit
    return {
        "nombre":      f"Autocorrelación (lag={lag})",
        "estadistico": round(abs(Z), 4),
        "critico":     Z_crit,
        "r":           round(r, 6),
        "alpha":       alpha,
        "aprueba_H0":  aprueba,
        "resultado":   "PASA" if aprueba else "FALLA",
    }


def test_poker(data: list[float], d: int = 5, alpha: float = 0.05) -> dict:
    """
    Test del Póker (para uniformidad en categorías de dígitos).
    Agrupa los números en grupos de d, clasifica cada grupo según
    cuántos valores distintos tiene (todos distintos, un par, etc.)
    y compara con frecuencias esperadas mediante Chi-cuadrado.

    Categorías para d=5:
      5 distintos, 1 par, 2 pares, 3 iguales, full, poker, quintilla
    """
    grupos = [data[i:i+d] for i in range(0, len(data) - len(data) % d, d)]
    m = len(grupos)

    # Discretizar a 10 dígitos
    def clasificar(grupo):
        digitos = [int(x * 10) for x in grupo]
        c = Counter(digitos)
        vals = sorted(c.values(), reverse=True)
        if   vals == [5]:           return "Quintilla"
        elif vals == [4, 1]:        return "Póker"
        elif vals == [3, 2]:        return "Full"
        elif vals == [3, 1, 1]:     return "Trío"
        elif vals == [2, 2, 1]:     return "Dos pares"
        elif vals == [2, 1, 1, 1]:  return "Un par"
        else:                       return "5 distintos"

    # Probabilidades teóricas (base 10, d=5)
    p_teoricas = {
        "5 distintos": 0.30240,
        "Un par":      0.50400,
        "Dos pares":   0.10800,
        "Trío":        0.07200,
        "Full":        0.00900,
        "Póker":       0.00450,
        "Quintilla":   0.00010,
    }

    obs = Counter(clasificar(g) for g in grupos)
    chi2 = sum((obs.get(cat, 0) - m * p)**2 / (m * p)
               for cat, p in p_teoricas.items())
    gl       = len(p_teoricas) - 1
    chi2_crit = 12.592  # chi2(6, 0.05)
    aprueba   = chi2 < chi2_crit
    return {
        "nombre":      "Póker",
        "estadistico": round(chi2, 4),
        "critico":     chi2_crit,
        "gl":          gl,
        "alpha":       alpha,
        "aprueba_H0":  aprueba,
        "resultado":   "PASA" if aprueba else "FALLA",
        "obs":         dict(obs),
    }


def ejecutar_todos_tests(data: list[float], nombre: str) -> list[dict]:
    """Ejecuta la batería completa de tests sobre 'data'."""
    resultados = []
    resultados.append(test_chi_cuadrado(data))
    resultados.append(test_ks(data))
    resultados.append(test_rachas(data))
    resultados.append(test_autocorrelacion(data, lag=1))
    resultados.append(test_poker(data))
    for r in resultados:
        r["generador"] = nombre
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

ms_gen  = MiddleSquares(seed=SEED)
gcl_gen = GCL(seed=SEED)

ms_data  = ms_gen.generate(N)
gcl_data = gcl_gen.generate(N)
py_data  = python_random(N, seed=SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 4. EJECUCIÓN DE TESTS
# ─────────────────────────────────────────────────────────────────────────────

resultados_ms  = ejecutar_todos_tests(ms_data,  "Cuadrados Medios")
resultados_gcl = ejecutar_todos_tests(gcl_data, "GCL (glibc)")
resultados_py  = ejecutar_todos_tests(py_data,  "Python (MT)")

todos = resultados_ms + resultados_gcl + resultados_py

# ─────────────────────────────────────────────────────────────────────────────
# 5. IMPRESIÓN DE RESULTADOS EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("TP 2.1 – GENERADORES PSEUDOALEATORIOS – Resultados de tests")
print("=" * 70)

for gen_nombre, res_list in [
    ("Cuadrados Medios", resultados_ms),
    ("GCL (glibc)",      resultados_gcl),
    ("Python (MT)",      resultados_py),
]:
    print(f"\n{'─'*70}")
    print(f"  Generador: {gen_nombre}")
    print(f"{'─'*70}")
    for r in res_list:
        print(f"  {r['nombre']:30s}  "
              f"estadístico={r['estadistico']:>10}  "
              f"crítico={r.get('critico','  N/A'):>8}  "
              f"→ {r['resultado']}")

print("\n")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

STYLE   = {"ms":  {"color": "#e05c5c", "label": "Cuadrados Medios"},
           "gcl": {"color": "#3a7dbf", "label": "GCL (glibc)"},
           "py":  {"color": "#27ae60", "label": "Python (MT)"}}


# ── Fig 1: Histogramas de uniformidad ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
fig.suptitle("Histogramas de uniformidad (n=1000, k=10 intervalos)",
             fontsize=13, fontweight="bold")

for ax, (key, d) in zip(axes, [("ms", ms_data), ("gcl", gcl_data), ("py", py_data)]):
    ax.hist(d, bins=10, range=(0, 1),
            color=STYLE[key]["color"], edgecolor="white", alpha=0.85)
    ax.axhline(N / 10, color="black", linestyle="--", linewidth=1.2,
               label=f"Esperado ({N//10})")
    ax.set_title(STYLE[key]["label"], fontsize=11)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("hist_uniformidad.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] hist_uniformidad.png")


# ── Fig 2: Dispersión X_n vs X_{n+1} (estructura serial) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Gráfico de dispersión $X_n$ vs $X_{n+1}$ (primeros 500 pares)",
             fontsize=13, fontweight="bold")

for ax, (key, d) in zip(axes, [("ms", ms_data), ("gcl", gcl_data), ("py", py_data)]):
    xs = d[:-1][:500]
    ys = d[1:][:500]
    ax.scatter(xs, ys, s=2.5, alpha=0.5, color=STYLE[key]["color"])
    ax.set_title(STYLE[key]["label"], fontsize=11)
    ax.set_xlabel("$X_n$")
    ax.set_ylabel("$X_{n+1}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("scatter_serial.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] scatter_serial.png")


# ── Fig 3: Secuencia temporal (primeros 200 valores) ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ms_data[:200],  color=STYLE["ms"]["color"],  alpha=0.8,
        linewidth=0.9, label=STYLE["ms"]["label"])
ax.plot(gcl_data[:200], color=STYLE["gcl"]["color"], alpha=0.8,
        linewidth=0.9, label=STYLE["gcl"]["label"])
ax.plot(py_data[:200],  color=STYLE["py"]["color"],  alpha=0.8,
        linewidth=0.9, label=STYLE["py"]["label"])
ax.set_title("Secuencia temporal de los tres generadores (primeros 200 valores)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Índice $n$")
ax.set_ylabel("Valor $U_n$")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("serie_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] serie_temporal.png")


# ── Fig 4: Función de distribución acumulada empírica (ECDF) ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for key, d in [("ms", ms_data), ("gcl", gcl_data), ("py", py_data)]:
    xs = sorted(d)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    ax.plot(xs, ys, linewidth=1.4,
            color=STYLE[key]["color"], label=STYLE[key]["label"])
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="U[0,1) teórica")
ax.set_title("ECDF vs distribución teórica U[0,1)", fontsize=12, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("$F_n(x)$")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ecdf.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] ecdf.png")


# ── Fig 5: Tabla visual de resultados de tests ───────────────────────────────
gen_labels  = ["Cuadrados\nMedios", "GCL\n(glibc)", "Python\n(MT)"]
test_labels = ["Chi²", "K-S", "Rachas", "Autocorr.", "Póker"]
results_all = [resultados_ms, resultados_gcl, resultados_py]

# Matriz de colores: verde=PASA, rojo=FALLA
colors = []
texts  = []
for res_list in results_all:
    row_c = []
    row_t = []
    for r in res_list:
        ok = r["aprueba_H0"]
        row_c.append("#27ae60" if ok else "#e05c5c")
        row_t.append("PASA" if ok else "FALLA")
    colors.append(row_c)
    texts.append(row_t)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.set_xlim(-0.5, len(test_labels) - 0.5)
ax.set_ylim(-0.5, len(gen_labels) - 0.5)

for i, gen in enumerate(gen_labels):
    for j, test in enumerate(test_labels):
        rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                              color=colors[i][j], alpha=0.85)
        ax.add_patch(rect)
        ax.text(j, i, texts[i][j], ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

ax.set_xticks(range(len(test_labels)))
ax.set_xticklabels(test_labels, fontsize=11)
ax.set_yticks(range(len(gen_labels)))
ax.set_yticklabels(gen_labels, fontsize=10)
ax.set_title("Resumen de tests estadísticos (α = 0.05)",
             fontsize=12, fontweight="bold")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("tabla_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] tabla_tests.png")


# ── Fig 6: Autocorrelación para distintos lags (sólo GCL y Python) ───────────
lags    = list(range(1, 51))
r_gcl   = []
r_py    = []
r_ms    = []

for lag in lags:
    def autocorr(d, lag):
        n    = len(d)
        xbar = sum(d) / n
        s2   = sum((x - xbar)**2 for x in d) / n
        if s2 == 0:
            return 0.0
        num = sum((d[i] - xbar) * (d[i + lag] - xbar) for i in range(n - lag))
        return num / ((n - lag) * s2)

    r_gcl.append(autocorr(gcl_data, lag))
    r_py.append(autocorr(py_data,   lag))
    r_ms.append(autocorr(ms_data,   lag))

z_95 = 1.96 / math.sqrt(N)
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(lags, r_gcl, width=0.3, align="edge",
       color=STYLE["gcl"]["color"], alpha=0.75, label=STYLE["gcl"]["label"])
ax.bar([l + 0.3 for l in lags], r_py, width=0.3, align="edge",
       color=STYLE["py"]["color"],  alpha=0.75, label=STYLE["py"]["label"])
ax.bar([l - 0.3 for l in lags], r_ms, width=0.3, align="edge",
       color=STYLE["ms"]["color"],  alpha=0.75, label=STYLE["ms"]["label"])
ax.axhline( z_95, color="black", linestyle="--", linewidth=1, label=f"±1.96/√n")
ax.axhline(-z_95, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Retardo (lag)")
ax.set_ylabel("Autocorrelación $r$")
ax.set_title("Autocorrelación para lags 1–50", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("autocorrelacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] autocorrelacion.png")


# ── Fig 7: Comparación de medias y desviaciones estándar ─────────────────────
stats = {
    "Cuadrados Medios": (np.mean(ms_data),  np.std(ms_data)),
    "GCL (glibc)":      (np.mean(gcl_data), np.std(gcl_data)),
    "Python (MT)":      (np.mean(py_data),  np.std(py_data)),
    "U[0,1) teórica":   (0.5,               1/math.sqrt(12)),
}

nombres = list(stats.keys())
medias  = [stats[k][0] for k in nombres]
desvs   = [stats[k][1] for k in nombres]
colores = [STYLE["ms"]["color"], STYLE["gcl"]["color"],
           STYLE["py"]["color"], "gray"]

x = np.arange(len(nombres))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Estadísticos descriptivos vs valor teórico U[0,1)",
             fontsize=12, fontweight="bold")

ax1.bar(x, medias, width=0.5, color=colores, alpha=0.85, edgecolor="white")
ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="μ teórico = 0.5")
ax1.set_title("Media muestral")
ax1.set_xticks(x)
ax1.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("$\\bar{X}$")
ax1.legend(fontsize=9)
ax1.set_ylim(0.45, 0.55)

ax2.bar(x, desvs, width=0.5, color=colores, alpha=0.85, edgecolor="white")
ax2.axhline(1/math.sqrt(12), color="black", linestyle="--", linewidth=1.2,
            label="σ teórico = 1/√12")
ax2.set_title("Desviación estándar")
ax2.set_xticks(x)
ax2.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
ax2.set_ylabel("$s$")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("estadisticos.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] estadisticos.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESUMEN FINAL IMPRESO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("TABLA RESUMEN")
print("=" * 70)
header = f"{'Generador':<22} {'Test':<28} {'Estadístico':>12} {'Crítico':>9} {'Resultado':>8}"
print(header)
print("-" * 70)
for r in todos:
    print(f"{r['generador']:<22} {r['nombre']:<28} "
          f"{str(r['estadistico']):>12} "
          f"{str(r.get('critico', 'N/A')):>9} "
          f"{r['resultado']:>8}")

print("\n[✓] Todas las figuras generadas correctamente.")
print("[✓] Script finalizado.")
