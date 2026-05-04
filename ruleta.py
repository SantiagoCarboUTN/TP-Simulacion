"""
TP 1.1 - Simulación de una Ruleta Europea (0-36)
Universidad Tecnológica Nacional - FRRO
Uso: python ruleta.py -c <corridas> -n <tiradas> -e <número_elegido>
Ejemplo: python ruleta.py -c 5 -n 1000 -e 7
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────
# Valores teóricos esperados (ruleta europea 0-36)
# ──────────────────────────────────────────────
NUMEROS = list(range(37))          # 0..36
P = 1 / 37                         # probabilidad de acertar cualquier número
VPE = sum(NUMEROS) / 37            # media esperada  = 18.0
VVE_TOTAL = sum((i - VPE) ** 2 for i in NUMEROS) / 37  # varianza total ≈ 114.0
VDE_TOTAL = np.sqrt(VVE_TOTAL)     # desvío estándar total ≈ 10.677
VVE_X = P * (1 - P)               # varianza de Bernoulli para el número elegido
VDE_X = np.sqrt(VVE_X)            # desvío estándar de Bernoulli


def simular_corrida(n_tiradas: int, n_elegido: int):
    """
    Simula n_tiradas tiradas de ruleta europea y devuelve
    los estadísticos acumulados a lo largo de la corrida.
    """
    resultados = []          # todas las tiradas
    indicador = []           # 1 si salió n_elegido, 0 si no

    frn = []   # frecuencia relativa acumulada de n_elegido
    vpn = []   # valor promedio acumulado de las tiradas
    vdn = []   # desvío estándar acumulado del indicador de n_elegido
    vvn = []   # varianza acumulada del indicador de n_elegido

    cuenta = 0

    for i in range(1, n_tiradas + 1):
        tirada = random.randint(0, 36)
        resultados.append(tirada)

        hit = 1 if tirada == n_elegido else 0
        indicador.append(hit)
        cuenta += hit

        # Frecuencia relativa acumulada del número elegido
        frn.append(cuenta / i)

        # Promedio acumulado de todas las tiradas
        vpn.append(sum(resultados) / i)

        # Varianza y desvío acumulados del indicador (Bernoulli)
        if i > 1:
            vvn.append(float(np.var(indicador)))
            vdn.append(float(np.std(indicador)))
        else:
            vvn.append(0.0)
            vdn.append(0.0)

    return frn, vpn, vdn, vvn


def graficar_corrida_unica(tiradas_eje, datos, n_elegido, n_tiradas, idx_corrida=0):
    """Genera las 4 gráficas requeridas para una sola corrida."""
    frn, vpn, vdn, vvn = datos

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Simulación Ruleta Europea  –  Corrida {idx_corrida + 1}  |  "
        f"n = {n_tiradas} tiradas  |  Número elegido: {n_elegido}",
        fontsize=13, fontweight="bold"
    )

    # ── Gráfica 1: Frecuencia Relativa ──
    ax = axes[0, 0]
    ax.plot(tiradas_eje, frn, color="red", linewidth=0.9, label="frn (simulado)")
    ax.axhline(P, color="blue", linewidth=2,
               label=f"fre = 1/37 ≈ {P:.5f}")
    ax.set_xlabel("n  (número de tiradas)")
    ax.set_ylabel("fr  (frecuencia relativa)")
    ax.set_title(f"Frecuencia Relativa del número {n_elegido}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Gráfica 2: Valor Promedio ──
    ax = axes[0, 1]
    ax.plot(tiradas_eje, vpn, color="red", linewidth=0.9, label="vpn (simulado)")
    ax.axhline(VPE, color="blue", linewidth=2,
               label=f"vpe = {VPE:.2f}")
    ax.set_xlabel("n  (número de tiradas)")
    ax.set_ylabel("vp  (valor promedio de las tiradas)")
    ax.set_title("Valor Promedio de las Tiradas")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Gráfica 3: Desvío Estándar ──
    ax = axes[1, 0]
    ax.plot(tiradas_eje, vdn, color="red", linewidth=0.9,
            label=f"vd del número {n_elegido} (simulado)")
    ax.axhline(VDE_X, color="blue", linewidth=2,
               label=f"vde = √(p·(1−p)) ≈ {VDE_X:.5f}")
    ax.set_xlabel("n  (número de tiradas)")
    ax.set_ylabel("vd  (valor del desvío)")
    ax.set_title(f"Desvío Estándar del número {n_elegido}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Gráfica 4: Varianza ──
    ax = axes[1, 1]
    ax.plot(tiradas_eje, vvn, color="red", linewidth=0.9,
            label=f"vvn del número {n_elegido} (simulado)")
    ax.axhline(VVE_X, color="blue", linewidth=2,
               label=f"vve = p·(1−p) ≈ {VVE_X:.5f}")
    ax.set_xlabel("n  (número de tiradas)")
    ax.set_ylabel("vv  (valor de la varianza)")
    ax.set_title(f"Varianza del número {n_elegido}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"graficas_corrida{idx_corrida + 1}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  [✓] Guardado: {fname}")
    plt.close()


def graficar_multicorridas(tiradas_eje, todas_frn, todas_vpn,
                            todas_vdn, todas_vvn, n_elegido, n_tiradas, n_corridas):
    """Genera 4 gráficas con todas las corridas superpuestas."""
    colores = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Simulación Ruleta Europea  –  {n_corridas} corridas superpuestas  |  "
        f"n = {n_tiradas} tiradas  |  Número elegido: {n_elegido}",
        fontsize=13, fontweight="bold"
    )

    configs = [
        (todas_frn,  P,      "fr", "frn", f"fre = {P:.5f}",    f"Frecuencia Relativa del número {n_elegido}"),
        (todas_vpn,  VPE,    "vp", "vpn", f"vpe = {VPE:.2f}",  "Valor Promedio de las Tiradas"),
        (todas_vdn,  VDE_X,  "vd", "vdn", f"vde ≈ {VDE_X:.5f}", f"Desvío Estándar del número {n_elegido}"),
        (todas_vvn,  VVE_X,  "vv", "vvn", f"vve ≈ {VVE_X:.5f}", f"Varianza del número {n_elegido}"),
    ]

    for ax, (datos_corridas, esperado, ylabel_short, label_sim, label_esp, titulo) \
            in zip(axes.flat, configs):
        for c, datos in enumerate(datos_corridas):
            ax.plot(tiradas_eje, datos,
                    color=colores[c % len(colores)],
                    linewidth=0.7, alpha=0.85,
                    label=f"Corrida {c + 1}")
        ax.axhline(esperado, color="black", linewidth=2.2,
                   linestyle="--", label=label_esp, zorder=5)
        ax.set_xlabel("n  (número de tiradas)")
        ax.set_ylabel(ylabel_short)
        ax.set_title(titulo)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "graficas_multicorridas.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  [✓] Guardado: {fname}")
    plt.close()


def imprimir_resumen(n_corridas, n_tiradas, n_elegido,
                     todas_frn, todas_vpn, todas_vdn, todas_vvn):
    sep = "=" * 55
    print(f"\n{sep}")
    print("  SIMULACIÓN RULETA EUROPEA (0 – 36)")
    print(sep)
    print(f"  Corridas : {n_corridas}  |  Tiradas/corrida : {n_tiradas}")
    print(f"  Número elegido : {n_elegido}")
    print(f"\n  --- Valores Teóricos Esperados ---")
    print(f"  Frecuencia relativa esperada  : {P:.6f}  (1/37)")
    print(f"  Valor promedio esperado        : {VPE:.4f}")
    print(f"  Varianza (indicador X) esperada: {VVE_X:.6f}  (p·(1−p))")
    print(f"  Desvío  (indicador X) esperado : {VDE_X:.6f}")
    print(f"\n  --- Valores Simulados (última corrida) ---")
    print(f"  Frecuencia relativa simulada   : {todas_frn[-1][-1]:.6f}")
    print(f"  Valor promedio simulado         : {todas_vpn[-1][-1]:.4f}")
    print(f"  Varianza simulada               : {todas_vvn[-1][-1]:.6f}")
    print(f"  Desvío simulado                 : {todas_vdn[-1][-1]:.6f}")
    print(sep)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TP1.1 – Simulación de Ruleta Europea | UTN FRRO"
    )
    parser.add_argument("-c", type=int, default=5,
                        help="Número de corridas (default: 5)")
    parser.add_argument("-n", type=int, default=1000,
                        help="Número de tiradas por corrida (default: 1000)")
    parser.add_argument("-e", type=int, default=7,
                        help="Número elegido 0-36 (default: 7)")
    args = parser.parse_args()

    n_corridas = args.c
    n_tiradas = args.n
    n_elegido = args.e

    if not (0 <= n_elegido <= 36):
        parser.error("El número elegido debe estar entre 0 y 36.")
    if n_tiradas < 2:
        parser.error("Se necesitan al menos 2 tiradas.")
    if n_corridas < 1:
        parser.error("Se necesita al menos 1 corrida.")

    tiradas_eje = list(range(1, n_tiradas + 1))

    todas_frn, todas_vpn, todas_vdn, todas_vvn = [], [], [], []

    print(f"\nSimulando {n_corridas} corridas de {n_tiradas} tiradas "
          f"(número elegido: {n_elegido})...")

    for c in range(n_corridas):
        frn, vpn, vdn, vvn = simular_corrida(n_tiradas, n_elegido)
        todas_frn.append(frn)
        todas_vpn.append(vpn)
        todas_vdn.append(vdn)
        todas_vvn.append(vvn)

        # Gráficas individuales por corrida
        graficar_corrida_unica(tiradas_eje,
                               (frn, vpn, vdn, vvn),
                               n_elegido, n_tiradas, idx_corrida=c)

    # Gráfica con todas las corridas superpuestas
    graficar_multicorridas(tiradas_eje,
                           todas_frn, todas_vpn, todas_vdn, todas_vvn,
                           n_elegido, n_tiradas, n_corridas)

    imprimir_resumen(n_corridas, n_tiradas, n_elegido,
                     todas_frn, todas_vpn, todas_vdn, todas_vvn)


if __name__ == "__main__":
    main()
