"""
TP 1.2 - Simulacion de Estrategias de Apuestas en Ruleta Europea
Universidad Tecnologica Nacional - FRRO

TIPOS DE APUESTA (-t):
  c   color      rojo(r) / negro(k)              18 nums  pago 1:1
  d   docena     1ra(1) / 2da(2) / 3ra(3)        12 nums  pago 2:1
  col columna    1ra(1) / 2da(2) / 3ra(3)        12 nums  pago 2:1
  n   pleno      numero exacto 0..36              1 num    pago 35:1

Uso:
  python programa.py -c 1000 -n 500 -s m -a f -t c  -e r
  python programa.py -c 1000 -n 500 -s d -a i -t d  -e 2
  python programa.py -c 1000 -n 500 -s f -a f -t n  -e 17
  python programa.py -c 1000 -n 500 -s m -a f -t ca -e 8-9
  python programa.py -c 1000 -n 500 -s o -a i -t tr -e 4
  python programa.py -c 1000 -n 500 -s m -a f -t se -e 1
"""

import argparse, random, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Compatibilidad UTF-8 en consola Windows --
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════
# CONSTANTES DE RULETA EUROPEA
# ═══════════════════════════════════════════════════════
NUMEROS = list(range(37))   # 0..36
ROJOS   = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
NEGROS  = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
PARES   = {n for n in range(2,37,2)}   # 0 NO cuenta como par
IMPARES = {n for n in range(1,37,2)}
DOCENA  = {1: set(range(1,13)),  2: set(range(13,25)), 3: set(range(25,37))}
COLUMNA = {1: set(range(1,37,3)), 2: set(range(2,37,3)), 3: set(range(3,37,3))}

# ═══════════════════════════════════════════════════════
# TABLA DE APUESTAS  (fuente: casino.es, 888casino.es)
#   pago_neto : multiplicador sobre la apuesta si se gana
#   prob      : probabilidad teorica exacta en ruleta europea
# ═══════════════════════════════════════════════════════
TIPOS_APUESTA = {
    'c'  : {'nombre': 'Color',        'pago': 1,  'prob': 18/37,  'cubre': 18},
    'd'  : {'nombre': 'Docena',       'pago': 2,  'prob': 12/37,  'cubre': 12},
    'col': {'nombre': 'Columna',      'pago': 2,  'prob': 12/37,  'cubre': 12},
    'n'  : {'nombre': 'Pleno',        'pago': 35, 'prob':  1/37,  'cubre':  1},
}

# ═══════════════════════════════════════════════════════
# CONSTRUIR SETS DE NUMEROS SEGUN APUESTA
# ═══════════════════════════════════════════════════════

def numeros_ganadores(tipo, eleccion):
    """Retorna el conjunto de numeros que hacen ganar esta apuesta."""
    if tipo == 'c':
        return PARES if eleccion == 'p' else IMPARES
    if tipo == 'd':
        return DOCENA[int(eleccion)]
    if tipo == 'col':
        return COLUMNA[int(eleccion)]
    if tipo == 'n':
        return {int(eleccion)}
    raise ValueError(f"Tipo de apuesta desconocido: {tipo}")

def resolver_apuesta(numero_ruleta, tipo, eleccion):
    """Retorna (gano: bool, pago_neto: int)"""
    ganadores = numeros_ganadores(tipo, eleccion)
    gano = numero_ruleta in ganadores
    return gano, TIPOS_APUESTA[tipo]['pago']

# ═══════════════════════════════════════════════════════
# ESTRATEGIAS DE APUESTA
# Las estrategias clasicas estan disenadas para apuestas 1:1
# Para otros pagos la apuesta necesaria para recuperar se ajusta:
#   apuesta_recuperacion = perdida_acumulada / pago_neto
# ═══════════════════════════════════════════════════════

def apostar_martingala(historial, base, pago):
    """
    Ajustado al pago: apuesta lo necesario para recuperar todo lo perdido
    mas ganar 1 unidad base.  Para 1:1 equivale a doblar.
    """
    if not historial or historial[-1]['gano']:
        return base
    perdida_acum = 0
    for h in reversed(historial):
        if not h['gano']:
            perdida_acum += h['apuesta']
        else:
            break
    # necesito: pago * B >= perdida_acum + base  =>  B = ceil((perdida_acum+base)/pago)
    import math
    return math.ceil((perdida_acum + base) / pago)

def apostar_dalembert(historial, base, pago):
    """
    Sube 1 unidad base por perdida, baja 1 por victoria.
    Independiente del pago (es la definicion original).
    """
    apuesta = base
    for h in historial:
        apuesta = apuesta + base if not h['gano'] else max(base, apuesta - base)
    return apuesta

def apostar_fibonacci(historial, base, pago):
    """
    Secuencia Fibonacci escalada por base.
    Avanza 1 lugar por perdida, retrocede 2 por victoria.
    Lo que hace es al perder multiplicar la base por el correspondiente nro de la sucesion fibbonacci. Si ganas retrocedes 2 lugares en la sucesion 
    """
    fib = [1, 1]
    while fib[-1] < 100000:
        fib.append(fib[-1] + fib[-2])
    fib = [f * base for f in fib]
    idx = 0
    for h in historial:
        idx = min(idx + 1, len(fib)-1) if not h['gano'] else max(0, idx - 2)
    return fib[idx]

def apostar_paroli(historial, base, pago):
    """
    Anti-Martingala: dobla tras victoria (max 3 consecutivas),
    reinicia tras derrota o al llegar a 3 victorias seguidas.
    """
    consec = 0
    for h in reversed(historial):
        if h['gano'] and consec < 3:
            consec += 1
        else:
            break
    if consec == 0 or consec >= 3:
        return base
    return base * (2 ** consec)

ESTRATEGIAS = {
    'm': ('Martingala', apostar_martingala),
    'd': ("D'Alembert", apostar_dalembert),
    'f': ('Fibonacci',  apostar_fibonacci),
    'o': ('Paroli',     apostar_paroli),
}

# ═══════════════════════════════════════════════════════
# SIMULACION
# ═══════════════════════════════════════════════════════

def simular(capital_ini, n_tiradas, base, fn_strat,
            tipo, eleccion, capital_infinito=False):
    pago_neto   = TIPOS_APUESTA[tipo]['pago']
    capital     = capital_ini
    historial   = []          # lista de dicts {gano, apuesta}
    flujo       = [capital]
    freq_acum   = []
    wins        = 0
    bancarrotas = 0

    for i in range(n_tiradas):
        apuesta = fn_strat(historial, base, pago_neto)
        apuesta = max(base, int(apuesta))   # minimo = base, siempre entero

        if not capital_infinito:
            if capital <= 0:
                bancarrotas += 1
                # En la vida real la sesion termina si te quedas sin capital;
                # registrar el estado y salir del bucle en lugar de reiniciar.
                break
            apuesta = min(apuesta, capital)

        numero = random.choice(NUMEROS)
        gano, pago = resolver_apuesta(numero, tipo, eleccion)

        capital   += pago * apuesta if gano else -apuesta
        wins      += int(gano)
        historial.append({'gano': gano, 'apuesta': apuesta})
        flujo.append(capital)
        freq_acum.append(wins / (i + 1))

    return flujo, freq_acum, capital - capital_ini, bancarrotas

# ═══════════════════════════════════════════════════════
# GRAFICAS
# ═══════════════════════════════════════════════════════

def safename(s):
    for ch in (' ', "'", '/', '-'):
        s = s.replace(ch, '_')
    return s

def _suptitle(strat, apuesta_nombre, eleccion_str, infinito):
    cap = "capital infinito" if infinito else "capital finito"
    return f"{strat}  |  {apuesta_nombre} ({eleccion_str})  |  {cap}"

def graficar_corrida(flujo, freq, prob_teo, strat, apuesta_nombre,
                     eleccion_str, cap_ini, infinito, outdir):
    n = len(flujo) - 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(_suptitle(strat, apuesta_nombre, eleccion_str, infinito),
                 fontsize=11, fontweight='bold')

    axes[0].bar(range(1, n+1), freq, color='crimson', alpha=0.7, width=0.9)
    axes[0].axhline(y=prob_teo, color='navy', linestyle='--', linewidth=1.3,
                    label=f'Prob. teorica = {prob_teo:.4f}')
    axes[0].set_xlabel("n (numero de tiradas)")
    axes[0].set_ylabel("frsa")
    axes[0].set_title("Frecuencia relativa acumulada de victorias")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim([0, n+1])
    axes[0].set_ylim([0, min(1.05, prob_teo * 3.5)])

    axes[1].plot(range(n+1), flujo, color='crimson', linewidth=1.0,
                 label='fc (flujo de caja)')
    axes[1].axhline(y=cap_ini, color='steelblue', linestyle='--',
                    linewidth=1.3, label=f'fci = {cap_ini}')
    axes[1].set_xlabel("n (numero de tiradas)")
    axes[1].set_ylabel("cc (cantidad de capital)")
    axes[1].set_title("Flujo de caja")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(outdir,
        f"{safename(strat)}{safename(apuesta_nombre)}{'inf' if infinito else 'fin'}.png")
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname

def graficar_multiples(corridas, prob_teo, strat, apuesta_nombre,
                       eleccion_str, cap_ini, infinito, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(_suptitle(strat, apuesta_nombre, eleccion_str, infinito)
                 + "  |  Multiples corridas", fontsize=10, fontweight='bold')

    colores = plt.cm.tab10(np.linspace(0, 1, len(corridas)))
    for i, (flujo, freq, _, _) in enumerate(corridas):
        n = len(flujo) - 1
        axes[0].plot(range(1, n+1), freq, color=colores[i], alpha=0.7,
                     linewidth=0.9, label=f"C{i+1}")
        axes[1].plot(range(n+1), flujo, color=colores[i], alpha=0.7,
                     linewidth=0.9, label=f"C{i+1}")

    axes[0].axhline(y=prob_teo, color='black', linestyle='--', linewidth=1.4,
                    label='Prob. teorica')
    axes[0].set_xlabel("n"); axes[0].set_ylabel("frsa")
    axes[0].set_title("Frecuencia relativa acumulada")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].axhline(y=cap_ini, color='black', linestyle='--', linewidth=1.4,
                    label='Capital inicial')
    axes[1].set_xlabel("n"); axes[1].set_ylabel("cc")
    axes[1].set_title("Flujo de caja - multiples corridas")
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fname = os.path.join(outdir,
        f"{safename(strat)}{safename(apuesta_nombre)}{'inf' if infinito else 'fin'}_multi.png")
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname

# ═══════════════════════════════════════════════════════
# VALIDACION DE ELECCION
# ═══════════════════════════════════════════════════════

DEFAULTS_ELECCION = {
    'c': 'r', 'pi': 'p', 'mi': 'f',
    'd': '1', 'col': '1',
    'se': '1', 'cu': '1', 'tr': '1',
    'ca': '1-2', 'n': '0',
}

def validar_eleccion(tipo, eleccion):
    """Lanza SystemExit con mensaje claro si la eleccion es invalida."""
    if tipo == 'c'   and eleccion not in ('r','k'):
        sys.exit("[ERROR] -t c requiere -e r (rojo) o -e k (negro)")
    if tipo in ('d','col') and eleccion not in ('1','2','3'):
        sys.exit(f"[ERROR] -t {tipo} requiere -e 1, 2 o 3")
    if tipo == 'n':
        try:
            assert 0 <= int(eleccion) <= 36
        except Exception:
            sys.exit("[ERROR] -t n requiere -e 0..36")

def eleccion_str_legible(tipo, eleccion):
    if tipo == 'c':   return 'Rojo' if eleccion=='r' else 'Negro'
    if tipo == 'pi':  return 'Par'  if eleccion=='p' else 'Impar'
    if tipo == 'mi':  return 'Falta (1-18)' if eleccion=='f' else 'Pasa (19-36)'
    if tipo == 'd':   etq={'1':'1-12','2':'13-24','3':'25-36'}; return etq[eleccion]
    if tipo == 'col': etq={'1':'Col 1','2':'Col 2','3':'Col 3'}; return etq[eleccion]
    if tipo == 'se':  inicio=int(eleccion); return f"{inicio}-{inicio+5}"
    if tipo == 'cu':  n=int(eleccion); return f"{n},{n+1},{n+3},{n+4}"
    if tipo == 'tr':  inicio=int(eleccion); return f"{inicio}-{inicio+2}"
    if tipo == 'ca':  return f"Caballo {eleccion}"
    if tipo == 'n':   return f"Numero {eleccion}"

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Simulador de apuestas en ruleta europea",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c',       type=float, default=1000.0, help='Capital inicial')
    parser.add_argument('-n',       type=int,   default=500,    help='Numero de tiradas')
    parser.add_argument('-s',       type=str,   default='m',    help='m|d|f|o')
    parser.add_argument('-a',       type=str,   default='f',    help='f(inito)|i(nfinito)')
    parser.add_argument('-t',       type=str,   default='c',
                        help='Tipo: c|pi|mi|d|col|se|cu|tr|ca|n')
    parser.add_argument('-e',       type=str,   default=None,
                        help='Eleccion segun tipo de apuesta (ver docstring)')
    parser.add_argument('--base',   type=float, default=10.0,  help='Apuesta base')
    parser.add_argument('--runs',   type=int,   default=10,    help='Corridas multiples')
    parser.add_argument('--outdir', type=str,   default='./graficas')
    args = parser.parse_args()

    tipo = args.t.lower()
    if tipo not in TIPOS_APUESTA:
        sys.exit(f"[ERROR] Tipo '{tipo}' invalido. Opciones: {list(TIPOS_APUESTA)}")

    eleccion = args.e if args.e else DEFAULTS_ELECCION[tipo]
    if tipo == 'c': eleccion = eleccion.lower()

    validar_eleccion(tipo, eleccion)

    info         = TIPOS_APUESTA[tipo]
    prob_teorica = info['prob']
    estr_label   = eleccion_str_legible(tipo, eleccion)
    nombre_strat, fn_strat = ESTRATEGIAS.get(args.s.lower(), ESTRATEGIAS['m'])
    capital_infinito = (args.a.lower() == 'i')

    os.makedirs(args.outdir, exist_ok=True)

    # ── Valor esperado por unidad apostada ──
    ve = info['pago'] * info['prob'] - (1 - info['prob'])

    print(f"\n{'='*62}")
    print(f" Estrategia       : {nombre_strat}")
    print(f" Tipo de apuesta  : {info['nombre']}  ->  {estr_label}")
    print(f" Numeros cubiertos: {info['cubre']} de 37")
    print(f" Pago             : {info['pago']}:1")
    print(f" Prob. teorica    : {prob_teorica:.4f}  ({info['cubre']}/37)")
    print(f" Valor esperado   : {ve:+.4f}  por unidad apostada")
    print(f" Capital          : {'inf (infinito)' if capital_infinito else f'{args.c:.0f} (finito)'}")
    print(f" Tiradas          : {args.n}  |  Apuesta base: {args.base}")
    print(f"{'='*62}\n")

    # Remove fixed RNG seed to make simulations non-deterministic
    flujo, freq, ganancia, bankr = simular(
        args.c, args.n, args.base, fn_strat, tipo, eleccion, capital_infinito)

    print(f" Capital final        : {args.c + ganancia:.2f}")
    print(f" Ganancia neta        : {ganancia:+.2f}")
    print(f" Frec. victoria final : {freq[-1]:.4f}  (teorica = {prob_teorica:.4f})")
    if not capital_infinito:
        print(f" Bancarrotas          : {bankr}")
    print()

    img1 = graficar_corrida(flujo, freq, prob_teorica, nombre_strat,
                             info['nombre'], estr_label,
                             args.c, capital_infinito, args.outdir)
    print(f" Grafica corrida unica  : {img1}")

    random.seed(None)
    corridas = [simular(args.c, args.n, args.base, fn_strat,
                        tipo, eleccion, capital_infinito)
                for _ in range(args.runs)]

    img2 = graficar_multiples(corridas, prob_teorica, nombre_strat,
                               info['nombre'], estr_label,
                               args.c, capital_infinito, args.outdir)
    print(f" Grafica multiples      : {img2}")
    print(f"\n Imagenes guardadas en: {args.outdir}/\n")

if __name__ == '__main__':
    main()