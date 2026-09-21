# ============================================================
# CARTILLA: Fundamentos de la Teoría de Juegos en IA
# Minimax y Poda Alpha-Beta aplicados a Tic-Tac-Toe
# ============================================================

import time

# ------------------------------------------------------------
# PARTE 1: Representación del tablero y funciones auxiliares
# ------------------------------------------------------------

def tablero_vacio():
    """Crea un tablero de 3x3 vacío, representado como una lista de 9 posiciones."""
    return [" " for _ in range(9)]

def imprimir_tablero(tablero):
    """Muestra el tablero en formato 3x3 legible en consola."""
    for i in range(0, 9, 3):
        fila = tablero[i:i+3]
        print(" | ".join(fila))
        if i < 6:
            print("-" * 9)
    print()

def movimientos_disponibles(tablero):
    """Retorna una lista con los índices de las casillas vacías."""
    return [i for i, casilla in enumerate(tablero) if casilla == " "]

def hay_ganador(tablero, jugador):
    """Verifica si 'jugador' (X u O) ha ganado la partida."""
    combinaciones_ganadoras = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # filas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columnas
        (0, 4, 8), (2, 4, 6)              # diagonales
    ]
    return any(
        all(tablero[pos] == jugador for pos in combinacion)
        for combinacion in combinaciones_ganadoras
    )

def tablero_lleno(tablero):
    """Retorna True si ya no quedan casillas vacías (empate potencial)."""
    return " " not in tablero

def juego_terminado(tablero):
    """Retorna True si el juego terminó, ya sea por victoria o por empate."""
    return hay_ganador(tablero, "X") or hay_ganador(tablero, "O") or tablero_lleno(tablero)

def evaluar(tablero):
    """
    Función de evaluación de un estado terminal:
    +1 si gana X (maximizador), -1 si gana O (minimizador), 0 en caso de empate.
    """
    if hay_ganador(tablero, "X"):
        return 1
    elif hay_ganador(tablero, "O"):
        return -1
    else:
        return 0


# ------------------------------------------------------------
# PARTE 2: Algoritmo Minimax puro (sin poda)
# ------------------------------------------------------------

nodos_explorados_minimax = 0

def minimax(tablero, profundidad, es_maximizador):
    """
    Implementación clásica de Minimax.
    Retorna el valor de utilidad del estado actual, asumiendo juego óptimo de ambos lados.
    """
    global nodos_explorados_minimax
    nodos_explorados_minimax += 1

    if juego_terminado(tablero):
        return evaluar(tablero)

    if es_maximizador:
        mejor_valor = float("-inf")
        for movimiento in movimientos_disponibles(tablero):
            tablero[movimiento] = "X"
            valor = minimax(tablero, profundidad + 1, False)
            tablero[movimiento] = " "
            mejor_valor = max(mejor_valor, valor)
        return mejor_valor
    else:
        mejor_valor = float("inf")
        for movimiento in movimientos_disponibles(tablero):
            tablero[movimiento] = "O"
            valor = minimax(tablero, profundidad + 1, True)
            tablero[movimiento] = " "
            mejor_valor = min(mejor_valor, valor)
        return mejor_valor

def mejor_movimiento_minimax(tablero):
    """Determina el mejor movimiento para X (maximizador) usando Minimax puro."""
    global nodos_explorados_minimax
    nodos_explorados_minimax = 0

    mejor_valor = float("-inf")
    mejor_mov = None

    for movimiento in movimientos_disponibles(tablero):
        tablero[movimiento] = "X"
        valor = minimax(tablero, 0, False)
        tablero[movimiento] = " "
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_mov = movimiento

    return mejor_mov, mejor_valor, nodos_explorados_minimax


# ------------------------------------------------------------
# PARTE 3: Minimax con poda Alpha-Beta
# ------------------------------------------------------------

nodos_explorados_alfabeta = 0

def minimax_alfabeta(tablero, profundidad, alfa, beta, es_maximizador):
    """
    Minimax optimizado con poda Alpha-Beta.
    Cuando beta <= alfa, se poda la rama restante porque ya no puede
    influir en la decisión final.
    """
    global nodos_explorados_alfabeta
    nodos_explorados_alfabeta += 1

    if juego_terminado(tablero):
        return evaluar(tablero)

    if es_maximizador:
        mejor_valor = float("-inf")
        for movimiento in movimientos_disponibles(tablero):
            tablero[movimiento] = "X"
            valor = minimax_alfabeta(tablero, profundidad + 1, alfa, beta, False)
            tablero[movimiento] = " "
            mejor_valor = max(mejor_valor, valor)
            alfa = max(alfa, mejor_valor)
            if beta <= alfa:
                break
        return mejor_valor
    else:
        mejor_valor = float("inf")
        for movimiento in movimientos_disponibles(tablero):
            tablero[movimiento] = "O"
            valor = minimax_alfabeta(tablero, profundidad + 1, alfa, beta, True)
            tablero[movimiento] = " "
            mejor_valor = min(mejor_valor, valor)
            beta = min(beta, mejor_valor)
            if beta <= alfa:
                break
        return mejor_valor

def mejor_movimiento_alfabeta(tablero):
    """Determina el mejor movimiento para X usando Minimax con poda Alpha-Beta."""
    global nodos_explorados_alfabeta
    nodos_explorados_alfabeta = 0

    mejor_valor = float("-inf")
    mejor_mov = None
    alfa = float("-inf")
    beta = float("inf")

    for movimiento in movimientos_disponibles(tablero):
        tablero[movimiento] = "X"
        valor = minimax_alfabeta(tablero, 0, alfa, beta, False)
        tablero[movimiento] = " "
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_mov = movimiento
        alfa = max(alfa, mejor_valor)

    return mejor_mov, mejor_valor, nodos_explorados_alfabeta


# ------------------------------------------------------------
# PARTE 4: Comparación de rendimiento entre ambos algoritmos
# ------------------------------------------------------------

def comparar_rendimiento(tablero):
    """
    Ejecuta ambos algoritmos sobre el mismo tablero y compara movimiento elegido,
    valor de utilidad, nodos explorados y tiempo de ejecución.
    """
    tablero_copia_1 = tablero.copy()
    inicio = time.time()
    mov_mm, valor_mm, nodos_mm = mejor_movimiento_minimax(tablero_copia_1)
    tiempo_mm = time.time() - inicio

    tablero_copia_2 = tablero.copy()
    inicio = time.time()
    mov_ab, valor_ab, nodos_ab = mejor_movimiento_alfabeta(tablero_copia_2)
    tiempo_ab = time.time() - inicio

    print("Tablero analizado:")
    imprimir_tablero(tablero)

    print(f"{'Métrica':<25}{'Minimax puro':<20}{'Minimax + Alpha-Beta':<20}")
    print("-" * 65)
    print(f"{'Movimiento elegido':<25}{mov_mm!s:<20}{mov_ab!s:<20}")
    print(f"{'Valor de utilidad':<25}{valor_mm!s:<20}{valor_ab!s:<20}")
    print(f"{'Nodos explorados':<25}{nodos_mm!s:<20}{nodos_ab!s:<20}")
    print(f"{'Tiempo (segundos)':<25}{tiempo_mm:.6f}{'':<12}{tiempo_ab:.6f}")

    if nodos_mm > 0:
        reduccion = (1 - nodos_ab / nodos_mm) * 100
        print(f"\nReducción de nodos explorados gracias a la poda Alpha-Beta: {reduccion:.2f}%")

    assert mov_mm == mov_ab, "¡Los algoritmos difieren en el movimiento elegido!"
    assert valor_mm == valor_ab, "¡Los algoritmos difieren en el valor de utilidad!"
    print("\nVerificación: ambos algoritmos coinciden en el resultado óptimo. ✔")


# ------------------------------------------------------------
# PARTE 5: Simulación de una partida completa
# ------------------------------------------------------------

def turno_humano(tablero):
    """Solicita al jugador humano (O) que elija una casilla disponible."""
    disponibles = movimientos_disponibles(tablero)
    movimiento = None
    while movimiento not in disponibles:
        try:
            movimiento = int(input(f"Elige tu casilla {disponibles}: "))
        except ValueError:
            print("Entrada inválida, intenta de nuevo.")
    tablero[movimiento] = "O"

def resultado_final(tablero):
    """Determina e imprime el resultado final de la partida."""
    if hay_ganador(tablero, "X"):
        print("¡Gana X (IA)! 🤖")
    elif hay_ganador(tablero, "O"):
        print("¡Gana O! 🎉")
    else:
        print("Empate. 🤝")

def jugar_ia_vs_ia():
    """
    Simula una partida completa entre dos IAs que usan Minimax con poda Alpha-Beta.
    Como ambas juegan de forma óptima, el resultado esperado es siempre un empate.
    """
    tablero = tablero_vacio()
    turno_x = True

    while not juego_terminado(tablero):
        if turno_x:
            movimiento, _, _ = mejor_movimiento_alfabeta(tablero)
            tablero[movimiento] = "X"
        else:
            mejor_valor = float("inf")
            mejor_mov = None
            for mov in movimientos_disponibles(tablero):
                tablero[mov] = "O"
                valor = minimax_alfabeta(tablero, 0, float("-inf"), float("inf"), True)
                tablero[mov] = " "
                if valor < mejor_valor:
                    mejor_valor = valor
                    mejor_mov = mov
            tablero[mejor_mov] = "O"

        imprimir_tablero(tablero)
        turno_x = not turno_x

    resultado_final(tablero)

def jugar_ia_vs_humano():
    """
    Simula una partida entre la IA (X, usando Alpha-Beta) y un jugador
    humano (O), que ingresa sus movimientos por consola.
    """
    tablero = tablero_vacio()
    turno_x = True

    print("Bienvenido al Tic-Tac-Toe contra la IA (Minimax + Alpha-Beta).")
    print("Las casillas se numeran del 0 al 8, de izquierda a derecha y de arriba hacia abajo.\n")
    imprimir_tablero(tablero)

    while not juego_terminado(tablero):
        if turno_x:
            print("Turno de la IA (X)...")
            movimiento, _, nodos = mejor_movimiento_alfabeta(tablero)
            tablero[movimiento] = "X"
            print(f"La IA jugó en la casilla {movimiento} (nodos explorados: {nodos})")
        else:
            turno_humano(tablero)

        imprimir_tablero(tablero)
        turno_x = not turno_x

    resultado_final(tablero)


# ------------------------------------------------------------
# PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":
    # 1) Compara nodos explorados y tiempo entre Minimax puro y Alpha-Beta
    tablero_prueba = tablero_vacio()
    comparar_rendimiento(tablero_prueba)

    print("\n" + "=" * 65 + "\n")

    # 2) Elige una de las dos simulaciones (descomenta la que quieras usar)
    # jugar_ia_vs_ia()
    jugar_ia_vs_humano()