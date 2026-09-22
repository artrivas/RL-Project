"""
Script de Prueba: Agente Aleatorio y Heurístico para RoboCup 2D
Demuestra la conexión con rcssserver, el ciclo de percepción y la toma de acciones.
"""

import os
import time
import random
from robocup_client import RoboCup2DClient

SERVER_HOST = os.getenv("SERVER_HOST", "rcssserver")
SERVER_PORT = int(os.getenv("SERVER_PORT", "6000"))

print("="*70)
print(f"Iniciando Agente de Prueba RoboCup 2D -> Servidor: {SERVER_HOST}:{SERVER_PORT}")
print("="*70)

client = RoboCup2DClient(host=SERVER_HOST, port=SERVER_PORT, team_name="UTEC_RL")

if not client.connect(init_pos=(-10.0, 0.0)):
    print("No se pudo conectar al servidor. Verifica que rcssserver esté corriendo.")
    exit(1)

print("\nEjecutando ciclo de control del agente durante 50 pasos...")
print("-" * 70)

try:
    for step in range(50):
        # 1. Obtener la observación más reciente
        obs = client.get_latest_observation()
        ball = obs["ball"]
        goal = obs["goal_opp"]

        # 2. Estrategia del Agente (Aleatoria / Reactiva Básica)
        if ball is not None:
            dist_ball, dir_ball = ball
            if dist_ball < 0.8:
                # Patear hacia la portería rival si está visible
                kick_dir = goal[1] if goal is not None else random.uniform(-30, 30)
                client.kick(power=100.0, direction=kick_dir)
                action_desc = f"KICK(power=100, dir={kick_dir:.1f}°)"
            elif abs(dir_ball) > 10:
                client.turn(moment=dir_ball)
                action_desc = f"TURN(dir={dir_ball:.1f}°)"
            else:
                client.dash(power=80.0)
                action_desc = f"DASH(power=80)"
        else:
            # Buscar el balón girando
            client.turn(moment=45.0)
            action_desc = "SEARCH_BALL (TURN 45°)"

        ball_str = f"Dist: {ball[0]:.1f}m, Dir: {ball[1]:.1f}°" if ball else "Buscando balón..."
        print(f"Paso {step:02d} | Balón: {ball_str:<25} | Acción: {action_desc}")

        time.sleep(0.08)  # Frecuencia de ciclo de simulación (~100ms)

except KeyboardInterrupt:
    print("\nDeteniendo agente...")

finally:
    client.close()
    print("\nAgente desconectado correctamente.")
    print("="*70)
