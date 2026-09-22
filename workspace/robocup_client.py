"""
Módulo de Comunicación y Cliente para RoboCup 2D Soccer Simulator (rcssserver).
Permite conectar un agente en Python mediante sockets UDP, parsear la percepción en tiempo real,
enviar comandos de acción (dash, turn, kick, move) y renderizar el campo de juego en 2D.
"""

import socket
import select
import re
import math
import time
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class RoboCup2DClient:
    """Cliente UDP robusto y no bloqueante para interactuar con rcssserver en entornos de RL."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6000, team_name: str = "UTEC_RL"):
        self.host = host
        self.port = port
        self.team_name = team_name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.server_assigned_addr = (self.host, self.port)
        self.side: Optional[str] = None
        self.uniform_number: Optional[int] = None
        self.play_mode: str = "before_kick_off"
        self.is_connected = False
        self.last_observation: Dict[str, Any] = {
            "ball": None,          # (dist, dir)
            "goal_opp": None,      # (dist, dir)
            "goal_own": None,      # (dist, dir)
            "stamina": 8000.0,
            "speed": 0.0,
            "time": 0
        }

    def connect(self, init_pos: Tuple[float, float] = (-10.0, 0.0), version: int = 15) -> bool:
        """
        Envía el handshake de inicialización al servidor, consume los paquetes
        de parámetros iniciales y posiciona al jugador en el campo.
        """
        init_cmd = f"(init {self.team_name} (version {version}))\x00"
        try:
            self.sock.sendto(init_cmd.encode("ascii"), (self.host, self.port))
            
            # Esperar respuesta de init con select
            ready = select.select([self.sock], [], [], 3.0)
            if not ready[0]:
                print(f"Timeout al conectar con rcssserver en {self.host}:{self.port}")
                return False

            data, addr = self.sock.recvfrom(8192)
            self.server_assigned_addr = addr  # El servidor responde desde un puerto dinámico
            response = data.decode("latin1").strip()
            
            # Parsear respuesta: (init <side> <unum> <play_mode>)
            match = re.match(r"\(init\s+([lr])\s+(\d+)\s+([^\)]+)\)", response)
            if match:
                self.side = match.group(1)
                self.uniform_number = int(match.group(2))
                self.play_mode = match.group(3)
                self.is_connected = True
                
                # Consumir paquetes de configuración inicial (server_param, player_param, player_type)
                self._flush_initial_parameters()
                
                # Posicionar al jugador en el campo
                self.move(init_pos[0], init_pos[1])
                time.sleep(0.05)
                
                print(f"Conectado a RoboCup 2D | Equipo: {self.team_name} | Lado: {self.side} | Dorsal: {self.uniform_number} | Modo: {self.play_mode}")
                return True
            else:
                print(f"Respuesta de inicialización no esperada: {response}")
                return False
        except Exception as e:
            print(f"Error al conectar: {e}")
            return False

    def _flush_initial_parameters(self):
        """Descarta los mensajes de configuración inicial para dejar el socket listo para mensajes (see)."""
        time.sleep(0.1)
        while True:
            ready = select.select([self.sock], [], [], 0.05)
            if ready[0]:
                try:
                    data, _ = self.sock.recvfrom(8192)
                    txt = data.decode("latin1")
                    if "(see" in txt:
                        self._parse_and_update(txt)
                        break
                except Exception:
                    break
            else:
                break

    def send_command(self, cmd_str: str):
        """Envía un comando al servidor (terminado en \\0)."""
        if not self.is_connected:
            return
        payload = f"{cmd_str}\x00".encode("ascii")
        try:
            self.sock.sendto(payload, self.server_assigned_addr)
        except Exception:
            pass

    def move(self, x: float, y: float):
        """Ubica al jugador en coordenadas (x, y) relativas al campo (válido en before_kick_off)."""
        self.send_command(f"(move {x:.1f} {y:.1f})")

    def dash(self, power: float, direction: float = 0.0):
        """Acelera hacia adelante/atrás (-100 a 100) y opcionalmente con dirección (-180 a 180)."""
        if direction == 0.0:
            self.send_command(f"(dash {power:.1f})")
        else:
            self.send_command(f"(dash {power:.1f} {direction:.1f})")

    def turn(self, moment: float):
        """Gira el cuerpo del jugador (en grados, de -180 a 180)."""
        self.send_command(f"(turn {moment:.1f})")

    def kick(self, power: float, direction: float):
        """Patea el balón si está dentro del radio de pateo (distancia < 0.8m)."""
        self.send_command(f"(kick {power:.1f} {direction:.1f})")

    def get_latest_observation(self) -> Dict[str, Any]:
        """
        Drena todos los paquetes disponibles en el buffer UDP en modo no bloqueante
        y actualiza la última observación visual recibida.
        """
        while True:
            ready = select.select([self.sock], [], [], 0.0)
            if not ready[0]:
                break
            try:
                data, _ = self.sock.recvfrom(8192)
                txt = data.decode("latin1")
                if "(see" in txt:
                    self._parse_and_update(txt)
                elif "(sense_body" in txt:
                    self._parse_sense_body(txt)
            except Exception:
                break
        return self.last_observation

    def _parse_and_update(self, msg: str):
        """Extrae la información visual de mensajes (see ...)."""
        t_match = re.search(r"\(see\s+(\d+)", msg)
        if t_match:
            self.last_observation["time"] = int(t_match.group(1))

        # Buscar balón: ((b) dist dir ...)
        ball_match = re.search(r"\(\(b\)\s+([\-\d\.]+)\s+([\-\d\.]+)", msg)
        if ball_match:
            self.last_observation["ball"] = (float(ball_match.group(1)), float(ball_match.group(2)))

        # Buscar portería rival y propia
        opp_side = "r" if self.side == "l" else "l"
        opp_goal_match = re.search(rf"\(\(g\s+{opp_side}\)\s+([\-\d\.]+)\s+([\-\d\.]+)", msg)
        if opp_goal_match:
            self.last_observation["goal_opp"] = (float(opp_goal_match.group(1)), float(opp_goal_match.group(2)))

        own_goal_match = re.search(rf"\(\(g\s+{self.side}\)\s+([\-\d\.]+)\s+([\-\d\.]+)", msg)
        if own_goal_match:
            self.last_observation["goal_own"] = (float(own_goal_match.group(1)), float(own_goal_match.group(2)))

    def _parse_sense_body(self, msg: str):
        """Extrae información interna del cuerpo (stamina, velocidad)."""
        stamina_match = re.search(r"\(stamina\s+([\-\d\.]+)", msg)
        if stamina_match:
            self.last_observation["stamina"] = float(stamina_match.group(1))
        speed_match = re.search(r"\(speed\s+([\-\d\.]+)", msg)
        if speed_match:
            self.last_observation["speed"] = float(speed_match.group(1))

    def render_field(self, title: str = "RoboCup 2D — Visualización del Terreno de Juego"):
        """
        Dibuja el campo de fútbol en 2D (dimensiones oficiales 105m x 68m)
        y la posición relativa del jugador, el balón y la portería.
        """
        fig, ax = plt.subplots(figsize=(10, 6.2))
        
        # Color césped
        ax.set_facecolor("#2e7d32")
        
        # Líneas del campo (105 x 68 metros, centrado en 0,0)
        pitch_border = patches.Rectangle((-52.5, -34), 105, 68, linewidth=2, edgecolor="white", facecolor="none")
        half_line = plt.Line2D([0, 0], [-34, 34], color="white", linewidth=2)
        center_circle = patches.Circle((0, 0), 9.15, linewidth=2, edgecolor="white", facecolor="none")
        center_spot = patches.Circle((0, 0), 0.5, color="white")
        
        # Áreas grandes y porterías
        left_box = patches.Rectangle((-52.5, -20.16), 16.5, 40.32, linewidth=1.5, edgecolor="white", facecolor="none")
        right_box = patches.Rectangle((36.0, -20.16), 16.5, 40.32, linewidth=1.5, edgecolor="white", facecolor="none")
        left_goal = patches.Rectangle((-54.5, -7.0), 2.0, 14.0, linewidth=2, edgecolor="yellow", facecolor="#ffffff", alpha=0.3)
        right_goal = patches.Rectangle((52.5, -7.0), 2.0, 14.0, linewidth=2, edgecolor="yellow", facecolor="#ffffff", alpha=0.3)
        
        ax.add_patch(pitch_border)
        ax.add_line(half_line)
        ax.add_patch(center_circle)
        ax.add_patch(center_spot)
        ax.add_patch(left_box)
        ax.add_patch(right_box)
        ax.add_patch(left_goal)
        ax.add_patch(right_goal)
        
        # Posición estimada del jugador
        p_x, p_y = -10.0, 0.0
        ax.plot(p_x, p_y, "o", markersize=13, color="#1976d2", label=f"Jugador (Lado {self.side or 'L'}, Dorsal {self.uniform_number or 1})")
        
        # Posición estimada del balón
        obs = self.last_observation
        if obs["ball"]:
            dist, angle_deg = obs["ball"]
            angle_rad = math.radians(angle_deg)
            b_x = p_x + dist * math.cos(angle_rad)
            b_y = p_y + dist * math.sin(angle_rad)
            ax.plot(b_x, b_y, "o", markersize=10, color="#ffffff", markeredgecolor="#000000", markeredgewidth=1.8, label=f"Balón (Distancia: {dist:.1f}m)")
            ax.plot([p_x, b_x], [p_y, b_y], "--", color="yellow", alpha=0.8, label="Vector de Visión")
        else:
            ax.plot(0, 0, "o", markersize=10, color="#ffffff", markeredgecolor="#000000", label="Balón (Centro)")
            
        ax.set_xlim(-58, 58)
        ax.set_ylim(-38, 38)
        ax.set_xlabel("Eje X (metros)", fontsize=11, color="white")
        ax.set_ylabel("Eje Y (metros)", fontsize=11, color="white")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def close(self):
        """Cierra el socket de comunicación."""
        if self.is_connected:
            try:
                self.send_command("(bye)")
            except Exception:
                pass
            self.sock.close()
            self.is_connected = False
