"""
Engine flood simulation com NumPy — versão D8 com flow directional.

This version prioritizes flow directional (D8/LDD) a partir do DEM, com
fallback difusivo para permitir transbordamento/espalhamento quando o
preferred path not available.

Melhorias sobre o modelo clássico de difusão-onda:
- Fluxo directional D8 (steepest descent) para rotear água de forma mais realista
- Intensidade espacial de chuva (possibilidade de não-uniforme)
- Fallback difusivo controlado para áreas de baixa declividade/planos

Author: Letícia Caldas
License: MIT
"""

import numpy as np
from typing import Optional, Set, Tuple, Dict, Any, List


class GamaFloodModelNumpy:
    """Engine flood simulation com NumPy (versão D8 com flow directional).

    This version prioritizes flow directional (D8/LDD) a partir do DEM, com
    fallback difusivo para permitir transbordamento/espalhamento quando o
    preferred path not available.

    Parâmetros
    ----------
    dem_data : np.ndarray
        Matriz 2D com altitudes (m a.s.l.)
    sources_mask : np.ndarray
        Máscara 2D (uint8/bool) com sources de chuva
    diffusion_rate : float
        Fração de água que pode se mover por passo (0 < rate ≤ 1)
    flood_threshold : float
        Limiar de inundação (m) para classificação
    cell_size_meters : float
        Tamanho da célula (m)
    river_mask : np.ndarray, optional
        Máscara opcional de rio (não obrigatório)
    sources_intensity : np.ndarray, optional
        Mapa de intensity espacial (float32) com valores relativos
    intensity_mode : str
        "relative" (padrão) normaliza intensity; "absolute" usa valores diretos
    """

    def __init__(
        self,
        dem_data: np.ndarray,
        sources_mask: np.ndarray,
        diffusion_rate: float,
        flood_threshold: float,
        cell_size_meters: float,
        river_mask: Optional[np.ndarray] = None,
        sources_intensity: Optional[np.ndarray] = None,
        intensity_mode: str = "relative",
    ) -> None:
        """Inicializa o modelo de inundação."""
        self.height, self.width = dem_data.shape
        self.diffusion_rate = float(diffusion_rate)
        self.flood_threshold = float(flood_threshold)
        self.cell_area = float(cell_size_meters) * float(cell_size_meters)

        # Converter DEM para float32, preenchendo NaN com mediana
        self.altitude = dem_data.astype(np.float32)
        if not np.isfinite(self.altitude).all():
            _median = float(
                np.nanmedian(self.altitude)
                if np.isfinite(self.altitude).any()
                else 0.0
            )
            self.altitude = np.where(
                np.isfinite(self.altitude), self.altitude, _median
            )

        # Máscaras
        self.is_source = (
            sources_mask.astype(bool)
            if sources_mask is not None
            else np.zeros_like(self.altitude, dtype=bool)
        )
        self.river_mask = (
            river_mask.astype(bool)
            if river_mask is not None
            else np.zeros_like(self.altitude, dtype=bool)
        )

        # Mapa de intensity espacial (float) — valores relativos
        # (qualquer escala). Normalizamos internamente quando aplicável.
        if sources_intensity is not None:
            try:
                si = np.asarray(sources_intensity, dtype=np.float32)
                # Garantir shape compatível
                if si.shape == self.altitude.shape:
                    self.sources_intensity = si
                else:
                    self.sources_intensity = None
            except Exception:
                self.sources_intensity = None
        else:
            self.sources_intensity = None

        self.intensity_mode = str(intensity_mode or "relative").lower()

        # Água
        self.water_height = np.zeros_like(self.altitude, dtype=np.float32)

        # Direção de fluxo (D8) calculada a partir do DEM
        # (com pequeno viés para rivers, se houver).
        dem_eff = self.altitude.copy()
        try:
            if np.any(self.river_mask):
                river_bias_m = float(
                    min(0.5, max(0.05, self.flood_threshold * 0.25))
                )
                dem_eff = (
                    self.altitude
                    - (self.river_mask.astype(np.float32) * river_bias_m)
                ).astype(np.float32)
        except Exception:
            dem_eff = self.altitude

        self._flow_dy, self._flow_dx = self._calculate_d8_flow_directions(
            dem_eff
        )

        # Rastreamento
        self.active_cells_coords: Set[Tuple[int, int]] = set(
            zip(*np.where(self.is_source))
        )
        self.simulation_time_minutes: int = 0
        self.overflow_time_minutes: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self.uniform_rain: bool = True

    @staticmethod
    def _calculate_d8_flow_directions(
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula direção de fluxo D8 (steepest descent) como deslocamentos dy/dx.

        Retorna (dy, dx) int8 do mesmo shape de dem.
        Células sem saída válida (pits/planos/nodata) ficam com (0,0).

        Parameters
        ----------
        dem : np.ndarray
            Matriz 2D de elevação

        Returns
        -------
        dy, dx : np.ndarray
            Deslocamentos inteiros Y e X para cada célula
        """
        dem_f = np.asarray(dem, dtype=np.float32)
        H, W = dem_f.shape
        if H == 0 or W == 0:
            return (
                np.zeros((H, W), dtype=np.int8),
                np.zeros((H, W), dtype=np.int8),
            )

        # Pad DEM com edge para evitar indexação fora de limites
        pad = np.pad(dem_f, pad_width=1, mode="edge")
        center = pad[1:-1, 1:-1]

        sqrt2 = np.float32(np.sqrt(2.0))
        dirs = [
            (-1, -1, sqrt2),
            (-1, 0, np.float32(1.0)),
            (-1, 1, sqrt2),
            (0, -1, np.float32(1.0)),
            (0, 1, np.float32(1.0)),
            (1, -1, sqrt2),
            (1, 0, np.float32(1.0)),
            (1, 1, sqrt2),
        ]

        slopes: List[np.ndarray] = []
        for dy, dx, dist in dirs:
            neigh = pad[(1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W)]
            diff = center - neigh
            s = diff / dist
            # Não permitir fluxo para cima/planos ou para nodata
            s = np.where(np.isfinite(s) & (s > 0), s, -np.inf)
            slopes.append(s)

        stack = np.stack(slopes, axis=0)  # (8, H, W)
        idx = np.argmax(stack, axis=0).astype(np.int8)
        max_s = np.take_along_axis(
            stack, idx[None, :, :].astype(np.int64), axis=0
        )[0]
        valid = np.isfinite(max_s)

        dy_list = np.array([d[0] for d in dirs], dtype=np.int8)
        dx_list = np.array([d[1] for d in dirs], dtype=np.int8)
        out_dy = dy_list[idx]
        out_dx = dx_list[idx]
        out_dy[~valid] = 0
        out_dx[~valid] = 0
        return out_dy, out_dx

    def _diffuse_overflow_step(
        self,
        prev: np.ndarray,
        y: int,
        x: int,
        move_amount: float,
        newly_active: Set[Tuple[int, int]],
        to_deactivate: Set[Tuple[int, int]],
    ) -> None:
        """Fallback difusivo local (8 vizinhos) para permitir transbordamento.

        Parameters
        ----------
        prev : np.ndarray
            Água no passo anterior
        y, x : int
            Coordenadas da célula
        move_amount : float
            Quantidade a mover
        newly_active : set
            Conjunto de células ativadas
        to_deactivate : set
            Conjunto de células a desativar
        """
        H, W = self.height, self.width
        cur_w = float(prev[y, x])
        if cur_w <= 1e-6 or move_amount <= 1e-6:
            to_deactivate.add((y, x))
            return

        cur_total = float(self.altitude[y, x]) + cur_w
        neigh = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neigh.append((ny, nx))

        if not neigh:
            to_deactivate.add((y, x))
            return

        ny, nx = zip(*neigh)
        n_total = self.altitude[ny, nx] + prev[ny, nx]
        mask_lower = np.isfinite(n_total) & (n_total < cur_total)

        if not np.any(mask_lower):
            to_deactivate.add((y, x))
            return

        lower_coords = np.array(neigh)[mask_lower]
        lower_total = n_total[mask_lower]
        total_diff = float(np.sum(cur_total - lower_total))

        if total_diff <= 0:
            to_deactivate.add((y, x))
            return

        # Distribuir proporcionalmente à diferença de energia;
        # limita para estabilidade
        for i, (ny2, nx2) in enumerate(lower_coords):
            diff = cur_total - float(lower_total[i])
            frac = float(diff) / total_diff
            wmv = min(move_amount * frac, diff / 2.0)
            if wmv > 1e-6:
                self.water_height[y, x] -= wmv
                self.water_height[ny2, nx2] += wmv
                newly_active.add((ny2, nx2))

    def add_water(self, rain_mm: float) -> None:
        """Adiciona água via chuva (uniforme ou espacialmente distribuída).

        Parameters
        ----------
        rain_mm : float
            Profundidade de chuva (mm)
        """
        water_to_add_meters = float(rain_mm) / 1000.0
        if water_to_add_meters <= 0:
            return

        # Chuva uniforme: aplica em toda a grade
        if self.uniform_rain:
            self.water_height += water_to_add_meters
            # Ativar todas as células que agora têm água
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))
            return

        # Caso não seja uniforme: se houver mapa de intensity espacial, usá-lo
        # preferencialmente
        if self.sources_intensity is not None and np.any(
            self.sources_intensity > 0
        ):
            try:
                si = self.sources_intensity
                if self.intensity_mode == "absolute":
                    # si contém mm por ciclo; converter para metros e aplicar
                    # diretamente
                    self.water_height += si.astype(np.float32) / 1000.0
                    ys2, xs2 = np.where(si > 0)
                    self.active_cells_coords.update(zip(ys2, xs2))
                    return
                else:
                    maxv = (
                        float(np.nanmax(si))
                        if np.isfinite(np.nanmax(si))
                        else 0.0
                    )
                    if maxv > 0.0:
                        norm_si = si / maxv
                        self.water_height += water_to_add_meters * norm_si
                        ys2, xs2 = np.where(norm_si > 0)
                        self.active_cells_coords.update(zip(ys2, xs2))
                        return
            except Exception:
                pass

        # Se não houver mapa de intensity, usar máscara de sources/rivers/fallback
        if np.any(self.is_source):
            self.water_height[self.is_source] += water_to_add_meters
            ys, xs = np.where(self.is_source)
            self.active_cells_coords.update(zip(ys, xs))
        elif np.any(self.river_mask):
            self.water_height[self.river_mask] += water_to_add_meters * 0.2
            ys, xs = np.where(self.river_mask)
            self.active_cells_coords.update(zip(ys, xs))
        else:
            # Fallback: sem sources nem rio definidos, distribuir uniformemente
            self.water_height += water_to_add_meters
            ys, xs = np.where(self.water_height > 0)
            self.active_cells_coords.update(zip(ys, xs))

    def run_flow_step(self) -> None:
        """Executa um passo de flow com fluxo D8 + fallback difusivo.

        Prioriza:
        1. Escoamento directional D8 (steepest descent)
        2. Fallback difusivo para transbordamento/planos
        """
        if not self.active_cells_coords:
            return

        newly_active: Set[Tuple[int, int]] = set()
        to_deactivate: Set[Tuple[int, int]] = set()
        prev = self.water_height.copy()
        H, W = self.height, self.width

        for y, x in list(self.active_cells_coords):
            cur_w = prev[y, x]
            if cur_w <= 1e-3:
                to_deactivate.add((y, x))
                continue

            # (1) Escoamento directional (D8) preferencial
            dy = int(self._flow_dy[y, x])
            dx = int(self._flow_dx[y, x])
            move_amount = float(cur_w) * float(self.diffusion_rate)

            routed = False
            if dy != 0 or dx != 0:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    cur_total = float(self.altitude[y, x]) + float(cur_w)
                    dn_total = (
                        float(self.altitude[ny, nx]) + float(prev[ny, nx])
                    )
                    if (
                        np.isfinite(dn_total)
                        and (dn_total < cur_total)
                        and (move_amount > 1e-6)
                    ):
                        diff = cur_total - dn_total
                        wmv = min(move_amount, diff / 2.0)
                        if wmv > 1e-6:
                            self.water_height[y, x] -= wmv
                            self.water_height[ny, nx] += wmv
                            newly_active.add((ny, nx))
                            # Manter origem ativa se ainda tem água
                            if (float(cur_w) - wmv) > 1e-3:
                                newly_active.add((y, x))
                            routed = True

            # (2) Fallback difusivo quando não há rota válida
            # (transbordamento/planos)
            if not routed:
                # Keeps preferência directional; espalha só parte
                overflow_amount = move_amount * 0.35
                self._diffuse_overflow_step(
                    prev, y, x, overflow_amount, newly_active, to_deactivate
                )

        self.active_cells_coords.difference_update(to_deactivate)
        self.active_cells_coords.update(newly_active)

    def update_stats(self, time_step_minutes: int) -> None:
        """Atualiza estatísticas da simulação.

        Parameters
        ----------
        time_step_minutes : int
            Duração do passo de tempo (minutos)
        """
        self.simulation_time_minutes += int(time_step_minutes)
        inundated = self.water_height > self.flood_threshold

        if (
            self.overflow_time_minutes is None
            and np.any(inundated & ~self.is_source)
        ):
            self.overflow_time_minutes = self.simulation_time_minutes

        self.history.append(
            {
                "time_minutes": self.simulation_time_minutes,
                "flooded_percent": (
                    float(np.sum(inundated)) / float(inundated.size) * 100.0
                ),
                "active_cells": int(len(self.active_cells_coords)),
                "max_depth": (
                    float(np.max(self.water_height))
                    if self.water_height.size > 0
                    else 0.0
                ),
                "total_water_volume_m3": float(
                    np.sum(self.water_height * self.cell_area)
                ),
                "water_height_snapshot": self.water_height.copy(),
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo da simulação."""
        return {
            "simulation_time_minutes": self.simulation_time_minutes,
            "max_water_depth": float(
                np.nanmax(self.water_height)
                if self.water_height.size > 0
                else 0.0
            ),
            "total_water_volume_m3": float(
                np.sum(self.water_height * self.cell_area)
            ),
            "flooded_area_percent": (
                float(np.sum(self.water_height > self.flood_threshold))
                / self.water_height.size
                * 100.0
            ),
            "overflow_time_minutes": self.overflow_time_minutes,
        }

    # Aliases para compatibilidade com API difusão-onda
    def apply_rainfall(self, rain_mm: float) -> None:
        """Alias para add_water()."""
        self.add_water(rain_mm)

    def advance_flow(self) -> None:
        """Alias para run_flow_step()."""
        self.run_flow_step()

    def record_diagnostics(self, time_step_minutes: int) -> None:
        """Alias para update_stats()."""
        self.update_stats(time_step_minutes)
