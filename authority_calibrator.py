"""
authority_calibrator.py -- Calibracion dinamica de autoridad para RiskAval.

Generalizacion directa de Cerebro._riesgo_kelly(), el motor de sizing que
corre HOY en produccion dentro de KMT (delta_std=1.0 y delta_std=1.5,
ambos live con capital real). La formula matematica es identica -- lo
unico que cambia es el vocabulario: "riesgo por trade en dolares" pasa a
ser "autoridad calibrada", y "trades" pasa a ser "outcome_history", para
que aplique a cualquier agente (no solo ordenes de trading).

QUE RESUELVE ESTO QUE policy_engine.py NO RESOLVIA ANTES:
Hasta ahora, Ejecutor.entrar() calculaba `qty` via _riesgo_kelly() ANTES
de llamar a PolicyEngine.evaluate(), pero nunca le pasaba la fraccion de
Kelly, el track record, ni el resultado del calculo -- solo el numero
final de contratos. El policy engine solo veia el resultado, nunca
participaba en como se calibro. Este modulo expone esa calibracion como
un objeto de primera clase que PolicyEngine SI puede consumir e incluir
en el contexto de evaluacion (ver integracion en policy_engine.py).

LA MATEMATICA (sin cambios respecto a _riesgo_kelly()):
  1. Con menos de `min_sample_size` resultados historicos, se usa
     `base_authority` sin ajustar -- no hay suficiente evidencia para
     calibrar con confianza.
  2. Con suficiente historial, se calcula Kelly sobre los ultimos
     `window` resultados: p = tasa de exito, b = ratio ganancia/perdida
     promedio, kelly = (p*b - (1-p)) / b.
  3. Kelly se reduce por `kelly_fraction` (half-Kelly = 0.5 por defecto,
     igual que KMT) -- Kelly completo es agresivo, fraccionado es lo que
     realmente corre en produccion.
  4. Si el Kelly ajustado es <= 0 (el track record reciente no sostiene
     ninguna autoridad), se cae a una fraccion de sondeo pequeña
     (`probe_fraction`) en vez de bloquear del todo -- permite seguir
     midiendo sin exponer autoridad significativa.
  5. Un techo (`ceiling_fraction` de la capacidad total) evita que un
     Kelly ajustado inusualmente alto autorice una fraccion irrazonable
     de la capacidad total de una sola vez.
  6. Un piso (`floor`) evita que la autoridad caiga a un nivel
     operativamente inutil.

Esto es matematicamente identico a Kelly criterion aplicado a position
sizing en trading -- la unica generalizacion es que "resultado positivo/
negativo" puede ser exito/fallo de cualquier accion de un agente, no
solo PnL de un trade.
"""
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------
# INTEGRACION con policy_engine.py: este modulo no depende de
# PolicyEngine (cero import circular), pero expone lo que
# PolicyEngine.evaluate() necesita para inyectar `calibrated_authority`
# en el contexto que las condiciones del YAML pueden leer -- mismo
# patron que in_trading_window() en policy_engine.py. Ver
# integration_example.py para el punto de conexion exacto.
# ---------------------------------------------------------------


@dataclass
class CalibrationResult:
    """
    Resultado completo de una calibracion -- no solo el numero final,
    sino el porque, para que quede auditable (mismo espiritu que
    AuditEntry en audit_schema.py: nunca una caja negra).
    """
    calibrated_authority: float
    is_calibrated: bool          # False si aun corre con base_authority
                                   # por falta de muestra (sample_size < min_sample_size)
    sample_size: int
    win_rate: Optional[float] = None
    payoff_ratio: Optional[float] = None
    raw_kelly: Optional[float] = None
    adjusted_kelly: Optional[float] = None
    reason: str = ""


class AuthorityCalibrator:
    """
    Calibrador de autoridad dinamica -- generalizacion de
    Cerebro._riesgo_kelly(). Un agente (de trading o de cualquier otro
    dominio) acumula un historial de resultados de sus acciones pasadas;
    este calibrador usa ese historial para determinar cuanta autoridad
    (tamaño de posicion, limite de gasto, alcance de una accion, lo que
    sea equivalente en el dominio) se le puede confiar AHORA MISMO --
    no un permiso fijo configurado una vez, sino algo que se contrae o
    se expande con el track record real y reciente.
    """

    def __init__(
        self,
        base_authority: float,
        floor: float,
        capacity: float,
        kelly_fraction: float = 0.50,
        min_sample_size: int = 30,
        window: int = 50,
        ceiling_fraction: float = 0.02,
        probe_fraction: float = 0.25,
    ):
        """
        base_authority:   autoridad por defecto mientras no hay muestra
                           suficiente (equivalente a riesgo_trade en KMT).
        floor:             piso absoluto -- la autoridad calibrada nunca
                           cae por debajo de esto (equivalente a
                           riesgo_minimo_dolares).
        capacity:          capacidad total del agente/cuenta (equivalente
                           a account_equity/capital) -- el techo se
                           calcula como fraccion de esto.
        kelly_fraction:    fraccion de Kelly completo a aplicar (0.50 =
                           half-Kelly, igual que KMT en produccion).
        min_sample_size:   minimo de resultados historicos antes de
                           calibrar con Kelly (igual que kelly_min_trades).
        window:            cuantos resultados recientes considerar
                           (igual que kelly_ventana -- ventana movil, no
                           todo el historico).
        ceiling_fraction:  techo como fraccion de `capacity` (igual que
                           kelly_techo_riesgo_pct).
        probe_fraction:    fraccion de base_authority a usar cuando Kelly
                           ajustado es <= 0 (igual que riesgo_sonda_fraccion).
        """
        if floor > base_authority:
            raise ValueError("floor no puede ser mayor que base_authority")
        if not (0 < kelly_fraction <= 1):
            raise ValueError("kelly_fraction debe estar en (0, 1]")
        if not (0 < ceiling_fraction <= 1):
            raise ValueError("ceiling_fraction debe estar en (0, 1]")

        self.base_authority = base_authority
        self.floor = floor
        self.capacity = capacity
        self.kelly_fraction = kelly_fraction
        self.min_sample_size = min_sample_size
        self.window = window
        self.ceiling_fraction = ceiling_fraction
        self.probe_fraction = probe_fraction

    def calibrate(self, outcome_history: list[float]) -> CalibrationResult:
        """
        outcome_history: lista de resultados pasados del agente, en el
        orden en que ocurrieron (mas antiguo primero, igual que
        self.e.trades en KMT). Un valor > 0 es un resultado exitoso, un
        valor <= 0 es un fallo -- la MAGNITUD importa (no solo signo),
        porque el ratio ganancia/perdida promedio es parte del calculo,
        exactamente como en _riesgo_kelly().
        """
        n_total = len(outcome_history)

        if n_total < self.min_sample_size:
            return CalibrationResult(
                calibrated_authority=max(self.base_authority, self.floor),
                is_calibrated=False,
                sample_size=n_total,
                reason=(
                    f"Muestra insuficiente ({n_total}/{self.min_sample_size}) -- "
                    f"usando autoridad base sin calibrar"
                ),
            )

        ventana = outcome_history[-self.window:]
        exitos = [x for x in ventana if x > 0]
        fallos = [x for x in ventana if x <= 0]

        if not exitos or not fallos:
            # Racha de solo exitos o solo fallos en la ventana -- Kelly
            # no esta definido (b requiere ambos). Igual que KMT: se
            # queda con base_authority en vez de dividir por cero.
            return CalibrationResult(
                calibrated_authority=max(self.base_authority, self.floor),
                is_calibrated=False,
                sample_size=n_total,
                reason="Ventana sin mezcla de exitos y fallos -- Kelly no definido, usando autoridad base",
            )

        p = len(exitos) / len(ventana)
        payoff_ratio = (sum(exitos) / len(exitos)) / abs(sum(fallos) / len(fallos))

        if payoff_ratio <= 0:
            return CalibrationResult(
                calibrated_authority=max(self.base_authority, self.floor),
                is_calibrated=False,
                sample_size=n_total,
                win_rate=p,
                reason="Payoff ratio invalido -- usando autoridad base",
            )

        raw_kelly = (p * payoff_ratio - (1 - p)) / payoff_ratio
        adjusted_kelly = raw_kelly * self.kelly_fraction

        if adjusted_kelly <= 0:
            authority = max(self.base_authority * self.probe_fraction, self.floor)
            return CalibrationResult(
                calibrated_authority=authority,
                is_calibrated=True,
                sample_size=n_total,
                win_rate=p,
                payoff_ratio=payoff_ratio,
                raw_kelly=raw_kelly,
                adjusted_kelly=adjusted_kelly,
                reason=(
                    f"Kelly ajustado no positivo ({adjusted_kelly:.4f}) -- track record "
                    f"reciente no sostiene autoridad plena, cayendo a fraccion de sondeo"
                ),
            )

        authority = self.capacity * min(adjusted_kelly, self.ceiling_fraction)
        authority = max(authority, self.floor)
        return CalibrationResult(
            calibrated_authority=authority,
            is_calibrated=True,
            sample_size=n_total,
            win_rate=p,
            payoff_ratio=payoff_ratio,
            raw_kelly=raw_kelly,
            adjusted_kelly=adjusted_kelly,
            reason=f"Calibrado sobre {len(ventana)} resultados (win_rate={p:.2%}, payoff={payoff_ratio:.2f})",
        )
