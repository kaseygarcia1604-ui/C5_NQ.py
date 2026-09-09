"""
audit_schema.py -- Modelos de datos del motor de politicas de RiskAval.

Cada accion propuesta por un motor de trading (o, en general, por un
agente autonomo de CUALQUIER dominio) se representa como UN AuditEntry:
contexto de mercado + estado de riesgo + decision tomada + motivo. Es
el objeto que fluye entre policy_engine.py, anonymizer.py y
aggregated_dataset.py.

IMPORTANTE (no es una restriccion, es una aclaracion): action_type
ACEPTA CUALQUIER STRING, no solo los valores de ActionType de abajo.
policy_engine.py y anonymizer.py ya usan
`action_type.value if isinstance(action_type, ActionType) else str(action_type)`
-- si le pasas "send_email" o "delete_file" en vez de un miembro del
enum, funciona igual, verificado. ActionType existe como CONVENIENCIA
para el caso de uso de trading (autocompletado, evitar typos en
"open_position"), no como una lista cerrada de lo unico permitido. Si
tu agente no es de trading, simplemente pasa tu propio string de accion
como `action_type` -- no hace falta extender este enum.

RECONSTRUIDO a partir de fragmentos recuperados de la sesion original
(chat "NQ", 2026-07-09) -- estos dataclasses coinciden con el diseño
verificado en esa sesion.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    OPEN_POSITION = "open_position"
    MODIFY_STOP = "modify_stop"
    MODIFY_TARGET = "modify_target"
    CLOSE_POSITION = "close_position"
    INCREASE_SIZE = "increase_size"
    CANCEL_ORDER = "cancel_order"
    QUERY_ACCOUNT_STATUS = "query_account_status"


class Decision(str, Enum):
    APPROVED = "approved"
    LOGGED = "logged"
    BLOCKED = "blocked"


@dataclass
class MarketContext:
    timestamp: datetime
    instrument: str
    price: float
    atr: Optional[float] = None
    delta_signal: Optional[str] = None
    session: Optional[str] = None
    minutes_to_next_news_event: Optional[int] = None


@dataclass
class RiskState:
    daily_pnl: float
    weekly_pnl: float
    open_contracts: int
    account_equity: float
    position_size_requested: Optional[int] = None


@dataclass
class AuditEntry:
    entry_id: str
    agent_id: str
    account_id: str
    action_type: ActionType
    proposed_params: dict
    market_context: MarketContext
    risk_state: RiskState
    decision: Decision
    rule_triggered: Optional[str] = None
    reason: Optional[str] = None
    outcome_pnl: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None
    # NUEVO (authority_calibrator): detalle de la calibracion de
    # autoridad usada en esta decision, si el PolicyEngine tenia un
    # AuthorityCalibrator configurado y se le paso outcome_history.
    # None si no aplica (motor sin calibrador, o sin historial aun).
    # Tipo real: Optional["CalibrationResult"] -- se evita el import
    # directo aqui para no crear dependencia circular entre
    # audit_schema.py y authority_calibrator.py; policy_engine.py hace
    # el import real y pasa el objeto.
    calibration_detail: Optional[object] = None
