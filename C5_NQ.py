"""
MOTOR C5 NQ_MINI - RUPTURA DE RANGO N40 (5min)
Databento live -> velas 1min -> velas 5min derivadas -> ruptura N=40 ->
PickMyTrade -> Tradovate/Apex

CAMBIO (6-oct-2026): este servicio conserva el nombre NQ_MINI, su Postgres,
su cuenta y sus variables de entorno, pero desde esta fecha opera la MISMA
logica y configuracion que C5_MNQ: MNQ x5, TP=50pts / SL=30pts, timeout
15min, breaker diario $1,000.

Motivo: en vivo, del 2 al 6-oct, las mismas senales dieron +65 pts con
SL=30 (MNQ) y -22 pts con SL=12 (NQ). El SL de 12 pts se tocaba con el
ruido normal tras la ruptura. Ampliar el SL en NQ x1 (=MNQ x10) llevaba
el riesgo a $600 por SL y peores dias de -$3,000, insostenible para una
cuenta de 50K. MNQ x5 con SL=30 arriesga $300 por SL.

Los trades de NQ anteriores al 6-oct-2026 quedan en la misma base de
datos (trades_live); separarlos por fecha al analizar.

Variables de entorno: DATABASE_URL, DATABENTO_API_KEY, PICKMYTRADE_WEBHOOK,
                      PICKMYTRADE_TOKEN, PICKMYTRADE_ACCOUNT, MODO_SHADOW,
                      TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, json, time, logging, threading, queue
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import psycopg2

from policy_engine import PolicyEngine
from audit_schema import ActionType, MarketContext, RiskState
from authority_calibrator import AuthorityCalibrator

# ============================================================
# CONFIG -- unicamente los parametros de RUPTURA_N40 validado
# ============================================================
CFG = {
    # -------- INSTRUMENTO: MNQ --------
    "symbol_db": "MNQ.c.0",       # simbolo continuo para Databento
    "symbol_exec": "MNQ1!",       # simbolo de ejecucion en PickMyTrade/Tradovate
    "dataset": "GLBX.MDP3",
    "usd_punto": 2.0,             # $/punto de MNQ

    # -------- SENAL: ruptura del rango de N40 velas de 5min --------
    "tf_senal_min": 5,
    "n_ruptura": 40,               # velas de 5min de historia para el rango
    "cooldown_velas_5m": 40,       # no repetir senal misma direccion antes de esto

    # -------- ENTRADA / SALIDA (EXACTO al backtest validado) --------
    "tp_puntos": 50,
    "sl_puntos": 30,
    "limite_minutos": 15,          # timeout: cierra a mercado si no toco TP ni SL
    "contratos_fijos": 5,           # SIN Kelly -- tamano fijo, siempre 5 micros de MNQ
                                     # Historial de escalado (riesgo real medido con
                                     # analisis_drawdown_mnq_sl_ancho.py, TP=50/SL=30):
                                     #   2 contratos: drawdown max -$526.50, peor dia
                                     #     -$336, peor racha -$415
                                     #   4 contratos: drawdown max -$1,053, peor dia
                                     #     -$672, peor racha -$830
                                     #   5 contratos: drawdown max -$1,316.25, peor dia
                                     #     -$840 (2026-06-17), peor racha -$1,038
                                     # El peor dia a 5 contratos (-$840) superaba el
                                     # breaker diario que estaba en $800 -- por eso
                                     # perdida_max_dia se subio a $1,000 junto con este
                                     # cambio, para que el breaker vuelva a tener margen
                                     # real sobre el peor escenario ya visto en backtest.
                                     # Desde 6-oct-2026 este servicio (NQ_MINI) corre
                                     # esta misma config de MNQ x5.

    # -------- Cuenta y riesgo --------
    "capital": 50_000,
    "riesgo_trade": 400,
    "riesgo_minimo_dolares": 400,
    "perdida_max_dia": 1000,        # subido de 800 -> 1000 junto con el escalado a 5
                                     # contratos (ver nota en contratos_fijos arriba)
    "perdida_max_sem": 2000,

    # -------- Kelly dinamico (informativo -- NO dimensiona el sizing real) --------
    "kelly_fraccion": 0.50,
    "kelly_min_trades": 30,
    "kelly_ventana": 50,
    "kelly_techo_riesgo_pct": 0.02,
    "riesgo_sonda_fraccion": 0.25,

    # -------- Costos reales -- DEBEN coincidir con lo validado --------
    "comision_por_contrato_rt": 1.25,
    "spread_puntos_ny": 1.5,
    "spread_puntos_overnight": 3.5,     # identicos al motor V7_C1 -- valores reales de MNQ
    "hora_ny_inicio": dtime(9, 30),
    "hora_ny_fin": dtime(16, 0),

    "mantenimiento_ini": dtime(17, 0),
    "mantenimiento_fin": dtime(18, 0),

    # -------- RiskAval --------
    "policy_config_path": "policy_config_mnq_micro.yaml",
}
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("motor_c5")


class Telegram:
    """Alertas a Telegram -- nunca debe tumbar el motor si falla."""
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id
        self.activo = bool(token and chat_id)
        if not self.activo:
            log.warning("Telegram no configurado -- alertas desactivadas")

    def enviar(self, texto: str):
        if not self.activo:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={"chat_id": self.chat_id, "text": texto}, timeout=5)
        except Exception as e:
            log.error(f"Telegram error (no critico, se ignora): {e}")


@dataclass
class Pos:
    precio_entrada: float
    contratos: int
    sl: float
    tp: float
    direccion: int          # 1 = largo, -1 = corto
    ts_entrada: datetime
    decision_id: int
    ts_limite: datetime     # ts_entrada + limite_minutos -- timeout


class Estado:
    def __init__(self):
        self.pos: Optional[Pos] = None
        self.pnl_dia = 0.0
        self.pnl_sem = 0.0
        self.dia = None
        self.sem = None
        self.velas_1m = pd.DataFrame()   # base: ts/open/high/low/close/volume
        self.trades: List[float] = []
        # Senal detectada al cierre de una vela de 5min, pendiente de
        # ejecutar en el OPEN de la siguiente vela de 1min.
        self.senal_pendiente: Optional[Dict] = None
        self.pausado = False
        # Estado interno de deteccion de ruptura (evita reprocesar toda
        # la historia en cada vela de 5min)
        self.ultimo_dir_ruptura = 0
        self.ultima_i5_ruptura = -10**9
        self.contador_velas_5m = 0


class DB:
    """Conexion persistente a Postgres, con reconexion automatica. Thread-safe."""
    def __init__(self, url):
        self.url = url
        self.conn = None
        self._lock = threading.Lock()
        self._connect()
        self._crear_tablas()

    def _connect(self):
        self.conn = psycopg2.connect(self.url)
        self.conn.autocommit = False
        log.info("Conexion a Postgres establecida")

    def _ensure(self):
        if self.conn is None or self.conn.closed:
            log.warning("Conexion a DB caida, reconectando...")
            self._connect()

    def _crear_tablas(self):
        self._ensure()
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS decisiones (
                id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT NOW(),
                operar BOOLEAN, razon TEXT, contratos INT, precio NUMERIC,
                sl NUMERIC, tp NUMERIC, direccion INT, shadow BOOLEAN);
            CREATE TABLE IF NOT EXISTS ordenes (
                id SERIAL PRIMARY KEY, decision_id INT, ts TIMESTAMPTZ DEFAULT NOW(),
                payload JSONB, respuesta TEXT, exito BOOLEAN);
            CREATE TABLE IF NOT EXISTS trades_live (
                id SERIAL PRIMARY KEY, decision_id INT,
                ts_entrada TIMESTAMPTZ, ts_salida TIMESTAMPTZ,
                precio_entrada NUMERIC, precio_salida NUMERIC, contratos INT,
                pnl NUMERIC, razon TEXT, shadow BOOLEAN);
            CREATE TABLE IF NOT EXISTS control_motor (
                id INT PRIMARY KEY DEFAULT 1, pausado BOOLEAN DEFAULT FALSE,
                ts_actualizado TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT single_row CHECK (id = 1));
            INSERT INTO control_motor (id, pausado) VALUES (1, FALSE)
                ON CONFLICT (id) DO NOTHING;
            CREATE TABLE IF NOT EXISTS policy_audit (
                id SERIAL PRIMARY KEY, decision_id INT,
                ts TIMESTAMPTZ DEFAULT NOW(),
                entry_id TEXT, decision TEXT, rule_triggered TEXT, reason TEXT);
            """)
        self.conn.commit()
        log.info("DB lista")

    def get_pausado(self) -> bool:
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT pausado FROM control_motor WHERE id = 1")
                    row = cur.fetchone()
                    return bool(row[0]) if row else False
            except Exception as e:
                log.error(f"DB error (get_pausado): {e}")
                return False

    def set_pausado(self, valor: bool):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE control_motor SET pausado = %s, ts_actualizado = NOW() WHERE id = 1",
                        (valor,))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (set_pausado): {e}")
                try: self.conn.rollback()
                except Exception: pass

    def decision(self, operar, razon, contratos=None, precio=None, sl=None, tp=None,
                 direccion=None, shadow=True):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO decisiones
                        (operar,razon,contratos,precio,sl,tp,direccion,shadow)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (operar, razon, contratos, precio, sl, tp, direccion, shadow))
                    i = cur.fetchone()[0]
                self.conn.commit()
                return i
            except Exception as e:
                log.error(f"DB error (decision): {e}")
                try: self.conn.rollback()
                except Exception: pass
                return None

    def orden(self, did, payload, resp, ok):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("INSERT INTO ordenes (decision_id,payload,respuesta,exito) VALUES (%s,%s,%s,%s)",
                                (did, json.dumps(payload), str(resp)[:2000], ok))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (orden): {e}")
                try: self.conn.rollback()
                except Exception: pass

    def policy_audit(self, did, entry_id, decision_str, rule_triggered, reason):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO policy_audit
                        (decision_id, entry_id, decision, rule_triggered, reason)
                        VALUES (%s,%s,%s,%s,%s)""",
                        (did, entry_id, decision_str, rule_triggered, reason))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (policy_audit): {e}")
                try: self.conn.rollback()
                except Exception: pass

    def trade(self, did, p: Pos, salida, ts_out, pnl, razon, shadow):
        with self._lock:
            self._ensure()
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""INSERT INTO trades_live
                        (decision_id,ts_entrada,ts_salida,precio_entrada,precio_salida,contratos,pnl,razon,shadow)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (did, p.ts_entrada, ts_out, p.precio_entrada, salida, p.contratos, pnl, razon, shadow))
                self.conn.commit()
            except Exception as e:
                log.error(f"DB error (trade): {e}")
                try: self.conn.rollback()
                except Exception: pass


class Ejecutor:
    """Identico en espiritu al motor anterior: PickMyTrade + PolicyEngine
    (RiskAval) como segunda linea de defensa, y los mismos 3 fixes
    criticos de sincronizacion con el broker."""
    def __init__(self, url, token, account_id, shadow, db, tg: Optional[Telegram] = None):
        self.url = url
        self.token = token
        self.account = account_id
        self.shadow = shadow
        self.db = db
        self.tg = tg or Telegram(None, None)
        self.authority_calibrator = AuthorityCalibrator(
            base_authority=CFG["riesgo_trade"],
            floor=CFG["riesgo_minimo_dolares"],
            capacity=CFG["capital"],
            kelly_fraction=CFG["kelly_fraccion"],
            min_sample_size=CFG["kelly_min_trades"],
            window=CFG["kelly_ventana"],
            ceiling_fraction=CFG["kelly_techo_riesgo_pct"],
            probe_fraction=CFG["riesgo_sonda_fraccion"],
        )
        self.policy_engine = PolicyEngine(
            CFG["policy_config_path"],
            authority_calibrator=self.authority_calibrator,
        )

    def _base(self, action, qty, price):
        return {
            "symbol": CFG["symbol_exec"],
            "strategy_name": "MOTOR_C5_NQ_MINI",
            "date": datetime.now(UTC).isoformat(),
            "data": action,
            "quantity": str(qty),
            "risk_percentage": 0,
            "price": str(round(price, 2)),
            "tp": 0, "percentage_tp": 0, "dollar_tp": 0,
            "sl": 0, "dollar_sl": 0, "percentage_sl": 0,
            "trail": 0, "trail_stop": 0, "trail_trigger": 0, "trail_freq": 0,
            "update_tp": False, "update_sl": False,
            "breakeven": 0, "breakeven_offset": 0,
            "token": self.token,
            "pyramid": False,
            "same_direction_ignore": True,
            "reverse_order_close": False,
            "multiple_accounts": [{
                "token": self.token, "account_id": self.account,
                "risk_percentage": 0, "quantity_multiplier": 1
            }]
        }

    def _post(self, did, payload):
        if self.shadow:
            log.info(f"[SHADOW] {payload['data']} {payload['quantity']}c @ {payload['price']} "
                      f"dollar_tp={payload['dollar_tp']} dollar_sl={payload['dollar_sl']}")
            self.db.orden(did, payload, "SHADOW", True)
            return True
        try:
            r = requests.post(self.url, json=payload, timeout=10)
            ok = 200 <= r.status_code < 300
            log.info(f"[LIVE] {r.status_code} {r.text[:200]}")
            self.db.orden(did, payload, r.text, ok)
            if not ok:
                self.tg.enviar(f"\u26a0\ufe0f ORDEN RECHAZADA por PickMyTrade\nHTTP {r.status_code}\n{r.text[:200]}")
            return ok
        except Exception as e:
            log.error(f"[LIVE] fallo: {e}")
            self.db.orden(did, payload, str(e), False)
            self.tg.enviar(f"\u26a0\ufe0f ORDEN FALLIDA (error de red hacia PickMyTrade)\n{e}")
            return False

    def entrar(self, did, qty, precio, sl_dist_pts, tp_dist_pts, direccion,
               daily_pnl, weekly_pnl, open_contracts, outcome_history=None):
        """Pasa primero por el policy engine (RiskAval). dollar_sl/dollar_tp
        se anclan al fill real -- mismo fix critico que el motor anterior
        (evita SL invalido si hubo slippage favorable en la entrada)."""
        audit_entry = self.policy_engine.evaluate(
            action_type=ActionType.OPEN_POSITION,
            proposed_params={"contracts": qty},
            market_context=MarketContext(
                timestamp=datetime.now(UTC), instrument=CFG["symbol_exec"],
                price=precio, atr=None, minutes_to_next_news_event=None,
            ),
            risk_state=RiskState(
                daily_pnl=daily_pnl, weekly_pnl=weekly_pnl,
                open_contracts=open_contracts, account_equity=CFG["capital"],
            ),
            agent_id="MOTOR_C5", account_id=self.account,
            outcome_history=outcome_history,
        )
        self.db.policy_audit(did, audit_entry.entry_id, audit_entry.decision.value,
                              audit_entry.rule_triggered, audit_entry.reason)

        if audit_entry.decision.value == "blocked":
            log.warning(f"[POLICY][{audit_entry.entry_id}] Orden BLOQUEADA: {audit_entry.reason}")
            self.tg.enviar(f"\U0001f6ab Orden bloqueada por policy engine: {audit_entry.reason}")
            return False
        if audit_entry.decision.value == "logged":
            log.info(f"[POLICY][{audit_entry.entry_id}] Aprobada con marca de revision: {audit_entry.reason}")

        accion = "buy" if direccion == 1 else "sell"
        sl_dist_dolares = round(sl_dist_pts * CFG["usd_punto"], 2)
        tp_dist_dolares = round(tp_dist_pts * CFG["usd_punto"], 2)
        p = self._base(accion, qty, precio)
        p["sl"] = 0
        p["dollar_sl"] = sl_dist_dolares
        p["tp"] = 0
        p["dollar_tp"] = tp_dist_dolares
        return self._post(did, p)

    def salir(self, did, qty, precio):
        # "flat" cierra toda la posicion del simbolo -- "exit" NO es una
        # accion valida en PickMyTrade y se rechaza en silencio.
        return self._post(did, self._base("flat", qty, precio))


class Cerebro:
    """Ruptura del rango de N=40 velas de 5min, ambas direcciones.
    TP/SL fijos en puntos, timeout 15min. Sin trailing, sin filtros de
    hora/noticia/regimen -- exacto al backtest validado."""

    def __init__(self, e: Estado, db: DB, ex: Ejecutor, tg: Optional[Telegram] = None):
        self.e, self.db, self.ex = e, db, ex
        self.tg = tg or Telegram(None, None)
        self._breaker_avisado_dia = None
        self._breaker_avisado_sem = None

    def _spread_por_sesion(self, hora_ny):
        if CFG["hora_ny_inicio"] <= hora_ny <= CFG["hora_ny_fin"]:
            return CFG["spread_puntos_ny"]
        return CFG["spread_puntos_overnight"]

    def _en_mantenimiento(self, ts_ny_time):
        return CFG["mantenimiento_ini"] <= ts_ny_time < CFG["mantenimiento_fin"]

    def _breakers(self, ts):
        ny = ts.astimezone(NY)
        d, w = ny.date(), ny.isocalendar()[1]
        if self.e.dia != d:
            self.e.dia, self.e.pnl_dia = d, 0.0
        if self.e.sem != w:
            self.e.sem, self.e.pnl_sem = w, 0.0
        if self.e.pnl_dia <= -CFG["perdida_max_dia"]:
            if self._breaker_avisado_dia != d:
                self._breaker_avisado_dia = d
                self.tg.enviar(f"\U0001f6d1 BREAKER DIARIO activado\nPnL del dia: ${self.e.pnl_dia:.2f}\nEntradas nuevas bloqueadas hasta manana.")
            return "BREAKER_DIARIO"
        if self.e.pnl_sem <= -CFG["perdida_max_sem"]:
            if self._breaker_avisado_sem != w:
                self._breaker_avisado_sem = w
                self.tg.enviar(f"\U0001f6d1 BREAKER SEMANAL activado\nPnL de la semana: ${self.e.pnl_sem:.2f}\nEntradas nuevas bloqueadas hasta la proxima semana.")
            return "BREAKER_SEMANAL"
        return None

    def _cerrar(self, p: Pos, precio_salida: float, ts, razon: str):
        """Mismo fix critico que el motor anterior: si el broker rechaza
        el cierre, NO se marca la posicion como cerrada -- se reintenta
        en la siguiente vela, nunca se abandona a medias."""
        ok = self.ex.salir(p.decision_id, p.contratos, precio_salida)
        if not ok:
            log.error(f"FALLO el cierre en broker (razon={razon}) -- posicion se mantiene ABIERTA internamente")
            self.tg.enviar(f"\U0001f534 CIERRE RECHAZADO por broker ({razon}). Motor sigue tratando la posicion como ABIERTA. Revisar manualmente si persiste.")
            return

        hora_salida = ts.astimezone(NY).time()
        spread_aplicable = self._spread_por_sesion(hora_salida)
        costo_spread = spread_aplicable * CFG["usd_punto"] * p.contratos * 0.5
        costo_comision = CFG["comision_por_contrato_rt"] * p.contratos
        pnl_bruto = (precio_salida - p.precio_entrada) * CFG["usd_punto"] * p.contratos * p.direccion
        pnl = pnl_bruto - costo_spread - costo_comision

        self.db.trade(p.decision_id, p, precio_salida, ts, pnl, razon, self.ex.shadow)
        self.e.pnl_dia += pnl
        self.e.pnl_sem += pnl
        self.e.trades.append(pnl)
        log.info(f"CIERRE {razon} pnl={pnl:.2f} dia={self.e.pnl_dia:.2f}")
        emoji = "\u2705" if pnl > 0 else ("\u26aa" if pnl == 0 else "\U0001f534")
        modo = "SHADOW" if self.ex.shadow else "LIVE"
        self.tg.enviar(f"{emoji} CIERRE [{modo}] {razon}\nPnL: ${pnl:.2f}\nPnL del dia: ${self.e.pnl_dia:.2f}")
        self.e.pos = None

    def _gestionar(self, v, ts):
        """Cada vela de 1min: chequea SL, TP, y timeout (15min). Sin
        trailing, sin parciales -- salida binaria, exacto al backtest."""
        p = self.e.pos
        if ts >= p.ts_limite:
            self._cerrar(p, float(v["close"]), ts, "TIMEOUT_15MIN")
            return
        sl_tocado = (v["low"] <= p.sl) if p.direccion == 1 else (v["high"] >= p.sl)
        tp_tocado = (v["high"] >= p.tp) if p.direccion == 1 else (v["low"] <= p.tp)
        # SL primero si ambos se tocan en la misma vela (mismo criterio
        # conservador que el backtest de verificacion).
        if sl_tocado:
            self._cerrar(p, p.sl, ts, "SL")
            return
        if tp_tocado:
            self._cerrar(p, p.tp, ts, "TP")
            return

    def _hueco_es_esperado(self, ts_antes: pd.Timestamp, ts_despues: pd.Timestamp) -> bool:
        """Distingue un cierre NORMAL de CME (diario 17:00-18:00 NY, o
        fin de semana viernes 17:00 -> domingo 18:00) de una caida real
        de feed.

        FIX (oct 2026):
          1. Las velas se etiquetan por su INICIO: la ultima vela antes
             del cierre es 16:55, que termina a las 17:00. Se compara el
             FIN de la vela, no su inicio (el bug bloqueaba ~200 min de
             rupturas tras cada reapertura).
          2. Tolerancia: si la primera vela tras la reapertura llega a
             las 18:05 o 18:10 (sin trades en el primer bloque), sigue
             siendo un cierre normal.
          3. Horas construidas por fecha en NY, no sumando horas, para
             que los cambios de horario (marzo/noviembre) no lo rompan."""
        tf = timedelta(minutes=CFG["tf_senal_min"])
        tol = tf * 3
        fin_antes = ts_antes.tz_convert("America/New_York") + tf
        despues = ts_despues.tz_convert("America/New_York")

        def _ny(fecha, hora):
            return pd.Timestamp(datetime.combine(fecha, hora)).tz_localize("America/New_York")

        cierre = _ny(fin_antes.date(), CFG["mantenimiento_ini"])          # 17:00 de ese dia
        if not (cierre - tol <= fin_antes <= cierre):
            return False

        # Cierre diario (lun-jue): reabre 18:00 el mismo dia
        reapertura = _ny(fin_antes.date(), CFG["mantenimiento_fin"])
        if reapertura <= despues <= reapertura + tol:
            return True

        # Cierre semanal: viernes 17:00 -> domingo 18:00
        if fin_antes.weekday() == 4:
            reap_dom = _ny(fin_antes.date() + timedelta(days=2), CFG["mantenimiento_fin"])
            if reap_dom <= despues <= reap_dom + tol:
                return True

        return False

    def _ventana_tiene_hueco(self, velas_5m: pd.DataFrame) -> bool:
        """FIX (sep 2026, tras auditoria de trades LIVE): detecta si las
        ultimas N+1 velas de 5min usadas para el rollmax/rollmin tienen un
        hueco de tiempo REAL (no un cierre de mercado normal) entre ellas
        -- misma clase de problema que el bug de rollover de contrato en
        el backtest (un salto de precio no representativo de un
        movimiento real disparando una ruptura falsa), pero causado aqui
        por caidas/reconexiones del feed en vivo (confirmadas en
        produccion: perdida de ticks parciales, corte de ~25h el
        12-sep-2026). Si la ventana no es continua Y el hueco no es un
        cierre esperado (fin de semana / mantenimiento), NO se confia en
        el rollmax/rollmin -- se descarta la deteccion de esa vela."""
        n = CFG["n_ruptura"]
        ventana = velas_5m.tail(n + 1)
        if len(ventana) < 2:
            return False
        ts_col = pd.to_datetime(ventana["ts"], utc=True)
        max_hueco_min = CFG["tf_senal_min"] * 1.5  # tolera jitter normal, no huecos reales
        for i in range(1, len(ts_col)):
            delta_min = (ts_col.iloc[i] - ts_col.iloc[i - 1]).total_seconds() / 60
            if delta_min > max_hueco_min:
                if not self._hueco_es_esperado(ts_col.iloc[i - 1], ts_col.iloc[i]):
                    return True
        return False

    def _detectar_ruptura_5m(self, velas_5m: pd.DataFrame) -> Optional[int]:
        """Ruptura del rango de las N40 velas de 5min ANTERIORES a la
        actual (shift, sin look-ahead). Cooldown de 40 velas misma
        direccion. Devuelve 1 (alcista), -1 (bajista), o None.

        FIX (sep 2026): antes de calcular rollmax/rollmin, verifica que
        la ventana sea temporalmente continua -- distinguiendo cierres de
        mercado normales (fin de semana, mantenimiento) de caidas reales
        de feed (ver _ventana_tiene_hueco). Sin este chequeo, el motor
        puede confundir un salto de precio causado por datos faltantes
        con una ruptura real -- exactamente el patron que la auditoria de
        trades en vivo encontro (6 de 8 entradas limpias sin ruptura real
        confirmada contra datos completos de Databento).

        No hay temporizador fijo de espera: en cuanto la ventana de N40
        velas vuelve a ser continua (naturalmente, apenas se acumulan
        datos limpios), la deteccion se reactiva sola -- no antes, porque
        no se puede evaluar con menos de 200min de historia real, pero
        tampoco despues, porque no hace falta esperar mas que eso."""
        n = CFG["n_ruptura"]
        if len(velas_5m) < n + 1:
            return None

        if self._ventana_tiene_hueco(velas_5m):
            log.warning("Ruptura descartada: la ventana de N40 velas tiene un hueco de "
                        "tiempo real (feed incompleto) -- rollmax/rollmin no confiable")
            return None

        h = velas_5m["high"].to_numpy()
        l = velas_5m["low"].to_numpy()
        c = velas_5m["close"].to_numpy()

        # AVISO (no bloqueante): salto de precio anormal entre cierres
        # consecutivos de 5min. El motor usa el simbolo continuo de
        # Databento (MNQ.c.0), que ya ajusta el rollover de contrato
        # automaticamente -- esto NUNCA deberia dispararse por un cambio
        # de contrato real. Es una alerta de visibilidad por si acaso,
        # no un fix a un bug conocido (el rollover solo afectaba a los
        # CSV crudos historicos del backtest, no a este feed en vivo).
        if len(c) >= 2 and not np.isnan(c[-2]):
            salto = abs(c[-1] - c[-2])
            if salto > 100:
                log.warning(f"Salto de precio inusual entre velas de 5min consecutivas: "
                            f"{salto:.2f}pts ({c[-2]:.2f} -> {c[-1]:.2f}) -- revisar, "
                            f"no deberia ocurrir con simbolo continuo")

        rollmax = pd.Series(h).rolling(n).max().shift(1).to_numpy()
        rollmin = pd.Series(l).rolling(n).min().shift(1).to_numpy()
        i = len(velas_5m) - 1
        if np.isnan(rollmax[i]) or np.isnan(rollmin[i]):
            return None
        d = 0
        if c[i] > rollmax[i]:
            d = 1
        elif c[i] < rollmin[i]:
            d = -1
        if d == 0:
            return None
        if d == self.e.ultimo_dir_ruptura and (self.e.contador_velas_5m - self.e.ultima_i5_ruptura) < n:
            return None
        self.e.ultimo_dir_ruptura = d
        self.e.ultima_i5_ruptura = self.e.contador_velas_5m
        return d

    def evaluar_cierre_5min(self, velas_5m: pd.DataFrame, ts):
        """Se llama SOLO cuando una vela de 1min cierra un bloque de
        5min. Detecta ruptura y, si hay, deja la senal pendiente para
        ejecutar en el open de la PROXIMA vela de 1min (igual patron
        que el motor anterior: direccion_pendiente)."""
        self.e.contador_velas_5m += 1
        if self.e.pos is not None or self.e.senal_pendiente is not None:
            return
        if self.e.pausado:
            return
        br = self._breakers(ts)
        if br:
            self.db.decision(False, br, shadow=self.ex.shadow)
            return
        ny = ts.astimezone(NY)
        if self._en_mantenimiento(ny.time()):
            return

        d = self._detectar_ruptura_5m(velas_5m)
        if d is None:
            return

        qty = CFG["contratos_fijos"]
        riesgo = qty * CFG["sl_puntos"] * CFG["usd_punto"]  # riesgo real implicado, solo para log/Telegram

        self.e.senal_pendiente = {"direccion": d, "qty": qty, "riesgo": riesgo, "ts_detectada": ts}
        log.info(f"RUPTURA N40 DETECTADA dir={'ALCISTA' if d==1 else 'BAJISTA'} "
                  f"(pendiente de ejecutar en la proxima vela de 1min) qty={qty}")

    def evaluar_1min(self, v, ts):
        """Se llama en CADA cierre de vela de 1min."""
        # PASO 1: ejecutar senal pendiente (si hay), en el open de esta vela.
        if self.e.senal_pendiente is not None and self.e.pos is None:
            if self.e.pausado:
                log.info("Senal pendiente retenida: motor pausado (kill switch)")
                return
            sp = self.e.senal_pendiente
            ts_det = sp.get("ts_detectada")
            if ts_det is not None and (ts - ts_det) > timedelta(minutes=CFG["tf_senal_min"] * 2):
                antiguedad = (ts - ts_det).total_seconds() / 60
                log.warning(f"Senal pendiente EXPIRADA ({antiguedad:.0f}min), descartada")
                self.db.decision(False, f"SENAL_EXPIRADA({antiguedad:.0f}min)", shadow=self.ex.shadow)
                self.tg.enviar(f"\u231b Senal pendiente expirada ({antiguedad:.0f}min sin ejecutarse) -- descartada")
                self.e.senal_pendiente = None
                return

            self.e.senal_pendiente = None
            d = sp["direccion"]
            hora_ny_actual = ts.astimezone(NY).time()
            medio_spread = self._spread_por_sesion(hora_ny_actual) * 0.5
            precio_entrada = float(v["open"]) + medio_spread * d
            sl = precio_entrada - CFG["sl_puntos"] * d
            tp = precio_entrada + CFG["tp_puntos"] * d

            did = self.db.decision(True, "RUPTURA_N40", sp["qty"], precio_entrada, sl, tp, d, self.ex.shadow)
            open_contratos_actual = self.e.pos.contratos if self.e.pos else 0

            if self.ex.entrar(did, sp["qty"], precio_entrada, CFG["sl_puntos"], CFG["tp_puntos"], d,
                               daily_pnl=self.e.pnl_dia, weekly_pnl=self.e.pnl_sem,
                               open_contracts=open_contratos_actual, outcome_history=self.e.trades):
                ts_limite = ts + timedelta(minutes=CFG["limite_minutos"])
                self.e.pos = Pos(precio_entrada, sp["qty"], sl, tp, d, ts, did, ts_limite)
                log.info(f"ENTRADA {'LARGO' if d==1 else 'CORTO'} {sp['qty']}c @ {precio_entrada:.2f} "
                          f"SL={sl:.2f} TP={tp:.2f} timeout={ts_limite.astimezone(NY).strftime('%H:%M')} NY "
                          f"riesgo=${sp['riesgo']:.0f}")
                modo = "SHADOW" if self.ex.shadow else "LIVE"
                self.tg.enviar(
                    f"\U0001f7e2 ENTRADA [{modo}] {'LARGO' if d==1 else 'CORTO'}\n"
                    f"{sp['qty']} contrato(s) @ {precio_entrada:.2f}\n"
                    f"SL={sl:.2f}  TP={tp:.2f}\n"
                    f"Timeout: {CFG['limite_minutos']}min  riesgo=${sp['riesgo']:.0f}"
                )
            return

        # PASO 2: gestionar posicion abierta (SL/TP/timeout).
        if self.e.pos:
            self._gestionar(v, ts)
            return


class Feed:
    """Construye velas de 1min desde ticks de Databento (solo precio y
    tamano -- esta senal no necesita clasificacion compra/venta). Al
    cerrar cada vela de 1min: (a) evalua ejecucion/gestion via
    Cerebro.evaluar_1min, (b) si esa vela cierra un bloque de 5min,
    deriva las velas de 5min por resample y llama a
    Cerebro.evaluar_cierre_5min. Patron productor-consumidor identico
    al motor anterior para no atrasarse frente a picos de volumen."""

    def __init__(self, key, e: Estado, cerebro: Cerebro, tg: Optional[Telegram] = None):
        self.key, self.e, self.cerebro = key, e, cerebro
        self.tg = tg or Telegram(None, None)
        self.buf: List[Dict] = []
        self.vela_ini = None
        self.map_instrumento = {}
        self._cola = queue.Queue(maxsize=50000)
        self._hilo_consumidor = None

    def _asegurar_consumidor(self):
        if self._hilo_consumidor is None or not self._hilo_consumidor.is_alive():
            self._hilo_consumidor = threading.Thread(target=self._consumidor, daemon=True)
            self._hilo_consumidor.start()
            log.info("Hilo consumidor iniciado")

    def _cerrar_vela_1m(self, ts):
        if not self.buf:
            return
        pr = [t["p"] for t in self.buf]
        fila = {"ts": self.vela_ini, "open": pr[0], "high": max(pr), "low": min(pr),
                "close": pr[-1], "volume": sum(t["s"] for t in self.buf)}
        self.buf = []
        self.e.velas_1m = pd.concat([self.e.velas_1m, pd.DataFrame([fila])],
                                     ignore_index=True).tail(3000).reset_index(drop=True)

        v = self.e.velas_1m.iloc[-1]
        self.cerebro.evaluar_1min(v, ts)

        ts_ny = self.vela_ini.astimezone(NY) if self.vela_ini.tzinfo else self.vela_ini.replace(tzinfo=UTC).astimezone(NY)
        if ts_ny.minute % CFG["tf_senal_min"] == (CFG["tf_senal_min"] - 1):
            velas_5m = self._derivar_5min()
            if velas_5m is not None and len(velas_5m):
                self.cerebro.evaluar_cierre_5min(velas_5m, ts)

    def _derivar_5min(self) -> Optional[pd.DataFrame]:
        df = self.e.velas_1m
        n_min = CFG["n_ruptura"] * CFG["tf_senal_min"] + CFG["tf_senal_min"] * 3
        if len(df) < CFG["tf_senal_min"]:
            return None
        cola = df.tail(min(len(df), n_min + 50)).copy()
        cola["ts"] = pd.to_datetime(cola["ts"], utc=True)
        cola = cola.set_index("ts")
        agregado = cola.resample(f"{CFG['tf_senal_min']}min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return agregado.reset_index()

    def _perder_buffer_parcial(self, motivo: str):
        if self.buf:
            log.warning(f"Buffer parcial descartado por {motivo}: {len(self.buf)} ticks")
            self.tg.enviar(f"\u26a0\ufe0f Reconexion del feed: se perdieron {len(self.buf)} ticks parciales de vela en curso")
        self.buf = []
        self.vela_ini = None
        # NOTA: ya no hace falta marcar un timestamp de "calentamiento" --
        # Cerebro._ventana_tiene_hueco() detecta directamente, en cada
        # evaluacion, si la ventana de 40 velas todavia contiene el hueco
        # dejado por esta reconexion. Se autocorrige apenas la ventana
        # vuelve a ser continua, sin esperar mas ni menos de lo necesario.

    def tick(self, ts, price, size):
        ini = ts.replace(second=0, microsecond=0)
        if self.vela_ini is None:
            self.vela_ini = ini
        elif ini > self.vela_ini:
            self._cerrar_vela_1m(ini)
            self.vela_ini = ini
        self.buf.append({"p": price, "s": size})

    def _instrument_id(self, r):
        hd = getattr(r, "hd", None)
        if hd is not None and hasattr(hd, "instrument_id"):
            return hd.instrument_id
        return getattr(r, "instrument_id", None)

    def _consumidor(self):
        avisado_cola_llena = False
        while True:
            try:
                r = self._cola.get(timeout=1)
            except queue.Empty:
                continue
            try:
                tipo = type(r).__name__
                if tipo == "SymbolMappingMsg":
                    out_sym = getattr(r, "stype_out_symbol", None)
                    iid = self._instrument_id(r)
                    if out_sym and iid is not None:
                        self.map_instrumento[iid] = True
                        log.info(f"Mapeo confirmado: instrument_id={iid} -> {out_sym}")
                    continue
                if tipo != "TradeMsg":
                    continue
                iid = self._instrument_id(r)
                if iid is None or iid not in self.map_instrumento:
                    continue
                ts = datetime.fromtimestamp(r.ts_event / 1e9, tz=UTC)
                self.tick(ts, r.price / 1e9, r.size)
            except Exception as ex:
                log.error(f"consumidor err: {ex}")

            qsize = self._cola.qsize()
            if qsize > 5000 and not avisado_cola_llena:
                avisado_cola_llena = True
                log.warning(f"Cola de procesamiento acumulando: {qsize} pendientes")
                self.tg.enviar(f"\U0001f7e0 Cola de ticks acumulando ({qsize} pendientes)")
            elif qsize < 500:
                avisado_cola_llena = False

    def correr(self):
        import databento as dbn
        self._perder_buffer_parcial("reconexion o arranque")
        self._asegurar_consumidor()
        log.info("Conectando Databento live...")
        cli = dbn.Live(key=self.key)
        cli.subscribe(dataset=CFG["dataset"], schema="trades",
                      symbols=[CFG["symbol_db"]], stype_in="continuous")
        log.info("Feed OK")
        for r in cli:
            try:
                self._cola.put_nowait(r)
            except queue.Full:
                try: self._cola.get_nowait()
                except queue.Empty: pass
                self._cola.put_nowait(r)


def kill_switch_listener(e: Estado, db: DB, tg: Telegram):
    if not tg.activo:
        return
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{tg.token}/getUpdates"
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                texto = (msg.get("text") or "").strip().lower()
                chat_id_msg = str(msg.get("chat", {}).get("id", ""))
                if chat_id_msg != str(tg.chat_id):
                    continue
                if texto == "/pausar":
                    e.pausado = True
                    db.set_pausado(True)
                    tg.enviar("\u23f8\ufe0f Motor PAUSADO. No se abriran entradas nuevas.\nUna posicion abierta se sigue gestionando normalmente.\nEnvia /reanudar para reactivar.")
                    log.info("Kill switch: PAUSADO por comando de Telegram")
                elif texto == "/reanudar":
                    e.pausado = False
                    db.set_pausado(False)
                    tg.enviar("\u25b6\ufe0f Motor REANUDADO.")
                    log.info("Kill switch: REANUDADO por comando de Telegram")
                elif texto == "/estado":
                    estado_txt = "\u23f8\ufe0f PAUSADO" if e.pausado else "\u25b6\ufe0f ACTIVO"
                    pos_txt = "Sin posicion abierta" if e.pos is None else \
                        f"Posicion abierta: {'LARGO' if e.pos.direccion==1 else 'CORTO'} {e.pos.contratos}c @ {e.pos.precio_entrada:.2f}"
                    tg.enviar(f"{estado_txt}\n{pos_txt}\nPnL del dia: ${e.pnl_dia:.2f}")
        except Exception as ex:
            log.error(f"kill_switch_listener error (no critico): {ex}")
            time.sleep(5)


def watchdog(e: Estado, tg: Telegram, umbral_min: int = 15):
    avisado = False
    ultimo_ts_vela = None
    ultima_vez_cambio = time.time()
    while True:
        time.sleep(300)
        try:
            ahora = datetime.now(UTC)
            ny = ahora.astimezone(NY)
            if CFG["mantenimiento_ini"] <= ny.time() < CFG["mantenimiento_fin"]:
                avisado = False
                continue
            ts_actual = e.velas_1m.iloc[-1]["ts"] if len(e.velas_1m) else None
            if ts_actual is not None and ts_actual != ultimo_ts_vela:
                ultimo_ts_vela = ts_actual
                ultima_vez_cambio = time.time()
                avisado = False
                continue
            minutos_silencio = (time.time() - ultima_vez_cambio) / 60
            if minutos_silencio >= umbral_min and not avisado:
                avisado = True
                tg.enviar(f"\U0001f7e0 WATCHDOG: sin velas nuevas hace ~{int(minutos_silencio)}min. Revisar Railway.")
                log.warning(f"WATCHDOG: sin velas nuevas hace {int(minutos_silencio)}min")
        except Exception as ex:
            log.error(f"watchdog error: {ex}")


def main():
    db_url = os.environ["DATABASE_URL"]
    shadow = os.environ.get("MODO_SHADOW", "true").lower() != "false"
    log.info("=" * 60)
    log.info(f"MOTOR C5 NQ_MINI -- RUPTURA_N40 | {'SHADOW' if shadow else 'LIVE'} | capital={CFG['capital']}")
    log.info(f"N={CFG['n_ruptura']} velas 5min | TP={CFG['tp_puntos']}pts SL={CFG['sl_puntos']}pts "
              f"(RR={CFG['tp_puntos']/CFG['sl_puntos']:.2f}:1) | timeout={CFG['limite_minutos']}min")
    log.info(f"Instrumento: {CFG['symbol_exec']} (usd_punto=${CFG['usd_punto']})")
    log.info(f"riesgo_trade=${CFG['riesgo_trade']} riesgo_minimo=${CFG['riesgo_minimo_dolares']} contratos_fijos={CFG['contratos_fijos']}")
    log.info(f"perdida_max_dia=${CFG['perdida_max_dia']} perdida_max_sem=${CFG['perdida_max_sem']}")
    log.info(f"Policy engine (RiskAval): {CFG['policy_config_path']}")
    log.info("=" * 60)

    tg = Telegram(os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))
    db = DB(db_url)
    e = Estado()
    e.pausado = db.get_pausado()
    ex = Ejecutor(os.environ.get("PICKMYTRADE_WEBHOOK", ""),
                  os.environ.get("PICKMYTRADE_TOKEN", ""),
                  os.environ.get("PICKMYTRADE_ACCOUNT", ""),
                  shadow, db, tg)
    cerebro = Cerebro(e, db, ex, tg)

    threading.Thread(target=watchdog, args=(e, tg), daemon=True).start()
    threading.Thread(target=kill_switch_listener, args=(e, db, tg), daemon=True).start()

    estado_inicial = "\u23f8\ufe0f PAUSADO (recuperado)" if e.pausado else "\u25b6\ufe0f ACTIVO"
    tg.enviar(f"\U0001f7e2 Motor C5 NQ_MINI [RUPTURA_N40] iniciado | {'SHADOW' if shadow else 'LIVE'} | "
              f"{CFG['symbol_exec']} | TP={CFG['tp_puntos']}/SL={CFG['sl_puntos']} | {estado_inicial}\n"
              f"Comandos: /pausar /reanudar /estado")

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        log.warning("Sin DATABENTO_API_KEY - standby")
        while True:
            time.sleep(60)

    feed = Feed(key, e, cerebro, tg)
    while True:
        try:
            feed.correr()
        except Exception as ex2:
            log.error(f"Feed caido: {ex2}. Retry 30s")
            tg.enviar(f"\U0001f534 Feed de Databento caido: {str(ex2)[:200]}\nReintentando en 30s...")
            time.sleep(30)


if __name__ == "__main__":
    main()
