import cflib
import cflib.crtp
import logging
import time
import threading
import numpy as np
from typing import Dict, Any, Optional

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

logger = logging.getLogger(__name__)


cflib.crtp.init_drivers(enable_debug_driver=False)


class CrazyFlieStateObserver:
    """
    Helper class that connects to the Crazyflie and maintains a 13D state vector:

        [x, y, z,
         qw, qx, qy, qz,
         vx, vy, vz,
         wx, wy, wz]

    - Position/velocity from stateEstimate
    - Quaternion from stateEstimate (normalized)
    - Angular velocity from gyro.* (deg/s -> rad/s)
    """

    def __init__(self, uri: str, log_period_ms: int = 10, cache_dir: str = "./cache"):
        self._uri = uri
        self.log_period_ms = int(log_period_ms)
        self._cache_dir = cache_dir

        self._cf: Optional[Crazyflie] = None
        self._lock = threading.Lock()

        self._state = np.zeros(13, dtype=np.float32)

        self._last_timestamp: float = 0.0

        # Connection + logging readiness flags
        self._connected = False
        self._ready = False

        self._got_posvel = False
        self._got_quat = False
        self._got_gyro = False

        # LogConfig handles
        self.lg_pos_vel: Optional[LogConfig] = None
        self.lg_quat: Optional[LogConfig] = None
        self.lg_gyro: Optional[LogConfig] = None


    def connect(self, timeout_s: float = 10.0) -> None:
        if self._cf is not None:
            raise RuntimeError("Already connected to Crazyflie")

        self._cf = Crazyflie(rw_cache=self._cache_dir)

        self._cf.connected.add_callback(self._on_connected)
        self._cf.connection_failed.add_callback(self._on_connection_failed)
        self._cf.connection_lost.add_callback(self._on_connection_lost)
        self._cf.disconnected.add_callback(self._on_disconnected)

        logger.info("Connecting to %s ...", self._uri)
        self._cf.open_link(self._uri)

        #Wait for connection
        t0 = time.time()
        while True:
            with self._lock:
                if self._connected:
                    break
            if time.time() - t0 >= timeout_s:
                try:
                    self._cf.close_link()
                except Exception:
                    pass
                raise TimeoutError("Timeout waiting for Crazyflie connection")
            time.sleep(0.05)

        #Wait for ALL log blocks
        t1 = time.time()
        while True:
            with self._lock:
                if self._ready:
                    break
            if time.time() - t1 >= timeout_s:
                try:
                    self._cf.close_link()
                except Exception:
                    pass
                raise TimeoutError("Timeout waiting for Crazyflie log data")
            time.sleep(0.05)

        logger.info("StateObserver ready: %s", self._ready)

    def close(self) -> None:
        """Stop logs and close link cleanly."""
        with self._lock:
            cf = self._cf

        if cf is None:
            return

        # Stop log configs if they exist
        for lg in [self.lg_pos_vel, self.lg_quat, self.lg_gyro]:
            try:
                if lg is not None:
                    lg.stop()
            except Exception:
                pass

        try:
            cf.close_link()
        except Exception:
            pass

        with self._lock:
            self._cf = None
            self._connected = False
            self._ready = False
            self._got_posvel = self._got_quat = self._got_gyro = False

    def get_state(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()

    def is_ready(self) -> bool:
        with self._lock:
            return bool(self._ready)

    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._connected)

    def last_update_age(self) -> float:
        with self._lock:
            if self._last_timestamp <= 0.0:
                return float("inf")
            return time.time() - self._last_timestamp


    def _on_connected(self, uri: str):
        logger.info("Connection established to %s", uri)

        with self._lock:
            self._connected = True
            self._ready = False
            self._got_posvel = self._got_quat = self._got_gyro = False

        self.lg_pos_vel = LogConfig(name="PosVel", period_in_ms=self.log_period_ms)
        self.lg_pos_vel.add_variable("stateEstimate.x", "float")
        self.lg_pos_vel.add_variable("stateEstimate.y", "float")
        self.lg_pos_vel.add_variable("stateEstimate.z", "float")
        self.lg_pos_vel.add_variable("stateEstimate.vx", "float")
        self.lg_pos_vel.add_variable("stateEstimate.vy", "float")
        self.lg_pos_vel.add_variable("stateEstimate.vz", "float")


        self.lg_quat = LogConfig(name="Quat", period_in_ms=self.log_period_ms)
        self.lg_quat.add_variable("stateEstimate.qw", "float")
        self.lg_quat.add_variable("stateEstimate.qx", "float")
        self.lg_quat.add_variable("stateEstimate.qy", "float")
        self.lg_quat.add_variable("stateEstimate.qz", "float")

        self.lg_gyro = LogConfig(name="Gyro", period_in_ms=self.log_period_ms)
        self.lg_gyro.add_variable("gyro.x", "float")
        self.lg_gyro.add_variable("gyro.y", "float")
        self.lg_gyro.add_variable("gyro.z", "float")

        try:
            assert self._cf is not None

            self._cf.log.add_config(self.lg_pos_vel)
            self._cf.log.add_config(self.lg_quat)
            self._cf.log.add_config(self.lg_gyro)

            # Data callbacks
            self.lg_pos_vel.data_received_cb.add_callback(self._on_log_data)
            self.lg_quat.data_received_cb.add_callback(self._on_log_data)
            self.lg_gyro.data_received_cb.add_callback(self._on_log_data)

            # Error callbacks (important)
            self.lg_pos_vel.error_cb.add_callback(self._on_log_error)
            self.lg_quat.error_cb.add_callback(self._on_log_error)
            self.lg_gyro.error_cb.add_callback(self._on_log_error)

            self.lg_pos_vel.start()
            self.lg_quat.start()
            self.lg_gyro.start()

        except KeyError as e:
            logger.error("Could not add log config, variable not found: %s", e)
        except AttributeError as e:
            logger.error("Could not add log config, bad configuration: %s", e)
        except Exception as e:
            logger.exception("Could not add log config: %s", e)

    def _on_connection_failed(self, uri: str, msg: str):
        logger.error("Connection to %s failed: %s", uri, msg)
        with self._lock:
            self._connected = False
            self._ready = False

    def _on_connection_lost(self, uri: str, msg: str):
        logger.warning("Connection to %s lost: %s", uri, msg)
        with self._lock:
            self._connected = False
            self._ready = False
            self._got_posvel = self._got_quat = self._got_gyro = False

    def _on_disconnected(self, uri: str):
        logger.info("Disconnected from %s", uri)
        with self._lock:
            self._connected = False
            self._ready = False
            self._got_posvel = self._got_quat = self._got_gyro = False

    def _on_log_error(self, logconf: LogConfig, msg: str):
        logger.error("[CF] Log error for %s: %s", logconf.name, msg)

    def _on_log_data(self, timestamp: int, data: Dict[str, Any], logconf: LogConfig):
        """
        Update only the part of the state that belongs to this log block.
        Order:
            [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
        """

        with self._lock:
            if logconf.name == "PosVel":
                self._state[0] = float(data.get("stateEstimate.x", self._state[0]))
                self._state[1] = float(data.get("stateEstimate.y", self._state[1]))
                self._state[2] = float(data.get("stateEstimate.z", self._state[2]))

                self._state[7] = float(data.get("stateEstimate.vx", self._state[7]))
                self._state[8] = float(data.get("stateEstimate.vy", self._state[8]))
                self._state[9] = float(data.get("stateEstimate.vz", self._state[9]))

                self._got_posvel = True

            elif logconf.name == "Quat":
                qw = float(data.get("stateEstimate.qw", 1.0))
                qx = float(data.get("stateEstimate.qx", 0.0))
                qy = float(data.get("stateEstimate.qy", 0.0))
                qz = float(data.get("stateEstimate.qz", 0.0))

                q = np.array([qw, qx, qy, qz], dtype=np.float32)
                norm_q = float(np.linalg.norm(q))
                if norm_q > 1e-6:
                    q /= norm_q

                self._state[3] = q[0]
                self._state[4] = q[1]
                self._state[5] = q[2]
                self._state[6] = q[3]

                self._got_quat = True

            elif logconf.name == "Gyro":
                wx_deg = float(data.get("gyro.x", 0.0))
                wy_deg = float(data.get("gyro.y", 0.0))
                wz_deg = float(data.get("gyro.z", 0.0))

                deg2rad = np.pi / 180.0
                self._state[10] = wx_deg * deg2rad
                self._state[11] = wy_deg * deg2rad
                self._state[12] = wz_deg * deg2rad

                self._got_gyro = True

            self._last_timestamp = time.time()

            self._ready = self._got_posvel and self._got_quat and self._got_gyro

    @property
    def cf(self) -> Crazyflie:
        """
        Safe accessor for the underlying Crazyflie object.

        Use this in scripts that need to send commands or add custom LogConfigs,
        instead of reaching into _cf directly.
        """
        with self._lock:
            if self._cf is None:
                raise RuntimeError("Crazyflie is not connected")
            return self._cf
