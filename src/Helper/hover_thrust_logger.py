import os
import time
import csv
import logging
import threading

import numpy as np
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch
from cflib.crazyflie.log import LogConfig

from CrazyFlieStateObserver import CrazyFlieStateObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


URI = "radio://0/80/2M/E7E7E7E7E7"

LOG_PERIOD_MS = 10  # ms for thrust log (match observer)

# List of (target_height_m, hover_time_s) scenarios
HOVER_SCENARIOS = [
    (1.0, 6.0),
    (1.5, 6.0),
    (0.70, 6.0),
]

PRE_HOVER_WAIT_S = 1.0  # let estimator/logs settle before first hover

CSV_PATH = "crazyflie_hover_data.csv"

# Just used for sanity-clamping thrust *values we log* (not commands)
MIN_THRUST = 0
MAX_THRUST = 65535


csv_lock = threading.Lock()
csv_file = None
csv_writer = None

run_id = None
start_time = None

current_episode_idx = 0
current_mode = "idle"
current_target_z = 0.0

observer: CrazyFlieStateObserver | None = None


def ensure_csv_header(path: str):
    """Create file with header if it does not exist or is empty."""
    file_exists = os.path.exists(path)
    needs_header = True
    if file_exists and os.path.getsize(path) > 0:
        needs_header = False

    if needs_header:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = [
                "run_id",
                "episode",
                "mode",
                "target_z_m",
                "timestamp_cf_ms",
                "t_rel_s",
                "thrust_counts",
                # 13D state from CrazyFlieStateObserver:
                # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
                # derived quantity
                "speed",  # sqrt(vx^2 + vy^2 + vz^2)
            ]
            w.writerow(header)
        logger.info("Created CSV with header at %s", path)


def thrust_log_cb(timestamp, data, logconf):
    """
    Called from Crazyflie logging thread whenever 'stabilizer.thrust'
    is updated. We grab the latest 13D state from the observer and
    append a row to the CSV immediately, flushing as we go.
    """
    global csv_writer, csv_file, run_id, start_time
    global current_episode_idx, current_mode, current_target_z
    global observer

    if csv_writer is None or observer is None:
        return  # nothing to do

    # --- read thrust ---
    try:
        thrust_counts = int(data["stabilizer.thrust"])
    except Exception as e:
        logger.error("Error reading thrust from log: %s", e)
        return

    thrust_counts = max(MIN_THRUST, min(MAX_THRUST, thrust_counts))

    # --- get current state from observer ---
    try:
        state = observer.get_state()  # np.ndarray(13,)
    except Exception as e:
        logger.error("Error getting state from observer: %s", e)
        return

    if state is None or state.shape[0] != 13:
        return  # skip if incomplete

    x, y, z = state[0], state[1], state[2]
    qw, qx, qy, qz = state[3], state[4], state[5], state[6]
    vx, vy, vz = state[7], state[8], state[9]
    wx, wy, wz = state[10], state[11], state[12]

    speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

    t_rel = time.time() - start_time if start_time is not None else 0.0

    row = [
        run_id,
        current_episode_idx,
        current_mode,
        current_target_z,
        timestamp,   # Crazyflie log timestamp (ms)
        t_rel,
        thrust_counts,
        x,
        y,
        z,
        qw,
        qx,
        qy,
        qz,
        vx,
        vy,
        vz,
        wx,
        wy,
        wz,
        speed,
    ]

    # Thread-safe write + flush, so data is persisted "as we go"
    with csv_lock:
        try:
            csv_writer.writerow(row)
            csv_file.flush()
        except Exception as e:
            logger.error("Error writing/flush row to CSV: %s", e)


def thrust_log_error_cb(msg):
    logger.error("[ThrustLog] Error: %s", msg)


def main():
    global csv_file, csv_writer, run_id, start_time
    global current_episode_idx, current_mode, current_target_z
    global observer

    thrust_logconf: LogConfig | None = None

    try:
        ensure_csv_header(CSV_PATH)
        csv_file = open(CSV_PATH, "a", newline="")
        csv_writer = csv.writer(csv_file)

        run_id = int(time.time()) 
        start_time = time.time()

        cflib.crtp.init_drivers(enable_debug_driver=False)

        observer = CrazyFlieStateObserver(URI, log_period_ms=LOG_PERIOD_MS)
        logger.info("Connecting CrazyFlieStateObserver ...")
        observer.connect(timeout_s=10.0)

        logger.info("Observer connected: %s", observer.is_connected())
        logger.info("Observer ready:     %s", observer.is_ready())
        logger.info("Initial state age: %.3f s", observer.last_update_age())

        cf = observer.cf

        thrust_logconf = LogConfig(name="Thrust", period_in_ms=LOG_PERIOD_MS)
        thrust_logconf.add_variable("stabilizer.thrust", "uint16_t")

        cf.log.add_config(thrust_logconf)
        thrust_logconf.data_received_cb.add_callback(thrust_log_cb)
        thrust_logconf.error_cb.add_callback(thrust_log_error_cb)

        logger.info("Starting thrust log ...")
        thrust_logconf.start()

        current_mode = "pre_hover"
        current_target_z = 0.0
        logger.info("Waiting %.1f s before first hover ...", PRE_HOVER_WAIT_S)
        time.sleep(PRE_HOVER_WAIT_S)

        dt = 0.1  # seconds between hover setpoints

        for idx, (target_z, hover_time_s) in enumerate(HOVER_SCENARIOS, start=1):
            current_episode_idx = idx
            current_mode = "hover"
            current_target_z = float(target_z)

            logger.info(
                "Episode %d: hover at %.2f m for %.1f s",
                current_episode_idx,
                current_target_z,
                hover_time_s,
            )

            steps = int(hover_time_s / dt)
            for _ in range(steps):
                cf.commander.send_hover_setpoint(
                    0.0,                # vx
                    0.0,                # vy
                    0.0,                # yawrate
                    current_target_z,   # z (m)
                )
                time.sleep(dt)

            # small pause between episodes
            current_mode = "between_episodes"
            logger.info("Episode %d done. Short pause ...", current_episode_idx)
            time.sleep(1.0)

        # After all scenarios, stop hover setpoints
        logger.info("All hover scenarios completed. Stopping hover setpoint.")
        cf.commander.send_stop_setpoint()
        time.sleep(0.5)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        try:
            if observer is not None and observer.cf is not None:
                observer.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    except Exception as e:
        logger.error("Unexpected error in main: %s", e)
        try:
            if observer is not None and observer.cf is not None:
                observer.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    finally:
        try:
            if thrust_logconf is not None:
                logger.info("Stopping thrust log (finally)...")
                try:
                    thrust_logconf.stop()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if observer is not None:
                logger.info("Closing observer (finally)...")
                observer.close()
        except Exception:
            pass

        # Close CSV safely
        try:
            if csv_file is not None:
                logger.info("Closing CSV file ...")
                csv_file.close()
        except Exception:
            pass

        # Always power-cycle like your other scripts
        try:
            logger.info("Forcing STM power cycle in finally ...")
            PowerSwitch(URI).stm_power_cycle()
            time.sleep(1.0)
        except Exception as e:
            logger.error("STM power cycle failed: %s", e)

        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
