import time
import logging
import statistics

import numpy as np
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch
from cflib.crazyflie.log import LogConfig

from CrazyFlieStateObserver import CrazyFlieStateObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




URI = "radio://0/80/2M/E7E7E7E7E7"


M_REAL_KG = 0.033  ##real crazyflie mass in kg
G = 9.81
W_NEWTON = M_REAL_KG * G

LOG_PERIOD_MS = 10

HOVER_HEIGHT_M = 0.5     
HOVER_TIME_S = 7.0     
PRE_HOVER_WAIT_S = 1.0   ##wait time before hover to let things settle


DISCARD_SAMPLES = 50

MIN_THRUST = 10001
MAX_THRUST = 60000


def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    observer = None
    thrust_logconf = None
    thrust_samples = []

    try:
        if M_REAL_KG <= 0:
            raise ValueError("M_REAL_KG must be > 0. Please set your drone mass in kg.")

        logger.info("Weight W = %.4f N (m=%.4f kg)", W_NEWTON, M_REAL_KG)
        observer = CrazyFlieStateObserver(URI, log_period_ms=LOG_PERIOD_MS)
        observer.connect(timeout_s=10.0)
        logger.info("Observer connected: %s", observer.is_connected())
        logger.info("Observer ready:     %s", observer.is_ready())
        age = observer.last_update_age()
        logger.info("Initial state age: %.3f s", age)
        cf = observer.cf 
        thrust_logconf = LogConfig(name="Thrust", period_in_ms=LOG_PERIOD_MS)
        thrust_logconf.add_variable("stabilizer.thrust", "uint16_t")
        
        def thrust_log_cb(timestamp, data, logconf):
            thrust_samples.append(int(data["stabilizer.thrust"]))

        def thrust_log_error_cb(msg):
            logger.error("[ThrustLog] Error: %s", msg)

        cf.log.add_config(thrust_logconf)
        thrust_logconf.data_received_cb.add_callback(thrust_log_cb)
        thrust_logconf.error_cb.add_callback(thrust_log_error_cb)

        logger.info("Starting thrust log...")
        thrust_logconf.start()

        logger.info("Waiting %.1f s before hover ...", PRE_HOVER_WAIT_S)
        time.sleep(PRE_HOVER_WAIT_S)


        logger.info("Hovering at %.2f m for %.1f s ...", HOVER_HEIGHT_M, HOVER_TIME_S)

        dt = 0.1
        steps = int(HOVER_TIME_S / dt)
        start_hover = time.time()

        for _ in range(steps):
            cf.commander.send_hover_setpoint(
                0.0,  # vx
                0.0,  # vy
                0.0,  # yawrate
                HOVER_HEIGHT_M
            )
            time.sleep(dt)

        # Stop sending hover setpoints
        logger.info("Stopping hover setpoint...")
        cf.commander.send_stop_setpoint()
        time.sleep(0.5)


        logger.info("Stopping thrust log...")
        thrust_logconf.stop()


        total_samples = len(thrust_samples)
        if total_samples <= DISCARD_SAMPLES:
            logger.error(
                "Not enough thrust samples collected (%d). "
                "Try increasing HOVER_TIME_S.",
                total_samples,
            )
            return

        steady_samples = thrust_samples[DISCARD_SAMPLES:]
        u_hover = statistics.mean(steady_samples)

        logger.info("-------------------------------------------------")
        logger.info("Mass (M_REAL_KG):           %.4f kg", M_REAL_KG)
        logger.info("Weight W:                   %.4f N", W_NEWTON)
        logger.info("Thrust samples collected:   %d", total_samples)
        logger.info("Used samples (after discard): %d", len(steady_samples))
        logger.info("u_hover (mean thrust cmd):  %.2f counts", u_hover)
        logger.info("-------------------------------------------------")

        counts_per_newton = u_hover / W_NEWTON
        logger.info("Counts per Newton (local linear around hover): %.2f", counts_per_newton)

        print("\n=== Calibration Result ===")
        print(f"Drone mass (kg):            {M_REAL_KG:.4f}")
        print(f"Weight W (N):               {W_NEWTON:.4f}")
        print(f"Mean hover thrust u_hover:  {u_hover:.2f} counts")
        print(f"Counts per Newton:          {counts_per_newton:.2f}")
        print()

        print("Use this mapping from MuJoCo thrust in N -> Crazyflie 16-bit thrust:")
        print()
        print("    COUNTS_PER_NEWTON = %.8f" % counts_per_newton)
        print("    MIN_THRUST = 10001")
        print("    MAX_THRUST = 60000")
        print()
        print("    def thrust_N_to_counts(T):")
        print("        u = T * COUNTS_PER_NEWTON")
        print("        u = max(MIN_THRUST, min(MAX_THRUST, u))")
        print("        return int(u)")
        print()
        print("Then in your real-flight evaluation:")
        print("    thrust_N = action_from_policy  # MuJoCo-style thrust in Newtons")
        print("    thrust_cmd = thrust_N_to_counts(thrust_N)")
        print("    cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust_cmd)")
        print("===========================\n")

    except TimeoutError as e:
        print("Timeout:", e)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print("Unexpected error:", e)

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

        try:
            print("Forcing STM power cycle in finally...")
            PowerSwitch(URI).stm_power_cycle()
            time.sleep(1.0)
        except Exception as e:
            print("STM power cycle failed:", e)


if __name__ == "__main__":
    main()
