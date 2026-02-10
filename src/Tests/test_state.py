import time
import numpy as np
import logging

import cflib.crtp
from cflib.utils.power_switch import PowerSwitch

from CrazyFlieStateObserver import CrazyFlieStateObserver


logging.basicConfig(level=logging.INFO)

# Adjust for your setup
URI = "radio://0/80/2M/E7E7E7E7E7"

LOG_PERIOD_MS = 10
TEST_DURATION_S = 10.0

PRINT_HZ = 10
SEND_SLEEP = 0.01

# Warn if no updates recently
MAX_ALLOWED_STALE_S = 0.25


def pretty_state(s):
    # Layout: [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
    return {
        "pos": s[0:3],
        "quat": s[3:7],
        "vel": s[7:10],
        "omega": s[10:13],
    }


def sanity_checks(s):
    issues = []

    q = s[3:7]
    qn = float(np.linalg.norm(q))
    if not (0.90 <= qn <= 1.10):
        issues.append(f"quat_norm={qn:.3f}")

    if np.isnan(s[0]) or np.isnan(s[1]) or np.isnan(s[2]):
        issues.append("pos contains NaN")

    wmag = float(np.linalg.norm(s[10:13]))
    if wmag > 10.0:
        issues.append(f"high |w|={wmag:.2f} rad/s")

    return issues, qn, wmag


def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    observer = None

    try:
        observer = CrazyFlieStateObserver(URI, log_period_ms=LOG_PERIOD_MS)

        observer.connect(timeout_s=10.0)

        print("\nObserver connected:", observer.is_connected())
        print("Observer ready     :", observer.is_ready())
        print("Starting 13D logging validation...\n")

        start = time.time()
        next_print = start
        print_dt = 1.0 / PRINT_HZ

        samples = 0
        stale_warnings = 0

        while time.time() - start < TEST_DURATION_S:
            age = observer.last_update_age()
            if age > MAX_ALLOWED_STALE_S:
                stale_warnings += 1
                print(f"[WARN] Log stale age={age:.3f}s")

            now = time.time()
            if now >= next_print:
                s = observer.get_state()
                info = pretty_state(s)
                issues, qn, wmag = sanity_checks(s)

                x, y, z = info["pos"]
                qw, qx, qy, qz = info["quat"]
                vx, vy, vz = info["vel"]
                wx, wy, wz = info["omega"]

                print(
                    f"t={now - start:5.2f}s | "
                    f"pos=({x: .3f},{y: .3f},{z: .3f}) | "
                    f"vel=({vx: .3f},{vy: .3f},{vz: .3f}) | "
                    f"quat=({qw: .3f},{qx: .3f},{qy: .3f},{qz: .3f}) | "
                    f"|q|={qn:.3f} | "
                    f"omega=({wx: .3f},{wy: .3f},{wz: .3f}) | "
                    f"|w|={wmag:.3f} | age={age:.3f}s"
                )

                if issues:
                    print("   issues:", "; ".join(issues))

                samples += 1
                next_print += print_dt

            time.sleep(SEND_SLEEP)

        print("\n--- Summary ---")
        print(f"Samples printed: {samples}")
        print(f"Stale warnings : {stale_warnings}")
        print("Done.\n")

    except TimeoutError as e:
        print("Timeout:", e)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print("Unexpected error:", e)

    finally:
        # Cleanly close observer if it exists
        try:
            if observer is not None:
                observer.close()
        except Exception:
            pass

        # ALWAYS power-cycle 
        try:
            print("Forcing STM power cycle in finally...")
            PowerSwitch(URI).stm_power_cycle()
            time.sleep(1.0)
        except Exception as e:
            print("STM power cycle failed:", e)


if __name__ == "__main__":
    main()
