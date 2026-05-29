"""
BasicHoverTest.py — Minimal sanity-check hover for the real Crazyflie.

This is a non-RL, non-policy test. It uses cflib's MotionCommander to do
a simple "take off, hover for N seconds, land" sequence. Use this BEFORE
running RL evaluation to confirm:
  - Radio link works
  - Motors arm correctly
  - State estimation is good enough for stable hover (Loco/Lighthouse/Flow)
  - Battery is healthy
  - The drone is physically OK (no broken propellers, motors balanced)

If THIS script can't hover, the RL eval will not work either — the problem
is hardware/positioning, not the model.

The MotionCommander uses the Crazyflie's onboard PID and high-level
commander, not our 4D commander interface. It's about as bulletproof as
flight gets — if this fails, fix that before touching the RL stack.

Safety guarantees (carried from EvaluationVelocityReal.py):
  - Power-cycles STM32 on exit no matter what, even on KeyboardInterrupt
    or unexpected exception
  - Closes cflib drivers cleanly
  - MotionCommander context manager auto-lands on exit if still flying
"""

import logging
import sys
import time
import traceback

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.power_switch import PowerSwitch


# ─────────────────────────────────────────────────────────────────────────────
# Logging — verbose so you can see what's happening
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hover_test")
logging.getLogger("cflib").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — adjust to match your drone
# ─────────────────────────────────────────────────────────────────────────────
URI = "radio://0/80/2M/E7E7E7E703"   # Crazyflie radio URI

HOVER_HEIGHT_M       = 1.0    # meters above takeoff point
HOVER_DURATION_S     = 5.0    # seconds to hover at altitude
TAKEOFF_VELOCITY_MS  = 0.5    # m/s during ascent
LANDING_VELOCITY_MS  = 0.3    # m/s during descent

# Cache directory for radio param TOC (created automatically)
RW_CACHE = "./cache"


# ─────────────────────────────────────────────────────────────────────────────
# Hover sequence
# ─────────────────────────────────────────────────────────────────────────────
def hover_sequence(scf: SyncCrazyflie):
    """
    Take off, hover for HOVER_DURATION_S, then land.
    MotionCommander handles all the velocity-control internals.
    """
    logger.info("Entering MotionCommander context...")
    with MotionCommander(scf, default_height=HOVER_HEIGHT_M) as mc:
        # `take_off` is implicit — entering the context climbs to default_height
        # at the default takeoff velocity. Log progress.
        logger.info("Takeoff command issued — climbing to %.2f m", HOVER_HEIGHT_M)

        # Brief wait for the takeoff transient to settle
        time.sleep(2.0)
        logger.info("Reached hover altitude. Holding for %.1f seconds...",
                    HOVER_DURATION_S)

        # Hover — MotionCommander.stop() holds position internally, no need
        # to send setpoints. We just sleep and let it run.
        t_start = time.time()
        while time.time() - t_start < HOVER_DURATION_S:
            elapsed = time.time() - t_start
            if int(elapsed) != int(elapsed - 0.1):  # rough ~1 Hz log
                logger.info("Hovering... %.1f / %.1f s",
                            elapsed, HOVER_DURATION_S)
            time.sleep(0.1)

        logger.info("Hover complete. Initiating landing...")
        # Exiting the context calls land() automatically, but call explicitly
        # for clarity and slightly better control over descent rate
        mc.land(velocity=LANDING_VELOCITY_MS)
        logger.info("Landing command sent.")

    logger.info("MotionCommander exited.")


# ─────────────────────────────────────────────────────────────────────────────
# Main with bulletproof cleanup
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 78)
    logger.info("CrazyFlie BASIC HOVER TEST  —  no RL, no policy, just MotionCommander")
    logger.info("=" * 78)
    logger.info("URI:               %s", URI)
    logger.info("Hover height:      %.2f m", HOVER_HEIGHT_M)
    logger.info("Hover duration:    %.1f s", HOVER_DURATION_S)
    logger.info("Takeoff velocity:  %.2f m/s", TAKEOFF_VELOCITY_MS)
    logger.info("Landing velocity:  %.2f m/s", LANDING_VELOCITY_MS)
    logger.info("=" * 78)

    logger.info("Initializing cflib drivers...")
    cflib.crtp.init_drivers(enable_debug_driver=False)

    try:
        logger.info("Opening SyncCrazyflie link to %s...", URI)
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=RW_CACHE)) as scf:
            logger.info("Link established. Running hover sequence...")
            hover_sequence(scf)
        logger.info("Hover test completed successfully.")

    except KeyboardInterrupt:
        # If the user cuts mid-flight the MotionCommander context manager
        # will already attempt landing on exit, but we log the interrupt.
        logger.warning("Interrupted by user (Ctrl+C). MotionCommander auto-lands on exit.")

    except Exception as e:
        logger.error("Hover test failed: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())

    finally:
        # ── STM power-cycle to clear any stuck state ──────────────────────────
        # CRITICAL: even if everything went smoothly, doing this prevents
        # the next session from inheriting a bad commander state. Always run.
        try:
            logger.info("Forcing STM32 power cycle on the Crazyflie...")
            PowerSwitch(URI).stm_power_cycle()
            time.sleep(1.0)
            logger.info("STM power cycle complete.")
        except Exception as e:
            logger.warning("STM power cycle failed: %s", e)

        # ── Close cflib drivers ───────────────────────────────────────────────
        try:
            cflib.crtp.close_all()
        except Exception:
            pass

        logger.info("=" * 78)
        logger.info("Cleanup complete. Drone should be sitting safely on the ground.")
        logger.info("=" * 78)


if __name__ == "__main__":
    main()