import os
import json
import io
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import fastavro
import pandas as pd
from confluent_kafka import Consumer

from superphot_boom_lsst import run_superphot, post_to_fritz_with_replace

# --- Configuration ---
NIGHT_START_HOUR = 18  # 6 PM local = start of observing night
LOCAL_TZ = ZoneInfo("US/Central")
ALERT_TRIGGER_THRESHOLD = 8
MAX_RUNS_PER_NIGHT = 3
CSV_FILE = "superphot_results_lsst.csv"
LOG_FILE = "superphot_lsst.log"
STATE_FILE = "consumer_state_lsst.json"
KAFKA_TOPIC = "LSST_alerts_results"
FILTER_NAME = "superphot_lsst"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class SourceTracker:
    """Tracks alert counts and superphot runs for a single source within one night."""
    alert_count: int = 0
    total_alerts: int = 0
    run_count: int = 0
    last_fritz_comment_id: int | None = None


def save_state(source_trackers, current_night_start):
    """Persist source trackers to disk so state survives restarts."""
    state = {
        "night_start": current_night_start.isoformat() if current_night_start else None,
        "trackers": {
            src_id: {
                "alert_count": t.alert_count,
                "total_alerts": t.total_alerts,
                "run_count": t.run_count,
                "last_fritz_comment_id": t.last_fritz_comment_id,
            }
            for src_id, t in source_trackers.items()
        },
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def load_state():
    """Load source trackers from disk. Returns (source_trackers, night_start) or empty defaults."""
    if not os.path.exists(STATE_FILE):
        return {}, None
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        night_start = (
            datetime.fromisoformat(state["night_start"]) if state.get("night_start") else None
        )
        trackers = {
            src_id: SourceTracker(**vals)
            for src_id, vals in state.get("trackers", {}).items()
        }
        logger.info("Restored state: %d sources from night %s", len(trackers), night_start)
        return trackers, night_start
    except Exception:
        logger.exception("Failed to load state file, starting fresh")
        return {}, None


def get_current_night_start():
    """Return the datetime of the start of the current observing night."""
    now = datetime.now(LOCAL_TZ)
    night_boundary_today = now.replace(
        hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0
    )
    if now < night_boundary_today:
        return night_boundary_today - timedelta(days=1)
    return night_boundary_today


def check_night_rollover(source_trackers, current_night_start):
    """Check if the night has rolled over; if so, reset all trackers and state file."""
    new_night_start = get_current_night_start()
    if current_night_start is None or new_night_start > current_night_start:
        logger.info("New night started at %s. Resetting all source trackers.", new_night_start)
        source_trackers.clear()
        current_night_start = new_night_start
        save_state(source_trackers, current_night_start)
    return current_night_start, source_trackers


def should_run_superphot(tracker):
    """Determine if superphot should run for this source."""
    if tracker.run_count >= MAX_RUNS_PER_NIGHT:
        return False
    if tracker.run_count == 0:
        return True
    if tracker.alert_count >= ALERT_TRIGGER_THRESHOLD:
        return True
    return False


def read_avro(msg):
    """Reads an Avro record from a Kafka message."""
    bytes_io = io.BytesIO(msg.value())
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record
    return None


thumbnail_types = [
    ("cutoutScience", "new"),
    ("cutoutTemplate", "ref"),
    ("cutoutDifference", "sub"),
]

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'umn_boom_kafka_consumer_group_superphot_lsst',
    'auto.offset.reset': 'earliest',
    "enable.auto.commit": False,
    "session.timeout.ms": 6000,
    "max.poll.interval.ms": 300000,
    "security.protocol": "PLAINTEXT",
})
consumer.subscribe([KAFKA_TOPIC])
logger.info("Subscribed to topic: %s", KAFKA_TOPIC)


def consume():
    logger.info("Listening for messages...")
    total_consumed = 0
    header_written = os.path.exists(CSV_FILE)
    source_trackers, current_night_start = load_state()

    try:
        while True:
            current_night_start, source_trackers = check_night_rollover(
                source_trackers, current_night_start
            )

            msg = consumer.poll(timeout=10.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            record = read_avro(msg)

            for cutout_type, _ in thumbnail_types:
                del record[cutout_type]

            if total_consumed == 0:
                with open("first_alert_lsst.json", "w") as f:
                    json.dump(record, f, indent=2)

            lsst_id = record["objectId"]
            passes_filter = any(
                FILTER_NAME in f["filter_name"] for f in record["filters"]
            )

            if passes_filter:
                if lsst_id not in source_trackers:
                    source_trackers[lsst_id] = SourceTracker()

                tracker = source_trackers[lsst_id]
                tracker.alert_count += 1
                tracker.total_alerts += 1

                logger.info(
                    "[%s] Alert #%d this night (+%d since last run, %d/%d runs)",
                    lsst_id, tracker.total_alerts, tracker.alert_count,
                    tracker.run_count, MAX_RUNS_PER_NIGHT,
                )

                if should_run_superphot(tracker):
                    logger.info("[%s] Triggering superphot run #%d", lsst_id, tracker.run_count + 1)
                    try:
                        result = run_superphot(lsst_id)
                    except Exception:
                        logger.exception("[%s] run_superphot failed", lsst_id)
                        result = None

                    if result is not None:
                        event_dict, image_path = result
                        if event_dict is not None:
                            event_dict['result_timestamp'] = datetime.now(LOCAL_TZ).isoformat()

                            row = pd.DataFrame([event_dict])
                            row.to_csv(CSV_FILE, mode="a", index=False, header=not header_written)
                            header_written = True
                            logger.info("[%s] Result saved to %s", lsst_id, CSV_FILE)

                            try:
                                comment_id = post_to_fritz_with_replace(
                                    event_dict, image_path, lsst_id,
                                    previous_comment_id=tracker.last_fritz_comment_id,
                                )
                                tracker.last_fritz_comment_id = comment_id
                                logger.info("[%s] Fritz comment posted: %s", lsst_id, comment_id)
                            except Exception:
                                logger.exception("[%s] Fritz posting failed", lsst_id)
                    else:
                        logger.warning("[%s] run_superphot returned no result", lsst_id)

                    tracker.alert_count = 0
                    tracker.run_count += 1
                    save_state(source_trackers, current_night_start)
            else:
                logger.debug("Alert %d didn't pass %s", total_consumed, FILTER_NAME)

            total_consumed += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages. Tracked %d unique sources this night.",
                     total_consumed, len(source_trackers))
        consumer.close()


consume()
