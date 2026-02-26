import os
import json
import io
import logging
from datetime import datetime
from dataclasses import dataclass

import fastavro
import pandas as pd
from confluent_kafka import Consumer

from superphot_boom_lsst import run_superphot, post_to_fritz_with_replace, annotate_fritz

# --- Configuration ---
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
    """Tracks alert counts and Fritz IDs for a single source."""
    total_alerts: int = 0
    last_fritz_comment_id: int | None = None
    last_annotation_id: int | None = None


def save_state(source_trackers):
    """Persist source trackers to disk so state survives restarts."""
    state = {
        "trackers": {
            src_id: {
                "total_alerts": t.total_alerts,
                "last_fritz_comment_id": t.last_fritz_comment_id,
                "last_annotation_id": t.last_annotation_id,
            }
            for src_id, t in source_trackers.items()
        },
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def load_state():
    """Load source trackers from disk. Returns source_trackers dict or empty default."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        trackers = {
            src_id: SourceTracker(**vals)
            for src_id, vals in state.get("trackers", {}).items()
        }
        logger.info("Restored state: %d sources", len(trackers))
        return trackers
    except Exception:
        logger.exception("Failed to load state file, starting fresh")
        return {}


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
    source_trackers = load_state()

    try:
        while True:
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
                tracker.total_alerts += 1

                logger.info("[%s] Alert #%d", lsst_id, tracker.total_alerts)

                logger.info("[%s] Triggering superphot run", lsst_id)
                try:
                    result = run_superphot(lsst_id)
                except Exception:
                    logger.exception("[%s] run_superphot failed", lsst_id)
                    result = None

                if result is not None:
                    event_dict, image_path = result
                    if event_dict is not None:
                        event_dict['result_timestamp'] = datetime.now(datetime.timezone.utc).isoformat()

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

                        try:
                            annotation_id = annotate_fritz(
                                event_dict, lsst_id,
                                previous_annotation_id=tracker.last_annotation_id,
                            )
                            tracker.last_annotation_id = annotation_id
                            logger.info("[%s] Fritz annotation saved: %s", lsst_id, annotation_id)
                        except Exception:
                            logger.exception("[%s] Fritz annotation failed", lsst_id)
                else:
                    logger.warning("[%s] run_superphot returned no result", lsst_id)

                save_state(source_trackers)
            else:
                logger.debug("Alert %d didn't pass %s", total_consumed, FILTER_NAME)

            total_consumed += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages. Tracked %d unique sources.",
                     total_consumed, len(source_trackers))
        consumer.close()


consume()
