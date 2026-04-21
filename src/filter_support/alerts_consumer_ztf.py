import os
import json
import io
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import fastavro
import pandas as pd
from confluent_kafka import Consumer

from superphot_boom_ztf import run_superphot, post_to_fritz_with_replace, annotate_fritz

# --- Configuration ---
CSV_FILE = "superphot_results_ztf.csv"
LOG_FILE = "superphot_ztf.log"
STATE_FILE = "consumer_state_ztf.json"
KAFKA_TOPIC = "ZTF_alerts_results"
FILTER_NAME = "superphot_ztf"

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
    'group.id': 'umn_boom_kafka_consumer_group_superphot_ztf',
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
    consecutive_empty_polls = 0
    header_written = os.path.exists(CSV_FILE)
    source_trackers = load_state()

    try:
        while True:
            msg = consumer.poll(timeout=10.0)
            if msg is None:
                consecutive_empty_polls += 1
                if consecutive_empty_polls % 6 == 1:
                    logger.info("No new messages (idle for ~%ds, consumed %d so far)",
                                consecutive_empty_polls * 10, total_consumed)
                continue
            consecutive_empty_polls = 0
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            logger.debug("Received message: topic=%s partition=%s offset=%s",
                         msg.topic(), msg.partition(), msg.offset())

            record = read_avro(msg)
            if record is None:
                logger.error("Failed to deserialize Avro message at offset %s", msg.offset())
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            for cutout_type, _ in thumbnail_types:
                del record[cutout_type]

            if total_consumed == 0:
                with open("first_alert_ztf.json", "w") as f:
                    json.dump(record, f, indent=2)

            ztf_id = record["objectId"]
            passes_filter = any(
                FILTER_NAME in f["filter_name"] for f in record["filters"]
            )

            if passes_filter:
                if ztf_id not in source_trackers:
                    source_trackers[ztf_id] = SourceTracker()
                    logger.info("[%s] New source, now tracking %d unique sources", ztf_id, len(source_trackers))

                tracker = source_trackers[ztf_id]
                tracker.total_alerts += 1

                logger.info("[%s] Alert #%d", ztf_id, tracker.total_alerts)

                logger.info("[%s] Triggering superphot run", ztf_id)
                try:
                    result = run_superphot(ztf_id)
                except Exception:
                    logger.exception("[%s] run_superphot failed", ztf_id)
                    result = None

                if result is not None:
                    event_dict, image_path = result
                    if event_dict is None:
                        logger.warning("[%s] run_superphot returned empty classification", ztf_id)
                    if event_dict is not None:
                        event_dict['result_timestamp'] = datetime.now(timezone.utc).isoformat()

                        row = pd.DataFrame([event_dict])
                        row.to_csv(CSV_FILE, mode="a", index=False, header=not header_written)
                        header_written = True
                        logger.info("[%s] Result saved to %s", ztf_id, CSV_FILE)

                        try:
                            comment_id = post_to_fritz_with_replace(
                                event_dict, image_path, ztf_id,
                                previous_comment_id=tracker.last_fritz_comment_id,
                            )
                            tracker.last_fritz_comment_id = comment_id
                            logger.info("[%s] Fritz comment posted: %s", ztf_id, comment_id)
                        except Exception:
                            logger.exception("[%s] Fritz posting failed", ztf_id)

                        try:
                            annotation_id = annotate_fritz(
                                event_dict, ztf_id,
                                previous_annotation_id=tracker.last_annotation_id,
                            )
                            tracker.last_annotation_id = annotation_id
                            logger.info("[%s] Fritz annotation saved: %s", ztf_id, annotation_id)
                        except Exception:
                            logger.exception("[%s] Fritz annotation failed", ztf_id)
                else:
                    logger.warning("[%s] run_superphot returned no result", ztf_id)

                save_state(source_trackers)
                logger.debug("[%s] State saved to %s", ztf_id, STATE_FILE)
            else:
                logger.debug("Alert %d didn't pass %s", total_consumed, FILTER_NAME)

            total_consumed += 1
            consumer.commit(message=msg)
            logger.debug("Committed offset %s (total consumed: %d)", msg.offset(), total_consumed)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages. Tracked %d unique sources.",
                     total_consumed, len(source_trackers))
        consumer.close()


consume()
