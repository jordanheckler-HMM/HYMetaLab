import json
import logging
import os
import sys
import time
import uuid

RUN_ID = os.environ.get("RUN_ID", f"{int(time.time())}-{uuid.uuid4().hex[:6]}")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lvl": record.levelname,
            "run_id": RUN_ID,
            "msg": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(base, ensure_ascii=False)


def get_logger(name="metolab"):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    lg = logging.getLogger(name)
    lg.handlers = []
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg
