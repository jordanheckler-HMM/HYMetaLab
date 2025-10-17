#!/bin/bash
# OriginChain v5 Catalyst - Cron Runner
# Generates ≥1 hypothesis per day
#
# Add to crontab for daily execution:
# 0 9 * * * /path/to/catalyst_cron.sh
#
# This runs daily at 9:00 AM

cd "$(dirname "$0")"

# Generate 1-2 hypotheses daily
python3 generative_catalyst.py generate --count 2 --output genesis_queue.md

# Log execution
echo "$(date): Generated hypotheses" >> catalyst_cron.log

