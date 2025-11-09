#!/usr/bin/env python3
"""
AutoLab ↔ CMV ↔ Guardian Feedback Loop

Reads Guardian Field Index (GFI) from CMV summaries
and adjusts AutoLab exploration rate η accordingly.

Strategy:
  - GFI < 85 (below target) → increase η (explore more)
  - GFI ≥ 85 (at/above target) → decrease η (exploit more)

This creates a self-regulating system where AutoLab automatically
adjusts its exploration strategy based on Guardian validation scores.
"""

import json
import csv
import os
import time
from pathlib import Path

FIELD_INDEX_PATH = "HYMetaLab_CMVDataset_v1/summaries/field_index.csv"
AUTO_KB_PATH = "autolab/knowledge.json"
BASE_ETA = 0.2  # Default exploration rate
MIN_ETA = 0.05  # Minimum exploration (pure exploitation)
MAX_ETA = 1.0   # Maximum exploration (pure exploration)

def read_field_index():
    """
    Read the latest Guardian Field Index from CMV summaries.
    
    Returns:
        float: Latest GFI score, or None if unavailable
    """
    try:
        path = Path(FIELD_INDEX_PATH)
        if not path.exists():
            print(f"[Feedback] Warning: {FIELD_INDEX_PATH} not found")
            return None
        
        with open(path) as f:
            rows = list(csv.DictReader(f))
            if not rows:
                print("[Feedback] Warning: field_index.csv is empty")
                return None
            
            # Get most recent entry
            last = rows[-1]
            gfi = float(last.get("guardian_field_index", last.get("GFI", 0)))
            
            if gfi == 0:
                print("[Feedback] Warning: GFI is 0, may be invalid")
                return None
            
            return gfi
    except FileNotFoundError:
        print(f"[Feedback] Warning: {FIELD_INDEX_PATH} not found")
        return None
    except Exception as e:
        print(f"[Feedback] Error reading field index: {e}")
        return None

def update_eta(last_gfi, base_eta=BASE_ETA):
    """
    Calculate new exploration rate based on Guardian Field Index.
    
    Args:
        last_gfi: Latest Guardian Field Index score (or None)
        base_eta: Baseline exploration rate
    
    Returns:
        float: New exploration rate η
    
    Strategy:
        - GFI < 85: Below Guardian threshold → explore more (η↑ by 20%)
        - GFI ≥ 85: At/above threshold → exploit more (η↓ by 10%)
        - No GFI: Use base rate (safe default)
    """
    if last_gfi is None:
        print(f"[Feedback] No GFI available, using base η={base_eta:.3f}")
        return base_eta
    
    if last_gfi < 85:  # Below Guardian PASS threshold
        # Increase exploration to find better hypotheses
        new_eta = min(base_eta * 1.2, MAX_ETA)
        print(f"[Feedback] GFI={last_gfi:.1f} < 85 → explore more (η↑)")
    else:  # At or above threshold
        # Decrease exploration, exploit known good hypotheses
        new_eta = max(base_eta * 0.9, MIN_ETA)
        print(f"[Feedback] GFI={last_gfi:.1f} ≥ 85 → exploit more (η↓)")
    
    return new_eta

def apply_feedback():
    """
    Main feedback loop: Read GFI → Update η → Save to knowledge base.
    
    Updates autolab/knowledge.json with:
        - last_gfi: Latest Guardian Field Index
        - eta: New exploration rate
        - timestamp: When feedback was applied
    """
    kb_path = Path(AUTO_KB_PATH)
    
    # Load existing knowledge base
    if kb_path.exists():
        with open(kb_path) as f:
            kb = json.load(f)
    else:
        # Create minimal KB if it doesn't exist
        kb = {"runs": [], "hypotheses": {}}
    
    # Get current GFI and calculate new η
    last_gfi = read_field_index()
    
    # Get current η from meta if it exists, otherwise use base
    current_eta = kb.get("meta", {}).get("eta", BASE_ETA)
    new_eta = update_eta(last_gfi, base_eta=current_eta)
    
    # Update metadata
    kb["meta"] = {
        "last_gfi": last_gfi,
        "eta": new_eta,
        "timestamp": time.time(),
        "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "feedback_iterations": kb.get("meta", {}).get("feedback_iterations", 0) + 1
    }
    
    # Save updated knowledge base
    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2)
    
    print(f"[Feedback] GFI={last_gfi} → η={new_eta:.3f} (saved to {AUTO_KB_PATH})")
    print(f"[Feedback] Iteration #{kb['meta']['feedback_iterations']}")
    
    return new_eta

def get_current_eta():
    """
    Get current exploration rate from knowledge base.
    
    Returns:
        float: Current η, or BASE_ETA if not set
    """
    kb_path = Path(AUTO_KB_PATH)
    if not kb_path.exists():
        return BASE_ETA
    
    try:
        with open(kb_path) as f:
            kb = json.load(f)
        return kb.get("meta", {}).get("eta", BASE_ETA)
    except Exception:
        return BASE_ETA

if __name__ == "__main__":
    print("═" * 79)
    print("  AutoLab ↔ CMV ↔ Guardian Feedback Loop")
    print("═" * 79)
    print()
    
    new_eta = apply_feedback()
    
    print()
    print("✅ Feedback loop complete")
    print(f"   New exploration rate: η={new_eta:.3f}")
    print()
    print("═" * 79)

