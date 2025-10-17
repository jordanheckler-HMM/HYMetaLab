#!/usr/bin/env python3
"""
HYMetaLab MetaDashboard v2.0 - Next-Gen Research Monitor
Features: Sortable, Themeable, Responsive, Lab-Color-Coded
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, send_from_directory

# Use absolute path to ensure it works when run via LaunchAgent
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
PUBLIC = SCRIPT_DIR / "public"
PUBLIC.mkdir(exist_ok=True)
DATA_CACHE = PUBLIC / "dashboard_data.json"
INDEX_HTML = PUBLIC / "index.html"

# Lab color scheme
LAB_COLORS = {
    "openlight": "#3b82f6",  # Blue
    "opentime": "#f97316",  # Orange
    "openmind": "#22c55e",  # Green
    "unknown": "#6b7280",  # Gray
}


def extract_lab(source_dir: str) -> str:
    """Extract lab name from directory path"""
    lower = source_dir.lower()
    if "openlight" in lower or "light" in lower:
        return "openlight"
    elif "opentime" in lower or "time" in lower:
        return "opentime"
    elif "openmind" in lower or "mind" in lower:
        return "openmind"
    else:
        return "unknown"


def extract_timestamp(source_dir: str) -> str:
    """Extract timestamp from directory name"""
    match = re.search(r"(\d{8}_\d{6})", source_dir)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass
    return "Unknown"


def load_df():
    rows = []
    for p in (SCRIPT_DIR / "discovery_results").glob("*/summary.json"):
        try:
            d = json.loads(p.read_text())
            d["_source_dir"] = str(p.parent.relative_to(SCRIPT_DIR))
            d["_timestamp"] = extract_timestamp(str(p.parent))

            # Extract abstract if available
            abstract_path = p.parent / "abstract.md"
            if abstract_path.exists():
                d["_abstract"] = abstract_path.read_text()[:300] + "..."
            else:
                d["_abstract"] = "No abstract available"

            # Extract DOI if available
            d["_doi"] = d.get("zenodo_doi", "Not published")

            rows.append(d)
        except:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows, max_level=2)

    # Normalize column names
    ren = {}
    # If summaries put numeric metrics under a 'metrics' dict (e.g. delta_cci_mean),
    # promote sensible metric keys to top-level columns for dashboard consumption.
    if "metrics" in df.columns:
        try:
            metrics_expanded = pd.json_normalize(df["metrics"]).add_prefix("metrics.")
            df = pd.concat([df.drop(columns=["metrics"]), metrics_expanded], axis=1)
        except Exception:
            # If expansion fails, continue without raising to avoid breaking dashboard build
            pass
    for c in df.columns:
        lc = c.lower()
        if "delta_cci" in lc or "Δcci" in c:
            ren[c] = "delta_cci"
        elif "delta_hazard" in lc or ("hazard" in lc and "delta" in lc):
            ren[c] = "delta_hazard"
        elif lc.endswith(".epsilon") or lc == "epsilon":
            ren[c] = "epsilon"
        elif ".rho" in lc or lc == "rho":
            ren[c] = "rho"
        elif "openlaws_score" in lc:
            ren[c] = "openlaws_score"
    df = df.rename(columns=ren)

    # Ensure required columns exist
    for k in ["delta_cci", "delta_hazard", "epsilon", "rho", "openlaws_score"]:
        if k not in df.columns:
            df[k] = None

    # Ensure column names are unique (append suffixes to duplicates)
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols

    # Extract lab and assign color
    df["lab"] = df["_source_dir"].apply(extract_lab)
    df["lab_color"] = df["lab"].map(LAB_COLORS)

    # Convert to numeric
    for col in ["delta_cci", "delta_hazard", "epsilon", "rho", "openlaws_score"]:
        try:
            # If column is a pandas Series of scalars, convert directly
            if hasattr(df[col], "apply"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                # Fallback: attempt to extract numeric from nested structures
                def extract_num(v):
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        return v
                    if isinstance(v, str):
                        try:
                            return float(v)
                        except:
                            return None
                    if isinstance(v, dict):
                        # common nested keys
                        for k in ["mean", "value", "val"]:
                            if k in v and isinstance(v[k], (int, float)):
                                return v[k]
                    return None

                df[col] = (
                    df[col].map(extract_num)
                    if hasattr(df[col], "map")
                    else [extract_num(x) for x in df[col]]
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = pd.Series([None] * len(df))

    return df


def build_advanced_dashboard():
    """Build advanced dashboard with sorting, theming, responsive design"""
    df = load_df()
    DATA_CACHE.write_text(df.to_json(orient="records", date_format="iso"))

    if df.empty:
        INDEX_HTML.write_text(
            "<h2>No data yet. Run studies to populate dashboard.</h2>"
        )
        return

    # Generate card-based HTML
    html = (
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <meta http-equiv="refresh" content="30"/>
    <title>HYMetaLab — MetaDashboard v2.0</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --bg-primary: #0b0b0b;
            --bg-secondary: #1a1a1a;
            --bg-card: #2a2a2a;
            --text-primary: #d9f99d;
            --text-secondary: #86efac;
            --border: #374151;
            --shadow: rgba(0,0,0,0.3);
        }
        
        [data-theme="light"] {
            --bg-primary: #f9fafb;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --text-primary: #1f2937;
            --text-secondary: #059669;
            --border: #e5e7eb;
            --shadow: rgba(0,0,0,0.1);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
            padding: 16px;
            transition: background 0.3s, color 0.3s;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }
        
        h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #86efac 0%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .theme-toggle {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-primary);
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow);
        }
        
        .sort-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
        }
        
        .sort-btn:hover { background: var(--bg-secondary); }
        .sort-btn.active { background: var(--text-secondary); color: var(--bg-primary); }
        
        .tags {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tag {
            padding: 6px 12px;
            border-radius: 8px;
            background: rgba(22, 163, 74, 0.15);
            font-size: 13px;
            font-weight: 500;
            border: 1px solid rgba(22, 163, 74, 0.3);
        }
        
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px var(--shadow);
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px var(--shadow);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 12px;
        }
        
        .card-title {
            font-weight: 600;
            font-size: 14px;
            flex: 1;
            line-height: 1.4;
        }
        
        .lab-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .lab-openlight { background: #3b82f630; color: #3b82f6; border: 1px solid #3b82f660; }
        .lab-opentime { background: #f9731630; color: #f97316; border: 1px solid #f9731660; }
        .lab-openmind { background: #22c55e30; color: #22c55e; border: 1px solid #22c55e60; }
        .lab-unknown { background: #6b728030; color: #6b7280; border: 1px solid #6b728060; }
        
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 16px 0;
        }
        
        .metric {
            background: var(--bg-secondary);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: 700;
        }
        
        .metric-value.positive { color: #22c55e; }
        .metric-value.negative { color: #ef4444; }
        
        .card-footer {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        
        .card-link {
            font-size: 12px;
            color: var(--text-secondary);
            text-decoration: none;
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
        }
        
        .card-link:hover { opacity: 0.8; }
        
        .timestamp {
            font-size: 11px;
            opacity: 0.6;
            margin-top: 8px;
        }
        
        .plot-container {
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-top: 24px;
        }
        
        @media (max-width: 768px) {
            .header { flex-direction: column; align-items: start; }
            .cards-grid { grid-template-columns: 1fr; }
            .controls { width: 100%; justify-content: start; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 HYMetaLab MetaDashboard v2.0</h1>
        <div class="controls">
            <button class="theme-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
            <button class="sort-btn active" onclick="sortBy('delta_cci')">Sort by ΔCCI</button>
            <button class="sort-btn" onclick="sortBy('delta_hazard')">Sort by Δhazard</button>
            <button class="sort-btn" onclick="sortBy('timestamp')">Sort by Time</button>
        </div>
    </div>
    
    <div class="tags">
        <span class="tag">✅ VALIDATED: ΔCCI ≥ 0.03</span>
        <span class="tag">⚠️ Δhazard ≤ −0.01</span>
        <span class="tag">🏆 OpenLawsScore ≥ 0.75</span>
    </div>
    
    <div id="cards-container" class="cards-grid"></div>
    
    <div class="plot-container" id="plot"></div>
    
    <script>
        let data = """
        + df.to_json(orient="records")
        + """;
        let currentSort = 'delta_cci';
        
        // Theme management
        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updatePlot();
        }
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        
        // Sorting
        function sortBy(field) {
            currentSort = field;
            document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            if (field === 'timestamp') {
                data.sort((a, b) => (b._timestamp || '').localeCompare(a._timestamp || ''));
            } else {
                data.sort((a, b) => (b[field] || -999) - (a[field] || -999));
            }
            renderCards();
        }
        
        function renderCards() {
            const container = document.getElementById('cards-container');
            container.innerHTML = data.map(d => `
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">${(d._source_dir || 'Unknown').split('/').pop()}</div>
                        <div class="lab-badge lab-${d.lab || 'unknown'}">${(d.lab || 'unknown').toUpperCase()}</div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">ΔCCI</div>
                            <div class="metric-value ${(d.delta_cci || 0) >= 0.03 ? 'positive' : ''}">${(d.delta_cci || 0).toFixed(4)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Δhazard</div>
                            <div class="metric-value ${(d.delta_hazard || 0) <= -0.01 ? 'positive' : 'negative'}">${(d.delta_hazard || 0).toFixed(4)}</div>
                        </div>
                    </div>
                    
                    <div class="card-footer">
                        <a href="#" class="card-link" onclick="alert('Abstract: ' + this.dataset.abstract); return false;" 
                           data-abstract="${(d._abstract || 'No abstract').replace(/"/g, '&quot;')}">📄 Abstract</a>
                        ${d._doi && d._doi !== 'Not published' ? 
                          `<a href="https://doi.org/${d._doi}" class="card-link" target="_blank">🔗 DOI</a>` : 
                          '<span class="card-link" style="opacity:0.5;">📦 Unpublished</span>'}
                    </div>
                    
                    <div class="timestamp">⏰ ${d._timestamp || 'Unknown time'}</div>
                </div>
            `).join('');
        }
        
        function updatePlot() {
            const theme = document.documentElement.getAttribute('data-theme') || 'dark';
            const bgColor = theme === 'dark' ? '#1a1a1a' : '#ffffff';
            const textColor = theme === 'dark' ? '#d9f99d' : '#1f2937';
            const gridColor = theme === 'dark' ? '#374151' : '#e5e7eb';
            
            const trace = {
                x: data.map(d => d.rho).filter(v => v != null),
                y: data.map(d => d.epsilon).filter(v => v != null),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: data.map(d => (d.openlaws_score || 0.5) * 20),
                    color: data.map(d => d.delta_cci || 0),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: { title: 'ΔCCI' }
                },
                text: data.map(d => `Lab: ${d.lab}<br>ΔCCI: ${(d.delta_cci || 0).toFixed(4)}`),
                hoverinfo: 'text'
            };
            
            const layout = {
                title: 'Research Space: ρ × ε (colored by ΔCCI)',
                xaxis: { title: 'ρ (density)', gridcolor: gridColor },
                yaxis: { title: 'ε (energy coupling)', gridcolor: gridColor },
                plot_bgcolor: bgColor,
                paper_bgcolor: bgColor,
                font: { color: textColor },
                height: 500
            };
            
            Plotly.newPlot('plot', [trace], layout, {responsive: true});
        }
        
        // Initial render
        sortBy('delta_cci');
        updatePlot();
        
        console.log('MetaDashboard v2.0 loaded with', data.length, 'studies');
    </script>
</body>
</html>"""
    )

    INDEX_HTML.write_text(html)


# Alias for backwards compatibility
build = build_advanced_dashboard

app = Flask(__name__)


@app.get("/")
def index():
    if not INDEX_HTML.exists():
        build_advanced_dashboard()
    return send_from_directory(PUBLIC, "index.html")


@app.get("/dashboard_data.json")
def data():
    if not DATA_CACHE.exists():
        build_advanced_dashboard()
    return send_from_directory(PUBLIC, "dashboard_data.json")


@app.get("/api/studies")
def api_studies():
    """API endpoint for study data"""
    df = load_df()
    return jsonify(json.loads(df.to_json(orient="records")))


if __name__ == "__main__":
    build_advanced_dashboard()
    print("✅ MetaDashboard v2.0 built successfully")
    print("🚀 Starting Flask server on http://localhost:5055")
    app.run(debug=True, port=5055)
