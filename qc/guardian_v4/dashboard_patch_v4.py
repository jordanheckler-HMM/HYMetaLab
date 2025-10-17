#!/usr/bin/env python3
"""
Guardian v4 - Dashboard Integration Patch
Adds Guardian v4 metrics to MetaDashboard with traffic-light visualization
"""
import json
from datetime import datetime
from pathlib import Path


class GuardianDashboardPatch:
    """
    Patch for integrating Guardian v4 into MetaDashboard
    Adds real-time traffic-light status and metric widgets
    """

    def __init__(self):
        self.guardian_data_file = Path("qc/guardian_v4/guardian_report_v4.json")
        self.dashboard_data_file = Path("public/dashboard_data.json")
        # Working root used for locating metadata/config files
        self.root = Path(".")

    def load_guardian_data(self) -> dict:
        """Load latest Guardian v4 report"""
        if not self.guardian_data_file.exists():
            return None

        try:
            return json.loads(self.guardian_data_file.read_text())
        except:
            return None

    def generate_traffic_light_widget(self, guardian_data: dict) -> dict:
        """
        Generate traffic-light widget data
        Green (≥90), Yellow (70-90), Red (<70)
        """
        if not guardian_data:
            return {
                "type": "traffic_light",
                "status": "gray",
                "score": 0,
                "message": "No Guardian data available",
            }

        score = guardian_data.get(
            "mean_score", guardian_data.get("guardian_alignment_score", 0)
        )

        if score >= 90:
            status = "green"
            message = "✅ Excellent ethical alignment"
        elif score >= 70:
            status = "yellow"
            message = "⚠️ Good - minor improvements recommended"
        else:
            status = "red"
            message = "❌ Needs review - below threshold"

        return {
            "type": "traffic_light",
            "status": status,
            "score": round(score, 1),
            "message": message,
            "timestamp": guardian_data.get("timestamp", datetime.now().isoformat()),
        }

    def generate_metrics_widget(self, guardian_data: dict) -> dict:
        """Generate component metrics widget"""
        if not guardian_data:
            return {"type": "metrics", "data": []}

        # Extract metrics from either single doc or corpus
        if "metrics" in guardian_data:
            metrics = guardian_data["metrics"]
        elif "documents" in guardian_data and guardian_data["documents"]:
            # Average metrics from corpus
            docs = guardian_data["documents"]
            metrics = {
                "objectivity_score": sum(
                    d["metrics"]["objectivity_score"] for d in docs
                )
                / len(docs),
                "transparency_index_v2": sum(
                    d["metrics"]["transparency_index_v2"] for d in docs
                )
                / len(docs),
                "language_safety_score": sum(
                    d["metrics"]["language_safety_score"] for d in docs
                )
                / len(docs),
                "sentiment_neutrality": sum(
                    d["metrics"]["sentiment_neutrality"] for d in docs
                )
                / len(docs),
            }
        else:
            return {"type": "metrics", "data": []}

        return {
            "type": "metrics_bar",
            "data": [
                {
                    "name": "Objectivity",
                    "value": round(metrics.get("objectivity_score", 0), 2),
                    "target": 0.80,
                    "color": "blue",
                },
                {
                    "name": "Transparency",
                    "value": round(metrics.get("transparency_index_v2", 0), 2),
                    "target": 0.90,
                    "color": "green",
                },
                {
                    "name": "Language Safety",
                    "value": round(metrics.get("language_safety_score", 0), 2),
                    "target": 0.85,
                    "color": "orange",
                },
                {
                    "name": "Sentiment",
                    "value": round(abs(metrics.get("sentiment_neutrality", 0)), 2),
                    "target": 0.10,
                    "inverted": True,  # Lower is better
                    "color": "purple",
                },
            ],
        }

    def generate_trend_widget(self, guardian_data: dict) -> dict:
        """Generate trend widget (historical scores)"""
        # For MVP, we'll just show current score
        # In production, this would load historical data
        return {
            "type": "trend_line",
            "data": [],
            "message": "Historical trend tracking coming soon",
        }

    def generate_context_map(self, guardian_data: dict) -> dict:
        """
        Generate context map visualization data (v6 feature)
        Shows claim nodes connected to evidence edges
        """
        if not guardian_data or "v6_context" not in guardian_data:
            return {"type": "context_map", "nodes": [], "edges": []}

        v6_data = guardian_data.get("v6_context", {})
        evidence_links = v6_data.get("evidence_links", {})

        nodes = []
        edges = []

        # Create nodes for each claim
        for i, pair in enumerate(
            evidence_links.get("claim_evidence_pairs", [])[:10]
        ):  # Limit to 10
            # Claim node
            nodes.append(
                {
                    "id": f"claim_{i}",
                    "type": "claim",
                    "label": pair["claim_text"][:50] + "...",
                    "supported": pair["is_supported"],
                }
            )

            # Evidence nodes and edges
            for j, evidence in enumerate(
                pair.get("evidence_items", [])[:3]
            ):  # Max 3 per claim
                evidence_id = f"evidence_{i}_{j}"
                nodes.append(
                    {
                        "id": evidence_id,
                        "type": evidence["type"],
                        "label": evidence["text"][:30],
                    }
                )

                # Edge from claim to evidence
                edges.append(
                    {
                        "from": f"claim_{i}",
                        "to": evidence_id,
                        "distance": evidence.get("distance", 0),
                    }
                )

        return {
            "type": "context_map",
            "nodes": nodes,
            "edges": edges,
            "claim_count": len(evidence_links.get("claim_evidence_pairs", [])),
            "evidence_coverage": evidence_links.get("evidence_coverage", 0),
        }

    def get_self_integrity_score(self) -> dict:
        """Load Self-Integrity Score from metadata"""
        metadata_path = (
            self.root / "qc" / "guardian_v4" / "config" / "self_integrity_metadata.json"
        )

        if not metadata_path.exists():
            return {"score": None, "status": "NO_DATA"}

        try:
            metadata = json.load(open(metadata_path))
            history = metadata.get("self_integrity_history", [])

            if not history:
                return {"score": None, "status": "NO_DATA"}

            latest = history[-1]
            return {
                "score": latest.get("self_integrity_score", 1.0),
                "status": latest.get("status", "UNKNOWN"),
                "passes": latest.get("passes_threshold", False),
            }
        except Exception:
            return {"score": None, "status": "ERROR"}

    def generate_html_widget(self, guardian_data: dict) -> str:
        """Generate HTML widget for dashboard (v6 + v7 + v9 + v10 enhanced)"""
        traffic_light = self.generate_traffic_light_widget(guardian_data)
        metrics = self.generate_metrics_widget(guardian_data)
        context_map = self.generate_context_map(guardian_data)
        self_integrity = self.get_self_integrity_score()

        # v7 Continuity data
        v7_memory = guardian_data.get("v7_memory", {})
        consistency = v7_memory.get("consistency", {})
        continuity_score = consistency.get("continuity_score", 1.0)
        continuity_pct = continuity_score * 100
        contradictions = consistency.get("contradictions", [])

        # v9 Explanations data (will be generated client-side or pre-computed)
        explanations_data = guardian_data.get("explanations", {})

        # v10 Self-Integrity badge
        integrity_score = self_integrity.get("score")
        integrity_status = self_integrity.get("status", "NO_DATA")
        integrity_class = (
            "green"
            if self_integrity.get("passes", False)
            else "yellow" if integrity_score and integrity_score >= 0.90 else "red"
        )

        integrity_display = (
            f"{integrity_score:.2f}"
            if isinstance(integrity_score, (int, float))
            else "N/A"
        )

        html = f"""
        <div class="guardian-widget">
            <div class="header-bar">
                <h2>🛡️ Guardian v4/v6/v7/v9/v10 Ethical Alignment</h2>
                <div class="self-integrity-badge badge-{integrity_class}">
                    Self-Integrity: {integrity_display}
                </div>
            </div>
            
            <div class="traffic-light-{traffic_light['status']}">
                <div class="score">{traffic_light['score']}/100</div>
                <div class="message">{traffic_light['message']}</div>
            </div>
            
            <div class="metrics-grid">
        """

        for metric in metrics.get("data", []):
            status = "✅" if metric["value"] >= metric["target"] else "⚠️"
            html += f"""
                <div class="metric-card">
                    <div class="metric-name">{metric['name']}</div>
                    <div class="metric-value">{metric['value']}</div>
                    <div class="metric-target">Target: {metric['target']}</div>
                    <div class="metric-status">{status}</div>
                </div>
            """

        html += """
            </div>
            
            <div class="context-map-section">
                <h3>🧠 Context Map (v6)</h3>
                <div class="context-stats">
                    <span>Claims: {claim_count}</span>
                    <span>Coverage: {coverage:.0f}%</span>
                </div>
                <div class="context-graph" id="contextGraph">
                    <!-- Graph rendered here -->
                    <p><em>Context map showing {node_count} nodes, {edge_count} connections</em></p>
                </div>
            </div>
            
            <div class="continuity-section">
                <h3>🔗 Continuity (v7)</h3>
                <div class="continuity-bar-container">
                    <div class="continuity-bar" style="width: {continuity_pct:.0f}%">
                        {continuity_score:.2f}
                    </div>
                </div>
                <div class="contradiction-list" id="contradictionList">
                    <!-- Contradictions listed here -->
                </div>
            </div>
            
            <div class="explanations-section">
                <h3>📖 Explanations (v9)</h3>
                <button class="toggle-explanations" onclick="toggleExplanations()">
                    Show/Hide Explanations
                </button>
                <div class="explanations-content" id="explanationsContent" style="display: none;">
                    <!-- Explanations will be rendered here -->
                </div>
            </div>
        </div>
        
        <script>
        // v6 Context map data
        const contextMapData = {context_map_json};
        
        // v7 Continuity data
        const continuityData = {continuity_json};
        
        // v9 Explanations data
        const explanationsData = {explanations_json};
        
        // Simple text-based visualization (upgrade to D3.js for production)
        function renderContextMap(data) {{
            const graphDiv = document.getElementById('contextGraph');
            if (!data.nodes || data.nodes.length === 0) {{
                graphDiv.innerHTML = '<p><em>No context map data available</em></p>';
                return;
            }}
            
            let html = '<ul class="context-list">';
            const claims = data.nodes.filter(n => n.type === 'claim');
            claims.forEach(claim => {{
                const edges = data.edges.filter(e => e.from === claim.id);
                const status = claim.supported ? '✅' : '⚠️';
                html += `<li>${{status}} ${{claim.label}}<ul>`;
                edges.forEach(edge => {{
                    const evidence = data.nodes.find(n => n.id === edge.to);
                    if (evidence) {{
                        html += `<li>→ ${{evidence.label}} (d=${{edge.distance}})</li>`;
                    }}
                }});
                html += '</ul></li>';
            }});
            html += '</ul>';
            graphDiv.innerHTML = html;
        }}
        
        // Render context map on load
        if (typeof contextMapData !== 'undefined') {{
            renderContextMap(contextMapData);
        }}
        
        // Render continuity contradictions
        function renderContradictions(data) {{
            const listDiv = document.getElementById('contradictionList');
            if (!data || !data.contradictions || data.contradictions.length === 0) {{
                listDiv.innerHTML = '<p><em>No contradictions detected ✅</em></p>';
                return;
            }}
            
            let html = '<ul class="contradiction-items">';
            data.contradictions.forEach((contr, idx) => {{
                html += `<li>
                    <strong>${{contr.claim_key}}</strong>: 
                    <span class="stance-${{contr.document_stance}}">${{contr.document_stance}}</span>
                    vs 
                    <span class="stance-${{contr.corpus_stance}}">${{contr.corpus_stance}}</span>
                    <br/>
                    <small>→ <a href="${{contr.corpus_file}}">${{contr.corpus_file}}</a></small>
                </li>`;
            }});
            html += '</ul>';
            listDiv.innerHTML = html;
        }}
        
        if (typeof continuityData !== 'undefined') {{
            renderContradictions(continuityData);
        }}
        
        // Render explanations (v9)
        function toggleExplanations() {{
            const contentDiv = document.getElementById('explanationsContent');
            if (contentDiv.style.display === 'none') {{
                contentDiv.style.display = 'block';
                renderExplanations(explanationsData);
            }} else {{
                contentDiv.style.display = 'none';
            }}
        }}
        
        function renderExplanations(data) {{
            const contentDiv = document.getElementById('explanationsContent');
            if (!data || !data.explanations) {{
                contentDiv.innerHTML = '<p><em>No explanations available</em></p>';
                return;
            }}
            
            let html = '';
            const explanations = data.explanations;
            
            for (const [metric, exp] of Object.entries(explanations)) {{
                html += `
                    <div class="explanation-card">
                        <h4>${{metric.replace(/_/g, ' ').toUpperCase()}} 
                            <span class="grade-badge grade-${{exp.grade}}">${{exp.grade}}</span>
                        </h4>
                        <div class="score-display">${{exp.score.toFixed(2)}}</div>
                        <p class="explanation-text">${{exp.explanation}}</p>
                        <details>
                            <summary>View Trace Data</summary>
                            <pre>${{JSON.stringify(exp.trace, null, 2)}}</pre>
                        </details>
                    </div>
                `;
            }}
            
            contentDiv.innerHTML = html;
        }}
        </script>
        
        <style>
        .guardian-widget {
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .header-bar h2 {{
            margin: 0;
        }}
        
        .self-integrity-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .badge-green {{
            background: #4CAF50;
            color: white;
        }}
        
        .badge-yellow {{
            background: #FFC107;
            color: black;
        }}
        
        .badge-red {{
            background: #f44336;
            color: white;
        }}
        
        .context-map-section {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }}
        
        .context-stats {{
            display: flex;
            gap: 20px;
            margin: 10px 0;
            font-weight: bold;
        }}
        
        .context-graph {{
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .context-list {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .context-list li {{
            margin: 5px 0;
            padding: 5px;
        }}
        
        .context-list ul {{
            padding-left: 20px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .continuity-section {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }}
        
        .continuity-bar-container {{
            width: 100%;
            height: 30px;
            background: #f0f0f0;
            border-radius: 4px;
            margin: 10px 0;
            position: relative;
        }}
        
        .continuity-bar {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        
        .contradiction-list {{
            margin-top: 15px;
        }}
        
        .contradiction-items {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .contradiction-items li {{
            margin: 10px 0;
            padding: 10px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }}
        
        .stance-positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        
        .stance-negative {{
            color: #f44336;
            font-weight: bold;
        }}
        
        .explanations-section {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }}
        
        .toggle-explanations {{
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        .toggle-explanations:hover {{
            background: #1976D2;
        }}
        
        .explanations-content {{
            margin-top: 15px;
        }}
        
        .explanation-card {{
            margin: 15px 0;
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
        
        .explanation-card h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        
        .grade-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .grade-A {{ background: #4CAF50; color: white; }}
        .grade-B {{ background: #8BC34A; color: white; }}
        .grade-C {{ background: #FFC107; color: black; }}
        .grade-D {{ background: #FF9800; color: white; }}
        .grade-F {{ background: #f44336; color: white; }}
        
        .score-display {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
            margin: 10px 0;
        }}
        
        .explanation-text {{
            line-height: 1.6;
            color: #555;
        }}
        
        .explanation-card details {{
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        
        .explanation-card summary {{
            cursor: pointer;
            font-weight: bold;
            color: #2196F3;
        }}
        
        .explanation-card pre {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
        }}
        
        .traffic-light-green {
            background: #4CAF50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .traffic-light-yellow {
            background: #FFC107;
            color: black;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .traffic-light-red {
            background: #F44336;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .score {
            font-size: 48px;
            font-weight: bold;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-name {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 32px;
            color: #2196F3;
        }
        </style>
        """

        # Fill in data placeholders (not CSS/JS braces)

        # Replace data placeholders manually to avoid CSS brace conflicts
        html = html.replace("{claim_count}", str(context_map.get("claim_count", 0)))
        html = html.replace(
            "{coverage:.0f}", f"{context_map.get('evidence_coverage', 0) * 100:.0f}"
        )
        html = html.replace("{node_count}", str(len(context_map.get("nodes", []))))
        html = html.replace("{edge_count}", str(len(context_map.get("edges", []))))
        html = html.replace("{continuity_score:.2f}", f"{continuity_score:.2f}")
        html = html.replace("{continuity_pct:.0f}", f"{continuity_pct:.0f}")
        html = html.replace("{context_map_json}", json.dumps(context_map))
        html = html.replace(
            "{continuity_json}", json.dumps({"contradictions": contradictions})
        )
        html = html.replace("{explanations_json}", json.dumps(explanations_data))

        return html

    def patch_dashboard(self):
        """Apply Guardian v4 patch to dashboard"""
        guardian_data = self.load_guardian_data()

        if not guardian_data:
            print("⚠️  No Guardian data found - run validation first")
            return False

        # Generate widgets
        traffic_light = self.generate_traffic_light_widget(guardian_data)
        metrics = self.generate_metrics_widget(guardian_data)

        # Create dashboard patch data
        patch_data = {
            "guardian_v4": {
                "traffic_light": traffic_light,
                "metrics": metrics,
                "last_updated": datetime.now().isoformat(),
            }
        }

        # Save patch data
        patch_file = Path("qc/guardian_v4/dashboard_patch_data.json")
        patch_file.write_text(json.dumps(patch_data, indent=2))

        print("✅ Dashboard patch applied")
        print(f"   Guardian Score: {traffic_light['score']}/100")
        print(f"   Status: {traffic_light['status'].upper()}")
        print(f"   Patch data: {patch_file}")

        # Generate HTML widget
        html_widget = self.generate_html_widget(guardian_data)
        html_file = Path("qc/guardian_v4/dashboard_widget.html")
        html_file.write_text(html_widget)
        print(f"   HTML widget: {html_file}")

        return True


def main():
    """CLI interface for dashboard patch"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 Dashboard Patch")
    parser.add_argument(
        "command", choices=["patch", "preview"], help="Command to execute"
    )

    args = parser.parse_args()

    patcher = GuardianDashboardPatch()

    if args.command == "patch":
        patcher.patch_dashboard()

    elif args.command == "preview":
        guardian_data = patcher.load_guardian_data()
        if guardian_data:
            html = patcher.generate_html_widget(guardian_data)
            print(html)
        else:
            print("⚠️  No Guardian data available")


if __name__ == "__main__":
    main()
