#!/usr/bin/env python3
"""
Guardian v4 - CI/CD Integration Hooks
Git hooks and CI/CD pipeline integration for automated ethical validation
"""
import subprocess
from pathlib import Path


class GuardianHooks:
    """
    Install and manage Guardian v4 hooks for Git and CI/CD
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.git_dir = self.root / ".git"
        self.hooks_dir = self.git_dir / "hooks"
        self.guardian_log = self.root / "qc" / "guardian_v4" / "guardian_validation.log"
        self.guardian_log.parent.mkdir(parents=True, exist_ok=True)

    def install_pre_commit_hook(self) -> bool:
        """Install pre-commit hook for Guardian validation"""
        if not self.git_dir.exists():
            print("⚠️  Not a Git repository")
            return False

        self.hooks_dir.mkdir(exist_ok=True)
        hook_path = self.hooks_dir / "pre-commit"

        hook_script = '''#!/usr/bin/env python3
"""
Guardian v4 Pre-Commit Hook
Validates ethical alignment before allowing commits
"""
import sys
import subprocess
from pathlib import Path

# Get list of staged files
result = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    capture_output=True,
    text=True
)

staged_files = [f for f in result.stdout.strip().split('\\n') if f]
markdown_files = [f for f in staged_files if f.endswith(('.md', '.txt'))]

if not markdown_files:
    print("✅ No documentation files to validate")
    sys.exit(0)

print(f"🔍 Guardian v4: Validating {len(markdown_files)} file(s)...")

# Run Guardian validation on staged files
failed_files = []
for file_path in markdown_files:
    if not Path(file_path).exists():
        continue
    
    result = subprocess.run(
        ["python3", "qc/guardian_v4/metrics/risk_assessor.py", "validate", "--file", file_path],
        capture_output=True,
        text=True
    )
    
    # Check if validation passed (Guardian score >= 70)
    if "FAIL" in result.stdout or result.returncode != 0:
        failed_files.append(file_path)
        print(f"   ❌ {file_path}: Failed Guardian validation")
    else:
        print(f"   ✅ {file_path}: Passed")

if failed_files:
    print(f"\\n❌ Guardian v4: {len(failed_files)} file(s) failed validation")
    print("   Minimum Guardian score: 70/100")
    print("   Run: python3 qc/guardian_v4/guardian_v4.py --validate <file>")
    print("   Or: git commit --no-verify (to skip validation)")
    sys.exit(1)

print("✅ Guardian v4: All files passed validation")
sys.exit(0)
'''

        hook_path.write_text(hook_script)
        hook_path.chmod(0o755)  # Make executable

        print(f"✅ Pre-commit hook installed: {hook_path}")
        return True

    def install_github_actions_workflow(self) -> bool:
        """Create GitHub Actions workflow for Guardian CI/CD"""
        workflow_dir = self.root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)

        workflow_path = workflow_dir / "guardian_v4_ci.yml"

        workflow_yaml = """name: Guardian v4 Ethical Alignment CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  guardian-validation:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        pip install pyyaml numpy pandas scipy scikit-learn
        
    - name: Run Guardian v4 validation
      run: |
        python3 qc/guardian_v4/guardian_v4.py --validate --report
        
    - name: Check Guardian score threshold
      run: |
        SCORE=$(python3 -c "import json; print(json.load(open('qc/guardian_v4/guardian_report_v4.json'))['guardian_alignment_score'])")
        echo "Guardian Alignment Score: $SCORE"
        if (( $(echo "$SCORE < 90" | bc -l) )); then
          echo "❌ Guardian score below deployment threshold (90)"
          exit 1
        fi
        echo "✅ Guardian validation passed"
        
    - name: Upload Guardian report
      uses: actions/upload-artifact@v3
      with:
        name: guardian-report
        path: qc/guardian_v4/guardian_report_v4.json
        
    - name: Comment PR with Guardian score
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = JSON.parse(fs.readFileSync('qc/guardian_v4/guardian_report_v4.json'));
          const score = report.guardian_alignment_score;
          const risk = report.risk_assessment.risk_level;
          const emoji = score >= 90 ? '🟢' : score >= 70 ? '🟡' : '🔴';
          
          const body = `## ${emoji} Guardian v4 Ethical Alignment Report
          
          **Guardian Score**: ${score.toFixed(1)}/100
          **Risk Level**: ${risk.toUpperCase()}
          
          ### Component Metrics
          - **Objectivity**: ${report.metrics.objectivity_score.toFixed(2)}
          - **Transparency**: ${report.metrics.transparency_index_v2.toFixed(2)}
          - **Language Safety**: ${report.metrics.language_safety_score.toFixed(2)}
          - **Sentiment**: ${report.metrics.sentiment_neutrality.toFixed(2)}
          
          ${score >= 90 ? '✅ Ready for deployment' : '⚠️ Review recommended'}
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: body
          });
"""

        workflow_path.write_text(workflow_yaml)
        print(f"✅ GitHub Actions workflow created: {workflow_path}")
        return True

    def create_ci_hooks_config(self) -> bool:
        """Create guardian_ci_hooks.yml configuration"""
        config_path = self.root / "qc" / "guardian_v4" / "guardian_ci_hooks.yml"

        config_yaml = """# Guardian v4 CI/CD Hooks Configuration
version: "1.0"

# Gate configurations
gates:
  pre_commit:
    enabled: true
    minimum_score: 70
    block_on_failure: true
    allowed_bypass: false
    
  pre_push:
    enabled: true
    minimum_score: 80
    block_on_failure: true
    
  pre_merge:
    enabled: true
    minimum_score: 90
    require_all_metrics_pass: true
    require_human_review: true
    
  pre_deploy:
    enabled: true
    minimum_score: 90
    require_all_metrics_pass: true
    require_sign_off: true

# File patterns to validate
validation_patterns:
  - "**/*.md"
  - "**/*.txt"
  - "docs/**/*"
  - "studies/**/*.yml"

# Exclusions
exclude_patterns:
  - "node_modules/**"
  - ".venv/**"
  - "**/__pycache__/**"
  - ".git/**"
  - "qc/QC_REPORT.json"  # Contains violations as metadata

# Metric thresholds
thresholds:
  objectivity_score: 0.80
  transparency_index_v2: 0.90
  language_safety_score: 0.85
  sentiment_neutrality: [-0.1, 0.1]

# Notification settings
notifications:
  slack:
    enabled: "${SLACK_ENABLED:-false}"
    webhook: "${SLACK_WEBHOOK_URL}"
    on_failure: true
    on_success: false
    
  email:
    enabled: false
    recipients: []
    
# Logging
logging:
  level: "INFO"
  file: "qc/guardian_v4/guardian_validation.log"
  format: "%(asctime)s - %(levelname)s - %(message)s"
"""

        config_path.write_text(config_yaml)
        print(f"✅ CI hooks config created: {config_path}")
        return True

    def install_all_hooks(self):
        """Install all Guardian CI/CD hooks"""
        print("🔧 Installing Guardian v4 CI/CD hooks...")

        results = {
            "pre_commit": self.install_pre_commit_hook(),
            "github_actions": self.install_github_actions_workflow(),
            "ci_config": self.create_ci_hooks_config(),
        }

        print("\n✅ Hook installation complete")
        print(f"   Pre-commit hook: {'✅' if results['pre_commit'] else '❌'}")
        print(f"   GitHub Actions: {'✅' if results['github_actions'] else '❌'}")
        print(f"   CI config: {'✅' if results['ci_config'] else '❌'}")

        return all(results.values())

    def test_hooks(self):
        """Test installed hooks"""
        print("🧪 Testing Guardian hooks...")

        # Create test file
        test_file = self.root / "test_guardian_hook.md"
        test_file.write_text("# Test\nThis suggests that testing is important.")

        # Stage file
        subprocess.run(["git", "add", str(test_file)], check=False)

        # Run pre-commit hook manually
        hook_path = self.hooks_dir / "pre-commit"
        if hook_path.exists():
            result = subprocess.run([str(hook_path)], capture_output=True, text=True)
            print(f"\n{result.stdout}")
            if result.returncode == 0:
                print("✅ Pre-commit hook test passed")
            else:
                print("⚠️  Pre-commit hook test failed (expected for test)")

        # Cleanup
        subprocess.run(["git", "reset", "HEAD", str(test_file)], check=False)
        test_file.unlink(missing_ok=True)


def main():
    """CLI interface for Guardian hooks"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v4 CI/CD Hooks")
    parser.add_argument(
        "command", choices=["install", "test", "uninstall"], help="Command to execute"
    )

    args = parser.parse_args()

    hooks = GuardianHooks()

    if args.command == "install":
        hooks.install_all_hooks()

    elif args.command == "test":
        hooks.test_hooks()

    elif args.command == "uninstall":
        print("🗑️  Uninstalling Guardian hooks...")
        hook_path = hooks.hooks_dir / "pre-commit"
        if hook_path.exists():
            hook_path.unlink()
            print("✅ Pre-commit hook removed")
        print("✅ Uninstall complete")


if __name__ == "__main__":
    main()
