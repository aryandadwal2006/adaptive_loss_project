"""
Automated Report Generator
Produces a Markdown report summarizing experiment results and embedding plots
"""

from pathlib import Path

class ReportGenerator:
    """
    Generates a Markdown report with metrics and figures.
    """
    def __init__(self, results_dir: str = "./experiment_results", report_path: str = "REPORT.md"):
        self.results_dir = Path(results_dir)
        self.report_path = Path(report_path)

    def generate(self, summary: dict, figures: list):
        lines = ["# Experiment Report\n"]
        lines.append("## Summary Metrics\n")
        for k,v in summary.items():
            lines.append(f"- **{k.replace('_',' ').title()}**: {v}\n")
        lines.append("\n## Figures\n")
        for fig in figures:
            rel = Path(fig).relative_to(Path.cwd())
            lines.append(f"![{fig}]({rel})\n")
        self.report_path.write_text("\n".join(lines))
        print(f"Report written to {self.report_path}")
