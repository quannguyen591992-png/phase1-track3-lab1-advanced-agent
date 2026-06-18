from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3, mode: Literal["mock", "gemini"] = "mock") -> None:
    load_dotenv()
    os.environ["REFLEXION_RUNTIME_MODE"] = mode
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    print(f"[cyan]Mode:[/cyan] {mode} | [cyan]Dataset:[/cyan] {dataset} | [cyan]Examples:[/cyan] {len(examples)}")
    react_records = []
    for idx, example in enumerate(examples, start=1):
        print(f"[yellow]ReAct[/yellow] {idx}/{len(examples)} {example.qid}: {example.question}")
        react_records.append(react.run(example))
    reflexion_records = []
    for idx, example in enumerate(examples, start=1):
        print(f"[yellow]Reflexion[/yellow] {idx}/{len(examples)} {example.qid}: {example.question}")
        reflexion_records.append(reflexion.run(example))
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
