from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agent1 import (
    Agent1,
    GeneratedAlgorithm,
    OfflineTemplateAgent,
    ProblemSpec,
    SolverConfig,
    find_best_record,
    load_openpyxl,
    read_text,
)
from agent2 import Agent2
from evaluate_agent import EvaluateAgent, is_retained


# Optional direct API key input position. Prefer environment variable
# SOPSOLVER_API_KEY or --api-key so secrets are not saved in code.
API_KEY = ""


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    api_key = args.api_key or os.getenv("SOPSOLVER_API_KEY", "") or API_KEY
    api_base_url = args.api_base_url or os.getenv("SOPSOLVER_API_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.getenv("SOPSOLVER_MODEL", "gpt-5.5")

    config = SolverConfig(
        project_root=project_root,
        api_key=api_key,
        api_base_url=api_base_url,
        model=model,
        temperature=args.temperature,
        request_timeout_sec=args.request_timeout,
        max_tokens=args.max_tokens,
    )

    if args.api_smoke_test:
        return api_smoke_test(config)

    if args.doctor:
        return doctor(config, args.matlab_command)

    prompt_file = Path(args.prompt_file) if args.prompt_file else project_root / "prompt_SOPSolver.md"
    call_rules_file = (
        Path(args.call_rules_file)
        if args.call_rules_file
        else project_root / "CEC2014_2017_2019_calling_rules.md"
    )
    prompt_text = read_text(prompt_file, max_chars=config.prompt_char_limit)
    call_rules_text = read_text(call_rules_file, max_chars=18000) if call_rules_file.exists() else ""

    problems = load_problem_specs(args)
    if args.dry_run:
        print("SOPSolver dry run")
        print(f"Project root: {project_root}")
        print(f"Prompt file: {prompt_file}")
        print(f"Call rules file: {call_rules_file}")
        print(f"Model: {model}")
        for problem in problems:
            best_record = find_best_record(config, problem)
            print(
                f"- {problem.function_name} {problem.dimension}D, "
                f"BestRecord={best_record.get('best')}, "
                f"max_iter={problem.max_iterations}, pop={problem.population_num}"
            )
        return 0

    if not api_key and not args.offline_template:
        print("API key is empty. Set SOPSOLVER_API_KEY, pass --api-key, or edit API_KEY in main.py.")
        print("For a no-key pipeline smoke test, run with --offline-template.")
        return 2

    if args.offline_template:
        agent1 = OfflineTemplateAgent(config, "Agent1")
        agent2 = OfflineTemplateAgent(config, "Agent2")
    else:
        agent1 = Agent1(config)
        agent2 = Agent2(config)
    evaluator = EvaluateAgent(
        config,
        matlab_command=args.matlab_command,
        pilot_runs=args.pilot_runs,
        final_runs=args.final_runs,
    )

    for problem in problems:
        run_problem_loop(
            problem,
            config=config,
            prompt_text=prompt_text,
            call_rules_text=call_rules_text,
            agent1=agent1,
            agent2=agent2,
            evaluator=evaluator,
            max_rounds=args.max_rounds,
            write_single_result=not args.no_single_result,
            force_formal_on_last_round=args.force_formal_on_last_round,
        )
    return 0


def run_problem_loop(
    problem: ProblemSpec,
    *,
    config: SolverConfig,
    prompt_text: str,
    call_rules_text: str,
    agent1: Agent1,
    agent2: Agent2,
    evaluator: EvaluateAgent,
    max_rounds: int,
    write_single_result: bool,
    force_formal_on_last_round: bool,
) -> None:
    best_record = find_best_record(config, problem)
    print(f"\n=== {problem.function_name} {problem.dimension}D ===")
    print(f"BestRecord: {best_record.get('best')} from {best_record.get('source')}")

    feedback_text = load_existing_feedback(config, problem)
    previous_agent1_code = ""
    previous_agent2_code = ""
    single_result_recorded = False
    last_agent1: Optional[GeneratedAlgorithm] = None
    last_agent2: Optional[GeneratedAlgorithm] = None

    for round_index in range(1, max_rounds + 1):
        print(f"\nRound {round_index}/{max_rounds}: generating Agent1 candidate...")
        last_agent1 = agent1.generate_algorithm(
            problem,
            prompt_text=prompt_text,
            call_rules_text=call_rules_text,
            best_record=best_record,
            feedback_text=feedback_text,
            peer_summary="Agent2 has not generated a candidate in this round yet.",
            previous_code=previous_agent1_code,
            round_index=round_index,
        )
        previous_agent1_code = last_agent1.code

        print(f"Round {round_index}/{max_rounds}: generating Agent2 candidate...")
        last_agent2 = agent2.generate_algorithm(
            problem,
            prompt_text=prompt_text,
            call_rules_text=call_rules_text,
            best_record=best_record,
            feedback_text=feedback_text,
            peer_summary=f"Agent1 candidate saved at {last_agent1.path.name}.",
            previous_code=previous_agent2_code,
            round_index=round_index,
        )
        previous_agent2_code = last_agent2.code

        print(f"Round {round_index}/{max_rounds}: running Evaluate_Agent pilot tests...")
        outcome = evaluator.evaluate_candidates(
            problem,
            last_agent1,
            last_agent2,
            best_record,
            round_index=round_index,
        )
        print_candidate_summary("Agent1", outcome.agent1)
        print_candidate_summary("Agent2", outcome.agent2)
        print(f"Winner: {outcome.winner.agent_id}; retained={outcome.retained}")

        if write_single_result and not single_result_recorded and is_retained(outcome.agent1, problem, best_record):
            print("Agent1 first satisfactory candidate detected; running 10-run single_result record...")
            evaluator.run_formal_experiment_and_record(
                problem,
                outcome.agent1.algorithm_path,
                agent_id="Agent1",
                workbook_name="single_result.xlsx",
                seed_offset=5000,
            )
            single_result_recorded = True

        should_stop = outcome.retained or (
            force_formal_on_last_round and round_index == max_rounds and outcome.winner.ok
        )
        if should_stop:
            if outcome.final_path is None:
                outcome.final_path = evaluator.save_final_algorithm(problem, outcome.winner.algorithm_path)
            print(f"Final algorithm saved: {outcome.final_path}")
            print("Running formal independent experiments for result.xlsx...")
            evaluator.run_formal_experiment_and_record(
                problem,
                outcome.final_path,
                agent_id=outcome.winner.agent_id,
                workbook_name="result.xlsx",
                seed_offset=9000,
            )
            print(f"Finished {problem.function_name} {problem.dimension}D.")
            return

        feedback_text = outcome.feedback_text
        if outcome.feedback_path:
            print(f"Feedback saved: {outcome.feedback_path}")

    print(f"Stopped after {max_rounds} rounds without a retained runnable winner.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOPSolver multi-agent API runner")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--call-rules-file", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--matlab-command", default=os.getenv("SOPSOLVER_MATLAB", "matlab"))

    parser.add_argument("--suite", default="CEC2017")
    parser.add_argument("--function", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--accept-gap", type=float, default=0.0)
    parser.add_argument("--strict-gap", action="store_true")
    parser.add_argument("--cases-file", default="")

    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--pilot-runs", type=int, default=1)
    parser.add_argument("--final-runs", type=int, default=10)
    parser.add_argument("--no-single-result", action="store_true")
    parser.add_argument("--force-formal-on-last-round", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--offline-template", action="store_true")
    parser.add_argument("--api-smoke-test", action="store_true")
    return parser.parse_args()


def resolve_project_root(raw_project_root: str) -> Path:
    if raw_project_root:
        return Path(raw_project_root).resolve()
    env_root = os.getenv("SOPSOLVER_PROJECT_ROOT", "")
    if env_root:
        return Path(env_root).resolve()

    start = Path(__file__).resolve().parent
    markers = [
        "prompt_SOPSolver.md",
        str(Path("Statistics") / "BestRecord.xlsx"),
        str(Path("CEC2017") / "CEC2017_evaluate.m"),
    ]
    for parent in [start, *start.parents]:
        if all((parent / marker).exists() for marker in markers):
            return parent

    #provide your own root or document position
    candidates = [
        Path.cwd(),
        Path.home() / "xxx",
        Path.home() / "xxx",
        # Path.home() / "Downloads" / "zoo_LLM_cec" / "zoo_LLM_cec",
        # Path.home() / "Downloads" / "zoo_LLM_cec",
    ]
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in markers):
            return candidate.resolve()
    return start.resolve()


def doctor(config: SolverConfig, matlab_command: str) -> int:
    print("SOPSolver doctor")
    print(f"Project root: {config.project_root}")
    checks = [
        ("prompt_SOPSolver.md", config.project_root / "prompt_SOPSolver.md"),
        ("BestRecord.xlsx", config.statistics_dir / "BestRecord.xlsx"),
        ("Algorithm_information.xlsx", config.statistics_dir / "Algorithm_information.xlsx"),
        ("CEC2017_evaluate.m", config.project_root / "CEC2017" / "CEC2017_evaluate.m"),
        ("CEC2019_evaluate.m", config.project_root / "CEC2019" / "CEC2019_evaluate.m"),
    ]
    ok = True
    for label, path in checks:
        exists = path.exists()
        ok = ok and exists
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {path}")
    try:
        openpyxl = load_openpyxl()
        print(f"[OK] openpyxl: {openpyxl.__version__}")
    except Exception as exc:
        ok = False
        print(f"[MISSING] openpyxl: {exc}")
    matlab_path = find_executable(matlab_command)
    if matlab_path:
        print(f"[OK] MATLAB command: {matlab_path}")
    else:
        ok = False
        print(f"[MISSING] MATLAB command not found: {matlab_command}")
    print(f"API key configured: {'yes' if bool(config.api_key) else 'no'}")
    print(f"Model: {config.model}")
    print(f"API base URL: {config.api_base_url}")
    return 0 if ok else 1


def find_executable(command: str) -> str:
    import shutil

    if not command:
        return ""
    path = Path(command)
    if path.exists():
        return str(path.resolve())
    found = shutil.which(command)
    return found or ""


def load_problem_specs(args: argparse.Namespace) -> List[ProblemSpec]:
    if not args.cases_file:
        return [
            ProblemSpec(
                suite=args.suite,
                func_num=args.function,
                dimension=args.dimension,
                max_iterations=args.max_iterations,
                population_num=args.population,
                time_limit_sec=args.time_limit,
                accept_gap=args.accept_gap,
                strict_gap=args.strict_gap,
            )
        ]
    path = Path(args.cases_file)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = read_cases_csv_or_lines(path)
    return [problem_from_row(row, args) for row in rows]


def read_cases_csv_or_lines(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if "," in first_line and any(name in first_line.lower() for name in ["suite", "function", "dimension"]):
        return list(csv.DictReader(text.splitlines()))
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = [part.strip() for part in clean.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Invalid case line: {line}")
        rows.append({"suite": parts[0], "function": parts[1], "dimension": parts[2]})
    return rows


def problem_from_row(row: Dict[str, object], defaults: argparse.Namespace) -> ProblemSpec:
    return ProblemSpec(
        suite=str(row.get("suite") or row.get("Suite") or defaults.suite),
        func_num=int(row.get("function") or row.get("func_num") or row.get("Function") or defaults.function),
        dimension=int(row.get("dimension") or row.get("Dimension") or defaults.dimension),
        max_iterations=int(row.get("max_iterations") or row.get("iteration") or defaults.max_iterations),
        population_num=int(row.get("population") or row.get("population_num") or defaults.population),
        time_limit_sec=int(row.get("time_limit") or defaults.time_limit),
        accept_gap=float(row.get("accept_gap") or defaults.accept_gap),
        strict_gap=parse_bool(row.get("strict_gap", defaults.strict_gap)),
    )


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(value)


def load_existing_feedback(config: SolverConfig, problem: ProblemSpec) -> str:
    path = config.evaluate_prompt_dir / f"{problem.file_stem}_prompt.md"
    if path.exists():
        return read_text(path, max_chars=config.feedback_char_limit)
    return ""


def print_candidate_summary(label: str, stats) -> None:
    gap = stats.agent_report.get("gap")
    gap_text = "NA" if gap is None else f"{gap:.6g}"
    print(
        f"{label}: ok={stats.ok}, best={stats.best:.12g}, "
        f"ave={stats.ave:.12g}, std={stats.std:.6g}, gap={gap_text}"
    )
    if stats.errors:
        print(f"{label} errors: {stats.errors[0][:500]}")


def api_smoke_test(config: SolverConfig) -> int:
    from agent1 import APIError, OpenAICompatibleClient

    if not config.api_key:
        print("API key is empty. Set SOPSOLVER_API_KEY or pass --api-key.")
        return 2
    client = OpenAICompatibleClient(
        config.api_key,
        config.api_base_url,
        config.model,
        config.request_timeout_sec,
    )
    try:
        answer = client.chat(
            [
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "API smoke test."},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        print(f"API smoke test response: {answer.strip()}")
        return 0
    except APIError as exc:
        print(f"API smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
