from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent1 import (
    GeneratedAlgorithm,
    OpenAICompatibleClient,
    ProblemSpec,
    SolverConfig,
    ensure_product_directories,
    normalize_matlab_function_name,
    update_statistics_workbook,
    write_text,
)


@dataclass
class CandidateStats:
    agent_id: str
    algorithm_path: Path
    ok: bool
    values: List[float] = field(default_factory=list)
    runtimes: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    best: float = math.inf
    ave: float = math.inf
    std: float = math.inf
    worst: float = math.inf
    time: float = math.inf
    iteration: int = 0
    population_num: int = 0
    algorithm_combination: str = ""
    combination_number: Optional[int] = None
    agent_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def gap(self) -> Optional[float]:
        return self.agent_report.get("gap")


@dataclass
class EvaluationOutcome:
    problem: ProblemSpec
    agent1: CandidateStats
    agent2: CandidateStats
    winner: CandidateStats
    retained: bool
    best_record: Dict[str, Any]
    feedback_path: Optional[Path] = None
    feedback_text: str = ""
    final_path: Optional[Path] = None


class EvaluateAgent:
    def __init__(
        self,
        config: SolverConfig,
        *,
        matlab_command: str = "matlab",
        pilot_runs: int = 1,
        final_runs: int = 10,
    ) -> None:
        self.config = config
        self.matlab_command = matlab_command
        self.pilot_runs = pilot_runs
        self.final_runs = final_runs
        self.client = OpenAICompatibleClient(
            config.api_key,
            config.api_base_url,
            config.model,
            config.request_timeout_sec,
        )

    def evaluate_candidates(
        self,
        problem: ProblemSpec,
        agent1_algorithm: GeneratedAlgorithm,
        agent2_algorithm: GeneratedAlgorithm,
        best_record: Dict[str, Any],
        *,
        round_index: int,
    ) -> EvaluationOutcome:
        ensure_product_directories(self.config)
        agent1_stats = self.run_matlab_algorithm(
            agent1_algorithm.path,
            problem,
            agent_id="Agent1",
            runs=self.pilot_runs,
            seed_offset=1000 + round_index * 100,
        )
        agent2_stats = self.run_matlab_algorithm(
            agent2_algorithm.path,
            problem,
            agent_id="Agent2",
            runs=self.pilot_runs,
            seed_offset=2000 + round_index * 100,
        )
        self._attach_gap(agent1_stats, best_record)
        self._attach_gap(agent2_stats, best_record)
        winner = choose_winner(agent1_stats, agent2_stats)
        retained = is_retained(winner, problem, best_record)

        outcome = EvaluationOutcome(
            problem=problem,
            agent1=agent1_stats,
            agent2=agent2_stats,
            winner=winner,
            retained=retained,
            best_record=best_record,
        )
        if retained:
            outcome.final_path = self.save_final_algorithm(problem, winner.algorithm_path)
        else:
            outcome.feedback_path, outcome.feedback_text = self.write_feedback(
                problem,
                outcome,
                round_index=round_index,
            )
        return outcome

    def run_formal_experiment_and_record(
        self,
        problem: ProblemSpec,
        winner_path: Path,
        *,
        agent_id: str,
        workbook_name: str = "result.xlsx",
        seed_offset: int = 9000,
    ) -> CandidateStats:
        stats = self.run_matlab_algorithm(
            winner_path,
            problem,
            agent_id=agent_id,
            runs=self.final_runs,
            seed_offset=seed_offset,
        )
        workbook = self.config.statistics_dir / workbook_name
        update_statistics_workbook(workbook, problem, stats_to_workbook_row(stats, problem))
        return stats

    def run_matlab_algorithm(
        self,
        algorithm_path: Path,
        problem: ProblemSpec,
        *,
        agent_id: str,
        runs: int,
        seed_offset: int,
    ) -> CandidateStats:
        function_name = algorithm_path.stem
        values: List[float] = []
        runtimes: List[float] = []
        errors: List[str] = []
        metadata: Dict[str, Any] = {}

        for run_idx in range(1, runs + 1):
            seed = seed_offset + run_idx
            one = self._run_one_matlab_call(function_name, problem, seed)
            if one.get("ok"):
                value = to_float(one.get("record_value"))
                if value is None:
                    value = to_float(one.get("best_value"))
                if value is None or not math.isfinite(value):
                    errors.append(f"Run {run_idx}: result has no finite best/record value.")
                    continue
                values.append(value)
                runtime = to_float(one.get("runtime"))
                if runtime is not None and math.isfinite(runtime):
                    runtimes.append(runtime)
                metadata = one
            else:
                errors.append(f"Run {run_idx}: {one.get('error', 'unknown MATLAB error')}")

        stats = summarize_candidate(
            agent_id=agent_id,
            algorithm_path=algorithm_path,
            values=values,
            runtimes=runtimes,
            errors=errors,
            metadata=metadata,
            problem=problem,
        )
        return stats

    def _run_one_matlab_call(self, function_name: str, problem: ProblemSpec, seed: int) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="sopsolver_matlab_", ignore_cleanup_errors=True) as temp_dir:
            temp = Path(temp_dir)
            output_json = temp / "result.json"
            driver = temp / "run_candidate_driver.m"
            write_text(
                driver,
                build_matlab_driver(
                    project_root=self.config.project_root,
                    function_name=function_name,
                    output_json=output_json,
                    problem=problem,
                    seed=seed,
                ),
            )
            command = [
                self.matlab_command,
            ]
            if os.name == "nt":
                command.append("-wait")
            command.extend(["-batch", f"run('{matlab_string(driver)}')"])
            try:
                completed = run_process(command, timeout_sec=problem.time_limit_sec + 90)
            except FileNotFoundError:
                return {
                    "ok": False,
                    "error": f"MATLAB command not found: {self.matlab_command}",
                }
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "error": f"MATLAB run timeout after {problem.time_limit_sec + 90} seconds.",
                }

            if output_json.exists():
                try:
                    return json.loads(output_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    return {
                        "ok": False,
                        "error": f"Cannot parse MATLAB JSON output: {exc}",
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    }
            return {
                "ok": False,
                "error": f"MATLAB did not produce {output_json}. Return code {completed.returncode}.",
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }

    def save_final_algorithm(self, problem: ProblemSpec, winner_path: Path) -> Path:
        final_path = self.config.final_code_dir / f"{problem.file_stem}.m"
        code = winner_path.read_text(encoding="utf-8")
        try:
            code = normalize_matlab_function_name(code, problem.file_stem)
            write_text(final_path, code)
        except Exception:
            wrapper = f"""function result = {problem.file_stem}(seed, options)
% Final retained wrapper for {problem.function_name} {problem.dimension}D.
if nargin < 1, seed = []; end
if nargin < 2, options = struct(); end
root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root_dir, 'compare_code'));
result = {winner_path.stem}(seed, options);
end
"""
            write_text(final_path, wrapper)
        return final_path

    def write_feedback(
        self,
        problem: ProblemSpec,
        outcome: EvaluationOutcome,
        *,
        round_index: int,
    ) -> tuple[Path, str]:
        feedback_path = self.config.evaluate_prompt_dir / f"{problem.file_stem}_prompt.md"
        deterministic = build_deterministic_feedback(problem, outcome, round_index)
        llm_feedback = ""
        try:
            llm_feedback = self._ask_llm_for_feedback(problem, outcome, deterministic)
        except Exception as exc:
            llm_feedback = f"\n\n## LLM feedback unavailable\n{exc}\n"
        text = deterministic + "\n\n" + llm_feedback
        write_text(feedback_path, text)
        return feedback_path, text

    def _ask_llm_for_feedback(
        self,
        problem: ProblemSpec,
        outcome: EvaluationOutcome,
        deterministic: str,
    ) -> str:
        code1 = safe_read_excerpt(outcome.agent1.algorithm_path, 7000)
        code2 = safe_read_excerpt(outcome.agent2.algorithm_path, 7000)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Evaluate_Agent in SOPSolver. Provide concise, honest, "
                    "actionable feedback for improving MATLAB metaheuristic code. "
                    "Do not invent experimental results."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem: {problem.function_name} {problem.dimension}D
Round feedback summary:
{deterministic}

Agent1 code excerpt:
```matlab
{code1}
```

Agent2 code excerpt:
```matlab
{code2}
```

Write a feedback markdown section with:
- useful operators to keep;
- ineffective mechanisms to discard;
- next design directions for Agent1 and Agent2;
- whether to change the combination strategy number.
""",
            },
        ]
        return self.client.chat(
            messages,
            temperature=0.3,
            max_tokens=min(self.config.max_tokens, 4096),
        )

    @staticmethod
    def _attach_gap(stats: CandidateStats, best_record: Dict[str, Any]) -> None:
        best_ref = to_float(best_record.get("best"))
        if best_ref is not None and math.isfinite(stats.best):
            stats.agent_report["gap"] = stats.best - best_ref


def build_matlab_driver(
    *,
    project_root: Path,
    function_name: str,
    output_json: Path,
    problem: ProblemSpec,
    seed: int,
) -> str:
    root = matlab_string(project_root)
    out = matlab_string(output_json)
    suite = problem.suite
    return f"""try
    addpath('{root}');
    addpath(fullfile('{root}', 'compare_code'));
    addpath(fullfile('{root}', 'final_code'));
    addpath(fullfile('{root}', 'code'));
    addpath(fullfile('{root}', '{suite}'));
    options = struct();
    options.max_iteration = {int(problem.max_iterations)};
    options.max_iter = {int(problem.max_iterations)};
    options.population_num = {int(problem.population_num)};
    options.pop_size = {int(problem.population_num)};
    options.max_runtime_sec = {int(problem.time_limit_sec)};
    result = feval('{function_name}', {int(seed)}, options);
    out = struct();
    out.ok = true;
    out.best_value = read_numeric(result, {{'best_value','Best_score','best_score','fbest','best'}}, NaN);
    out.record_value = read_numeric(result, {{'record_value','best_value','Best_score','best_score','fbest','best'}}, NaN);
    out.runtime = read_numeric(result, {{'runtime','time','Time'}}, NaN);
    out.iteration = read_numeric(result, {{'iteration','iterations','max_iteration','Max_iterations'}}, {int(problem.max_iterations)});
    out.population_num = read_numeric(result, {{'population_num','pop_size','population','SearchAgents_no'}}, {int(problem.population_num)});
    out.combination_number = read_numeric(result, {{'combination_number','strategy_number'}}, NaN);
    out.evaluation_count = read_numeric(result, {{'evaluation_count','fes','FES'}}, NaN);
    out.algorithm_combination = read_text_field(result, {{'algorithm_combination','algorithm','name'}}, '');
    out.agent_id = read_text_field(result, {{'agent_id','agent'}}, '');
catch ME
    out = struct();
    out.ok = false;
    out.error = getReport(ME, 'extended', 'hyperlinks', 'off');
end
fid = fopen('{out}', 'w', 'n', 'UTF-8');
if fid < 0
    error('Cannot open result json for writing.');
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(out));

function value = read_numeric(result, names, default_value)
value = default_value;
if isnumeric(result) && ~isempty(result)
    value = double(result(1));
    return;
end
if ~isstruct(result)
    return;
end
for i = 1:numel(names)
    name = names{{i}};
    if isfield(result, name)
        candidate = result.(name);
        if isnumeric(candidate) && ~isempty(candidate)
            value = double(candidate(1));
            return;
        end
    end
end
end

function text_value = read_text_field(result, names, default_value)
text_value = default_value;
if ~isstruct(result)
    return;
end
for i = 1:numel(names)
    name = names{{i}};
    if isfield(result, name)
        candidate = result.(name);
        if isstring(candidate) || ischar(candidate)
            text_value = char(candidate);
            return;
        end
    end
end
end
"""


def summarize_candidate(
    *,
    agent_id: str,
    algorithm_path: Path,
    values: Sequence[float],
    runtimes: Sequence[float],
    errors: Sequence[str],
    metadata: Dict[str, Any],
    problem: ProblemSpec,
) -> CandidateStats:
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    ok = len(finite_values) > 0
    if ok:
        best = min(finite_values)
        ave = statistics.fmean(finite_values)
        worst = max(finite_values)
        std = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0
    else:
        best = ave = worst = std = math.inf
    finite_times = [float(t) for t in runtimes if math.isfinite(float(t))]
    avg_time = statistics.fmean(finite_times) if finite_times else math.inf
    combination_number = to_int(metadata.get("combination_number"))
    return CandidateStats(
        agent_id=agent_id,
        algorithm_path=algorithm_path,
        ok=ok,
        values=list(finite_values),
        runtimes=list(finite_times),
        errors=list(errors),
        best=best,
        ave=ave,
        std=std,
        worst=worst,
        time=avg_time,
        iteration=to_int(metadata.get("iteration")) or problem.max_iterations,
        population_num=to_int(metadata.get("population_num")) or problem.population_num,
        algorithm_combination=str(metadata.get("algorithm_combination") or ""),
        combination_number=combination_number,
        agent_report=dict(metadata),
    )


def choose_winner(agent1: CandidateStats, agent2: CandidateStats) -> CandidateStats:
    if agent1.ok and not agent2.ok:
        return agent1
    if agent2.ok and not agent1.ok:
        return agent2
    if agent1.best < agent2.best:
        return agent1
    if agent2.best < agent1.best:
        return agent2
    if agent1.std < agent2.std:
        return agent1
    if agent2.std < agent1.std:
        return agent2
    if agent1.time <= agent2.time:
        return agent1
    return agent2


def is_retained(stats: CandidateStats, problem: ProblemSpec, best_record: Dict[str, Any]) -> bool:
    best_ref = to_float(best_record.get("best"))
    if best_ref is None or not math.isfinite(stats.best):
        return False
    gap = stats.best - best_ref
    if problem.strict_gap:
        return gap < problem.accept_gap
    return gap <= problem.accept_gap


def stats_to_workbook_row(stats: CandidateStats, problem: ProblemSpec) -> Dict[str, Any]:
    return {
        "best": sci4(stats.best),
        "ave": sci4(stats.ave),
        "std": sci4(stats.std),
        "worst": sci4(stats.worst),
        "time": round(stats.time, 4) if math.isfinite(stats.time) else "",
        "iteration": int(stats.iteration or problem.max_iterations),
        "population_num": int(stats.population_num or problem.population_num),
        "algorithm_combination": stats.algorithm_combination,
        "combination_number": stats.combination_number if stats.combination_number is not None else "",
    }


def build_deterministic_feedback(
    problem: ProblemSpec,
    outcome: EvaluationOutcome,
    round_index: int,
) -> str:
    best_ref = outcome.best_record.get("best")
    lines = [
        f"# {problem.file_stem} Evaluate_Agent Feedback",
        "",
        f"- Round: {round_index}",
        f"- Problem: {problem.function_name} {problem.dimension}D",
        f"- BestRecord reference: {best_ref}",
        f"- Acceptance gap: {problem.accept_gap}; strict gap: {problem.strict_gap}",
        "",
        "## Candidate Results",
        candidate_line(outcome.agent1),
        candidate_line(outcome.agent2),
        "",
        f"Winner so far: {outcome.winner.agent_id}",
        f"Retained: {outcome.retained}",
        "",
        "## Agent1 Errors",
        "\n".join(f"- {e[:1000]}" for e in outcome.agent1.errors) or "- None",
        "",
        "## Agent2 Errors",
        "\n".join(f"- {e[:1000]}" for e in outcome.agent2.errors) or "- None",
        "",
        "## Required Next Step",
    ]
    if outcome.retained:
        lines.append("- Retain the winner and run formal independent experiments.")
    else:
        lines.append("- Redesign or materially modify operators; do not only tune parameters.")
        lines.append("- Keep useful mechanisms from the better candidate and abandon failing components.")
    return "\n".join(lines)


def candidate_line(stats: CandidateStats) -> str:
    gap = stats.agent_report.get("gap")
    gap_text = "NA" if gap is None else f"{gap:.6g}"
    return (
        f"- {stats.agent_id}: ok={stats.ok}, best={stats.best:.12g}, "
        f"ave={stats.ave:.12g}, std={stats.std:.6g}, worst={stats.worst:.12g}, "
        f"time={stats.time:.4g}, gap={gap_text}, algorithm={stats.algorithm_combination}"
    )


def matlab_string(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def safe_read_excerpt(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n% [TRUNCATED]"
    return text


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(float(value))
    except Exception:
        return None


def sci4(value: float) -> Any:
    if not math.isfinite(value):
        return ""
    if value == 0:
        return 0
    return f"{value:.4e}"


def copy_without_metadata(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def run_process(command: Sequence[str], *, timeout_sec: int) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        stdout, stderr = process.communicate(timeout=10)
        raise subprocess.TimeoutExpired(command, timeout_sec, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
