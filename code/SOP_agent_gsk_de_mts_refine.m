function result = SOP_agent_gsk_de_mts_refine(problem, seed, options)
% GSK + L-SHADE/L-SHADE-CMA followed by MTS-style coordinate refinement.
%
% This keeps the successful GSK-to-DE hybrid as the basin finder and spends
% only the remaining time on derivative-free coordinate probes. No gradient,
% Newton, SQP, or benchmark internals are used.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if isempty(seed)
    seed = randi(1000000);
end

t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
base_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', max_runtime_sec - 12));

base_options = options;
base_options.max_runtime_sec = max(1, base_runtime_sec);
base_options.verbose = false;
base = SOP_agent_gsk_de_hybrid(problem, double(seed), base_options);

remaining_time = max_runtime_sec - toc(t_start);
remaining_fes = max(0, max_fes - base.evaluation_count);
if remaining_time > 0.5 && remaining_fes > 0
    [best_x, best_raw, curve, eval_count, iter] = mts_refine(problem, base.best_position, base.best_value, ...
        remaining_fes, remaining_time, options);
else
    best_x = base.best_position;
    best_raw = base.best_value;
    curve = [];
    eval_count = 0;
    iter = 0;
end

result = base;
if SOP_cec_record_value(best_raw, problem) < base.record_value
    result.best_value = best_raw;
    result.record_value = SOP_cec_record_value(best_raw, problem);
    result.best_position = best_x;
end
base_raw_curve = get_curve(base, 'raw_convergence_curve');
result.raw_convergence_curve = [base_raw_curve(:); curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + iter;
result.evaluation_count = base.evaluation_count + eval_count;
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK)\nL-SHADE-CMA differential refinement\nMTS-style coordinate local refinement');
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function curve = get_curve(result, field_name)
if isfield(result, field_name) && ~isempty(result.(field_name))
    curve = result.(field_name);
elseif isfield(result, 'convergence_curve') && ~isempty(result.convergence_curve)
    curve = result.convergence_curve;
else
    curve = result.best_value;
end
end

function [best_x, best_raw, curve, eval_count, iter] = mts_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
step = get_option(options, 'mts_step', 0.001) .* span;
min_step = get_option(options, 'mts_min_step', 1e-10) .* max(1, span);
shrink = get_option(options, 'mts_shrink', 0.5);
expand = get_option(options, 'mts_expand', 1.10);
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, 2 * D))), 1);

while eval_count + 2 <= max_fes && toc(t_start) < max_runtime_sec && any(step > min_step)
    iter = iter + 1;
    improved_any = false;
    order = randperm(D);
    for k = 1:D
        d = order(k);
        if eval_count + 2 > max_fes || toc(t_start) >= max_runtime_sec
            break;
        end
        plus = best_x;
        minus = best_x;
        plus(d) = min(ub(d), plus(d) + step(d));
        minus(d) = max(lb(d), minus(d) - step(d));
        values = SOP_cec_evaluate([plus; minus], problem);
        eval_count = eval_count + 2;
        [trial_raw, which] = min(values);
        if trial_raw < best_raw
            best_raw = trial_raw;
            if which == 1
                best_x = plus;
            else
                best_x = minus;
            end
            step(d) = min(step(d) * expand, 0.02 * span(d));
            improved_any = true;
        else
            step(d) = max(step(d) * shrink, min_step(d));
        end
    end
    if ~improved_any
        step = max(step .* shrink, min_step);
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
