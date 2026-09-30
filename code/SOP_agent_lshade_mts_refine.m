function result = SOP_agent_lshade_mts_refine(problem, seed, options)
% L-SHADE-family basin search followed by MTS-style coordinate local search.
%
% Multiple Trajectory Search style coordinate probing is a derivative-free
% metaheuristic local refinement. It tests positive/negative coordinate
% moves and shrinks step sizes when no improvement is found.
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
base_fes = min(max_fes, get_option(options, 'base_max_fes', 1000000));
base_algorithm = lower(string(get_option(options, 'base_algorithm', 'lshade_cma')));

base_options = options;
base_options.max_fes = base_fes;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 0.55 * max_runtime_sec));
base_options.verbose = false;
switch base_algorithm
    case "lshade"
        base = SOP_agent_lshade(problem, double(seed), base_options);
        base_label = 'L-SHADE success-history adaptation';
    otherwise
        base = SOP_agent_lshade_cma(problem, double(seed), base_options);
        base_label = 'L-SHADE with elite covariance sampling';
end

remaining_time = max_runtime_sec - toc(t_start);
remaining_fes = max(0, max_fes - base.evaluation_count);
[best_x, best_raw, refine_curve, refine_eval, refine_iter] = mts_refine(problem, base.best_position, base.best_value, ...
    remaining_fes, remaining_time, options);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = [base.raw_convergence_curve(:); refine_curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + refine_iter;
result.evaluation_count = base.evaluation_count + refine_eval;
result.algorithm_combination = sprintf('Differential Evolution (DE)\n%s\nMTS-style adaptive coordinate local search', base_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function [best_x, best_raw, curve, eval_count, iter] = mts_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
step = get_option(options, 'mts_step', 0.01) .* span;
min_step = get_option(options, 'mts_min_step', 1e-9) .* max(1, span);
shrink = get_option(options, 'mts_shrink', 0.5);
expand = get_option(options, 'mts_expand', 1.15);
permute_each = get_option(options, 'mts_permute_each', true);
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, 2 * D))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec && any(step > min_step)
    iter = iter + 1;
    improved_any = false;
    if permute_each
        order = randperm(D);
    else
        order = 1:D;
    end
    for idx = 1:D
        d = order(idx);
        if step(d) <= min_step(d) || eval_count + 2 > max_fes
            continue;
        end
        plus = best_x;
        minus = best_x;
        plus(d) = min(ub(d), plus(d) + step(d));
        minus(d) = max(lb(d), minus(d) - step(d));
        values = SOP_cec_evaluate([plus; minus], problem);
        eval_count = eval_count + 2;
        if toc(t_start) >= max_runtime_sec
            break;
        end
        [trial_raw, which] = min(values);
        if trial_raw < best_raw
            best_raw = trial_raw;
            if which == 1
                best_x = plus;
            else
                best_x = minus;
            end
            step(d) = min(step(d) * expand, 0.05 * span(d));
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
