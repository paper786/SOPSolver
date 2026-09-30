function result = SOP_agent_lshade_restart_mts_refine(problem, seed, options)
% L-SHADE-CMA local restart followed by MTS coordinate refinement.
%
% This is a metaheuristic hybrid for near-threshold high-dimensional cases:
% L-SHADE-CMA performs basin search, then MTS probes positive/negative moves
% along coordinates with adaptive step-size shrink/expand.
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

restart_options = options;
restart_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'pre_runtime_sec', 0.80 * max_runtime_sec));
restart_options.max_fes = min(max_fes, get_option(options, 'pre_max_fes', max_fes));
restart_options.base_algorithm = get_option(options, 'base_algorithm', 'lshade_cma');
restart_options.local_algorithm = get_option(options, 'local_algorithm', 'lshade_cma');
restart_options.base_max_fes = get_option(options, 'base_max_fes', 1000000);
restart_options.local_radius = get_option(options, 'local_radius', 0.006);
restart_options.local_population_num = get_option(options, 'local_population_num', max(80, round(0.55 * get_option(options, 'population_num', 180))));
restart_options.use_base_elites = get_option(options, 'use_base_elites', true);
restart_options.pattern_after = false;
restart_options.verbose = false;
restart = SOP_agent_lshade_local_restart(problem, double(seed), restart_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - restart.evaluation_count);
[best_x, best_raw, refine_curve, refine_eval, refine_iter] = mts_refine(problem, restart.best_position, restart.best_value, ...
    remaining_fes, remaining_time, options);

result = restart;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = [restart.raw_convergence_curve(:); refine_curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.runtime = toc(t_start);
result.iteration = restart.iteration + refine_iter;
result.evaluation_count = restart.evaluation_count + refine_eval;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE-CMA local restart\nMTS adaptive coordinate local search');
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function [best_x, best_raw, curve, eval_count, iter] = mts_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
step = get_option(options, 'mts_step', 0.001) .* span;
min_step = get_option(options, 'mts_min_step', 1e-10) .* max(1, span);
shrink = get_option(options, 'mts_shrink', 0.55);
expand = get_option(options, 'mts_expand', 1.08);
permute_each = get_option(options, 'mts_permute_each', true);
batch_pair_count = get_option(options, 'mts_pair_batch', 12);
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
    for first = 1:batch_pair_count:D
        dims = order(first:min(D, first + batch_pair_count - 1));
        active = dims(step(dims) > min_step(dims));
        if isempty(active) || eval_count + 2 * numel(active) > max_fes
            continue;
        end
        candidates = repmat(best_x, 2 * numel(active), 1);
        for k = 1:numel(active)
            d = active(k);
            candidates(2 * k - 1, d) = min(ub(d), best_x(d) + step(d));
            candidates(2 * k, d) = max(lb(d), best_x(d) - step(d));
        end
        values = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(values);
        if toc(t_start) >= max_runtime_sec
            break;
        end
        for k = 1:numel(active)
            d = active(k);
            pair = values(2 * k - 1:2 * k);
            [trial_raw, which] = min(pair);
            if trial_raw < best_raw
                best_raw = trial_raw;
                best_x = candidates(2 * k - 2 + which, :);
                step(d) = min(step(d) * expand, 0.02 * span(d));
                improved_any = true;
            else
                step(d) = max(step(d) * shrink, min_step(d));
            end
        end
    end
    if ~improved_any
        step = max(step .* shrink, min_step);
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
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
