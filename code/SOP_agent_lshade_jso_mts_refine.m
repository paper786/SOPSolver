function result = SOP_agent_lshade_jso_mts_refine(problem, seed, options)
% jSO/L-SHADE basin search followed by MTS/pattern refinement.
%
% This combines the effective jSO-RSP search found for F8/F21 with a
% conservative derivative-free coordinate refinement. No gradients or
% benchmark internals are used.
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

base_options = options;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'pre_runtime_sec', 210));
base_options.max_fes = min(max_fes, get_option(options, 'pre_max_fes', floor(0.82 * max_fes)));
base_options.verbose = false;
base = SOP_agent_lshade_jso(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
[best_x, best_raw, curve, eval_count, iter] = refine_best(problem, base.best_position, base.best_value, ...
    remaining_fes, remaining_time, options);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = [base.raw_convergence_curve(:); curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + iter;
result.evaluation_count = base.evaluation_count + eval_count;
result.algorithm_combination = sprintf('jSO/L-SHADE success-history adaptation\nRank-based selective pressure when enabled\nMTS coordinate and stochastic pattern refinement');
result.combination_number = 4;
result.agent_id = 'Agent1';
end

function [best_x, best_raw, curve, eval_count, iter] = refine_best(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
step = get_option(options, 'mts_step', 0.00020) .* span;
min_step = get_option(options, 'mts_min_step', 1e-9) .* max(1, span);
shrink = get_option(options, 'mts_shrink', 0.60);
expand = get_option(options, 'mts_expand', 1.04);
pair_batch = get_option(options, 'mts_pair_batch', 20);
pattern_sigma = get_option(options, 'pattern_sigma', 0.00045) .* span;
block_rate = get_option(options, 'pattern_block_rate', max(0.03, 4 / D));
pattern_samples = get_option(options, 'pattern_samples', 128);
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, 2 * D + pattern_samples))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec && any(step > min_step)
    iter = iter + 1;
    improved = false;
    order = randperm(D);
    for first = 1:pair_batch:D
        dims = order(first:min(D, first + pair_batch - 1));
        dims = dims(step(dims) > min_step(dims));
        if isempty(dims) || eval_count + 2 * numel(dims) > max_fes
            continue;
        end
        candidates = repmat(best_x, 2 * numel(dims), 1);
        for k = 1:numel(dims)
            d = dims(k);
            candidates(2 * k - 1, d) = min(ub(d), best_x(d) + step(d));
            candidates(2 * k, d) = max(lb(d), best_x(d) - step(d));
        end
        values = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(values);
        [trial_raw, idx] = min(values);
        if trial_raw < best_raw
            best_raw = trial_raw;
            best_x = candidates(idx, :);
            touched = dims(ceil(idx / 2));
            step(touched) = min(0.01 * span(touched), step(touched) * expand);
            improved = true;
        else
            step(dims) = max(step(dims) .* shrink, min_step(dims));
        end
        if toc(t_start) >= max_runtime_sec
            break;
        end
    end

    count = min(pattern_samples, max_fes - eval_count);
    if count > 0 && toc(t_start) < max_runtime_sec
        candidates = make_pattern_candidates(best_x, lb, ub, pattern_sigma, block_rate, count);
        values = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(values);
        [trial_raw, idx] = min(values);
        if trial_raw < best_raw
            best_raw = trial_raw;
            best_x = candidates(idx, :);
            pattern_sigma = max(0.94 .* pattern_sigma, min_step);
            improved = true;
        else
            pattern_sigma = max(0.76 .* pattern_sigma, min_step);
        end
    end

    if ~improved
        step = max(step .* shrink, min_step);
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
end

function candidates = make_pattern_candidates(best_x, lb, ub, sigma, block_rate, count)
D = numel(best_x);
candidates = repmat(best_x, count, 1);
for i = 1:count
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randi(D)) = true;
    end
    if rand() < 0.35
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -8), 8);
    else
        noise = randn(1, D);
    end
    x = best_x;
    x(mask) = x(mask) + noise(mask) .* sigma(mask);
    candidates(i, :) = min(max(x, lb), ub);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
