function result = SOP_agent_center_mts_refine(problem, seed, options)
% Center-start Multiple Trajectory Search style coordinate refinement.
%
% This derivative-free metaheuristic starts from the search-space center and
% alternates coordinate probes, randomized coordinate order, and stochastic
% block perturbations. It is useful when center seeding is already effective.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isempty(seed)
    rng(double(seed), 'twister');
else
    rng('shuffle');
end

t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
best_x = get_option(options, 'initial_point', 0.5 * (lb + ub));
best_x = min(max(best_x, lb), ub);
best_raw = SOP_cec_evaluate(best_x, problem);
eval_count = 1;
step = get_option(options, 'initial_step', 0.020) .* span;
min_step = get_option(options, 'min_step', 1e-9) .* span;
shrink = get_option(options, 'shrink', 0.62);
expand = get_option(options, 'expand', 1.08);
pair_batch = get_option(options, 'pair_batch', 20);
pattern_sigma = get_option(options, 'pattern_sigma', 0.006) .* span;
pattern_samples = get_option(options, 'pattern_samples', 96);
block_rate = get_option(options, 'block_rate', max(0.035, 5 / D));
restart_step = get_option(options, 'restart_step', 0.006) .* span;
stall = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, 2 * D + pattern_samples))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec && any(step > min_step)
    iter = iter + 1;
    improved = false;
    order = randperm(D);
    for first = 1:pair_batch:D
        dims = order(first:min(D, first + pair_batch - 1));
        dims = dims(step(dims) > min_step(dims));
        if isempty(dims) || eval_count >= max_fes || toc(t_start) >= max_runtime_sec
            continue;
        end
        count = min(2 * numel(dims), max_fes - eval_count);
        candidates = repmat(best_x, count, 1);
        row = 0;
        touched = zeros(count, 1);
        for k = 1:numel(dims)
            d = dims(k);
            if row + 1 <= count
                row = row + 1;
                candidates(row, d) = min(ub(d), best_x(d) + step(d));
                touched(row) = d;
            end
            if row + 1 <= count
                row = row + 1;
                candidates(row, d) = max(lb(d), best_x(d) - step(d));
                touched(row) = d;
            end
        end
        values = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(values);
        [trial_raw, idx] = min(values);
        if trial_raw < best_raw
            best_raw = trial_raw;
            best_x = candidates(idx, :);
            d = touched(idx);
            if d > 0
                step(d) = min(0.10 * span(d), step(d) * expand);
            end
            improved = true;
            stall = 0;
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
            pattern_sigma = max(0.92 .* pattern_sigma, min_step);
            improved = true;
            stall = 0;
        else
            pattern_sigma = max(0.78 .* pattern_sigma, min_step);
        end
    end

    if ~improved
        stall = stall + 1;
        step = max(step .* shrink, min_step);
    end
    if stall >= 18
        step = max(step, restart_step .* (0.65 + 0.70 * rand(1, D)));
        pattern_sigma = max(pattern_sigma, 0.5 .* restart_step);
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 256, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

runtime = toc(t_start);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = iter;
result.population_num = 1;
result.evaluation_count = eval_count;
if isfield(options, 'initial_point') && ~isempty(options.initial_point)
    start_label = 'Initial-point coordinate probes';
else
    start_label = 'Center-start coordinate probes';
end
result.algorithm_combination = sprintf('Multiple Trajectory Search (MTS)\n%s\nStochastic block pattern refinement', start_label);
result.combination_number = 2;
result.agent_id = 'Agent2';
result.problem = problem;
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
