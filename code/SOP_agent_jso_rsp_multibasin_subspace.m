function result = SOP_agent_jso_rsp_multibasin_subspace(problem, seed, options)
% jSO/RSP basin search with warm restarts and archive-guided subspace search.
%
% This Agent2 candidate keeps the currently useful jSO/RSP pressure, then
% uses small Cauchy warm restarts and stochastic coordinate-subspace
% perturbations around the retained basin. It is derivative-free and does
% not use any benchmark internals.
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
base_fraction = get_option(options, 'base_fraction', 0.72);
restart_fraction = get_option(options, 'restart_fraction', 0.18);

base_options = make_jso_options(options);
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.max_runtime_sec = max(1, base_fraction * max_runtime_sec);
base_options.verbose = false;
base = SOP_agent_lshade_jso(problem, double(seed), base_options);

best_x = base.best_position;
best_raw = base.best_value;
eval_count = base.evaluation_count;
curve = base.raw_convergence_curve(:);
anchors = best_x;
iter_count = base.iteration;

restart_count = get_option(options, 'restart_count', 2);
restart_population = max(24, round(get_option(options, 'restart_population_factor', 0.45) * ...
    get_option(options, 'population_num', 180)));
restart_radii = get_option(options, 'restart_radii', [0.010, 0.0035]);
restart_budget_total = floor(restart_fraction * max_fes);

for r = 1:restart_count
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        break;
    end
    remaining_time = max(1, max_runtime_sec - toc(t_start));
    remaining_fes = max_fes - eval_count;
    local_options = make_jso_options(options);
    local_options.population_num = restart_population;
    local_options.max_fes = min(remaining_fes, max(1000, floor(restart_budget_total / max(1, restart_count))));
    local_options.max_runtime_sec = min(remaining_time, max(1, remaining_time * 0.42));
    local_options.initial_point = best_x;
    local_options.initial_radius = pick_radius(restart_radii, r);
    local_options.initial_cauchy = true;
    local_options.include_center = true;
    local_options.verbose = false;
    local = SOP_agent_lshade_jso(problem, double(seed) + 7919 * r, local_options);
    eval_count = eval_count + local.evaluation_count;
    iter_count = iter_count + local.iteration;
    anchors = [anchors; local.best_position]; %#ok<AGROW>
    if local.best_value < best_raw
        best_raw = local.best_value;
        best_x = local.best_position;
    end
    curve = append_monotone_curve(curve, local.raw_convergence_curve(:), best_raw);
end

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - eval_count);
[best_x, best_raw, refine_curve, refine_eval, refine_iter] = subspace_refine( ...
    problem, best_x, best_raw, anchors, remaining_fes, remaining_time, options);
eval_count = eval_count + refine_eval;
iter_count = iter_count + refine_iter;
curve = append_monotone_curve(curve, refine_curve(:), best_raw);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = iter_count;
result.evaluation_count = eval_count;
result.population_num = get_option(options, 'population_num', base.population_num);
result.algorithm_combination = sprintf('jSO/RSP L-SHADE basin search\nCauchy warm multi-basin restart\nArchive-guided stochastic subspace refinement');
result.combination_number = 5;
result.agent_id = 'Agent2';
result.problem = problem;
end

function options = make_jso_options(options)
options.include_center = get_option(options, 'include_center', true);
options.ranked_r1 = get_option(options, 'ranked_r1', true);
options.rank_pressure = get_option(options, 'rank_pressure', 1.7);
options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
options.p_rate_start = get_option(options, 'p_rate_start', 0.20);
options.p_rate_end = get_option(options, 'p_rate_end', 0.040);
options.weight_start = get_option(options, 'weight_start', 0.62);
options.weight_end = get_option(options, 'weight_end', 1.36);
options.archive_factor_start = get_option(options, 'archive_factor_start', 1.8);
options.archive_factor_end = get_option(options, 'archive_factor_end', 3.2);
end

function [best_x, best_raw, curve, eval_count, iter] = subspace_refine(problem, best_x, best_raw, anchors, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
samples = get_option(options, 'subspace_samples', max(80, round(0.26 * get_option(options, 'population_num', 180))));
subspace_rate = get_option(options, 'subspace_rate', max(0.035, 5 / D));
sigma = get_option(options, 'refine_sigma', 0.0026) .* span;
min_sigma = get_option(options, 'min_sigma', 1e-8) .* max(1, span);
reset_sigma = get_option(options, 'reset_sigma', 0.008) .* span;
anchor_rate = get_option(options, 'anchor_rate', 0.28);
eval_count = 0;
iter = 0;
stall = 0;
curve = zeros(max(1, ceil(max_fes / max(1, samples))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(samples, max_fes - eval_count);
    centers = repmat(best_x, count, 1);
    if size(anchors, 1) > 1
        use_anchor = rand(count, 1) < anchor_rate;
        anchor_rows = randi(size(anchors, 1), count, 1);
        centers(use_anchor, :) = 0.72 .* centers(use_anchor, :) + 0.28 .* anchors(anchor_rows(use_anchor), :);
    end
    masks = rand(count, D) < subspace_rate;
    empty = ~any(masks, 2);
    if any(empty)
        rows = find(empty);
        cols = randi(D, numel(rows), 1);
        for k = 1:numel(rows)
            masks(rows(k), cols(k)) = true;
        end
    end
    noise = randn(count, D);
    cauchy_mask = rand(count, D) < 0.30;
    cauchy_noise = tan(pi * (rand(count, D) - 0.5));
    cauchy_noise = min(max(cauchy_noise, -8), 8);
    noise(cauchy_mask) = cauchy_noise(cauchy_mask);
    candidates = centers + masks .* noise .* repmat(sigma, count, 1);
    candidates = min(max(candidates, lb), ub);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        anchors = [best_x; anchors]; %#ok<AGROW>
        sigma = max(0.982 .* sigma, min_sigma);
        stall = 0;
    else
        sigma = max(0.955 .* sigma, min_sigma);
        stall = stall + 1;
    end
    if stall >= 28
        sigma = max(sigma, reset_sigma .* (0.75 + 0.50 * rand(1, D)));
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
end

function curve = append_monotone_curve(curve, tail, current_best)
if isempty(tail)
    return;
end
if isempty(curve)
    curve = tail;
else
    tail = min(tail, curve(end));
    curve = [curve; tail]; %#ok<AGROW>
end
curve = cummin([curve; current_best]);
end

function radius = pick_radius(radii, idx)
if numel(radii) == 1
    radius = radii;
else
    radius = radii(min(idx, numel(radii)));
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
