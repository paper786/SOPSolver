function result = SOP_agent_lshade_local_restart(problem, seed, options)
% Full L-SHADE-family basin search followed by local L-SHADE-family restart.
%
% This keeps the base algorithm's successful evaluation schedule intact,
% then starts a compact local population around its best point for a second
% DE/CMA pass within the remaining runtime.
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
local_algorithm = lower(string(get_option(options, 'local_algorithm', 'lshade_cma')));

base_options = options;
base_options.max_fes = base_fes;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 0.55 * max_runtime_sec));
base_options.verbose = false;
[base, base_label] = run_family(problem, double(seed), base_options, base_algorithm);

remaining_time = max_runtime_sec - toc(t_start);
local_options = options;
local_options.population_num = get_option(options, 'local_population_num', max(80, round(0.55 * get_option(options, 'population_num', 180))));
local_options.max_runtime_sec = max(1, remaining_time);
local_options.max_fes = max(1000, max_fes - base.evaluation_count);
local_options.initial_point = base.best_position;
local_options.initial_radius = get_option(options, 'local_radius', 0.006);
local_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
if get_option(options, 'use_base_elites', false) && isfield(base, 'final_population') && ~isempty(base.final_population)
    local_options.initial_population = build_elite_initial_population(base.best_position, base.final_population, ...
        local_options.population_num, local_options.initial_radius, problem.lb, problem.ub);
end
if lower(string(get_option(options, 'restart_builder', ''))) == "cro_bbo" && ...
        isfield(base, 'final_population') && ~isempty(base.final_population)
    local_options.initial_population = build_cro_bbo_initial_population(base.best_position, base.final_population, ...
        base.final_fitness, local_options.population_num, local_options.initial_radius, problem.lb, problem.ub, options);
    local_options.preserve_initial_population_after_radius = true;
end
local_options.verbose = false;
[local, local_label] = run_family(problem, double(seed) + 65537, local_options, local_algorithm);

if local.record_value < base.record_value
    result = local;
else
    result = base;
end
extra_eval = 0;
extra_iter = 0;
extra_curve = [];
extra_raw_curve = [];
if get_option(options, 'pattern_after', false) && toc(t_start) < max_runtime_sec
    pattern_time = max(1, max_runtime_sec - toc(t_start));
    pattern_fes = max(0, max_fes - base.evaluation_count - local.evaluation_count);
    if pattern_fes > 0
        pattern = pattern_refine(problem, result.best_position, result.best_value, pattern_fes, pattern_time, options);
        if pattern.record_value < result.record_value
            result.best_value = pattern.best_value;
            result.record_value = pattern.record_value;
            result.best_position = pattern.best_position;
        end
        extra_eval = pattern.evaluation_count;
        extra_iter = pattern.iteration;
        extra_curve = pattern.convergence_curve(:);
        extra_raw_curve = pattern.raw_convergence_curve(:);
    end
end
result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + local.evaluation_count + extra_eval;
result.iteration = base.iteration + local.iteration + extra_iter;
result.convergence_curve = [base.convergence_curve(:); local.convergence_curve(:); extra_curve(:)];
result.raw_convergence_curve = [base.raw_convergence_curve(:); local.raw_convergence_curve(:); extra_raw_curve(:)];
if lower(string(get_option(options, 'restart_builder', ''))) == "cro_bbo"
    restart_label = sprintf('CRO/BBO reaction-migration restart with %s', local_label);
else
    restart_label = sprintf('Local restart with %s', local_label);
end
result.algorithm_combination = sprintf('Differential Evolution (DE)\n%s\n%s', base_label, restart_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function [result, label] = run_family(problem, seed, options, algorithm)
switch algorithm
    case "sade"
        options.max_iter = max(1, floor((options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
        result = SOP_agent1_adaptive_de(problem, seed, options);
        label = 'Self-Adaptive DE/current-to-pbest with archive';
    case "lshade"
        result = SOP_agent_lshade(problem, seed, options);
        label = 'L-SHADE success-history adaptation';
    case "lshade_jso"
        result = SOP_agent_lshade_jso(problem, seed, options);
        label = 'jSO/L-SHADE success-history adaptation';
    otherwise
        result = SOP_agent_lshade_cma(problem, seed, options);
        label = 'L-SHADE with elite covariance sampling';
end
end

function initial_population = build_elite_initial_population(best_x, elite_population, NP, radius_scale, lb, ub)
D = numel(best_x);
span = ub - lb;
radius = radius_scale .* span;
elite_count = min(size(elite_population, 1), max(4, floor(0.45 * NP)));
initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
initial_population(1, :) = best_x;
initial_population(2:elite_count + 1, :) = elite_population(1:elite_count, :);
remaining = NP - elite_count - 1;
if remaining > 0
    donor = elite_population(randi(elite_count, remaining, 1), :);
    rows = elite_count + 2:NP;
    initial_population(rows, :) = donor + randn(remaining, D) .* repmat(0.35 .* radius, remaining, 1);
end
initial_population = min(max(initial_population, lb), ub);
end

function initial_population = build_cro_bbo_initial_population(best_x, elite_population, elite_fitness, NP, radius_scale, lb, ub, options)
D = numel(best_x);
span = ub - lb;
radius = radius_scale .* span;
if nargin < 3 || isempty(elite_fitness) || numel(elite_fitness) ~= size(elite_population, 1)
    elite_fitness = (1:size(elite_population, 1))';
end
[elite_fitness, order] = sort(elite_fitness(:)); %#ok<ASGLU>
elite_population = elite_population(order, :);
elite_count = min(size(elite_population, 1), max(6, round(get_option(options, 'cro_bbo_elite_rate', 0.45) * NP)));
initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
initial_population(1, :) = best_x;
copy_count = min(elite_count, NP - 1);
initial_population(2:copy_count + 1, :) = elite_population(1:copy_count, :);
start_row = copy_count + 2;
if start_row > NP
    initial_population = min(max(initial_population, lb), ub);
    return;
end
inv_fit = max(elite_fitness(1:elite_count)) - elite_fitness(1:elite_count) + eps;
habitat_prob = inv_fit ./ sum(inv_fit);
cum_prob = cumsum(habitat_prob);
for row = start_row:NP
    rank = (row - start_row) / max(1, NP - start_row);
    parent_a = elite_population(randi(elite_count), :);
    parent_b = elite_population(randi(elite_count), :);
    child = parent_a;
    op = rand();
    if op < get_option(options, 'cro_synthesis_rate', 0.36)
        mask = rand(1, D) < get_option(options, 'cro_synthesis_mask_rate', 0.42);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        child(mask) = parent_b(mask);
        child = child + randn(1, D) .* (get_option(options, 'cro_synthesis_noise', 0.20) .* radius);
    elseif op < get_option(options, 'cro_synthesis_rate', 0.36) + get_option(options, 'cro_decomposition_rate', 0.30)
        block_rate = get_option(options, 'cro_decomposition_block_rate', max(0.04, 5 / D));
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -7), 7);
        child(mask) = best_x(mask) + noise(mask) .* (get_option(options, 'cro_decomposition_scale', 0.38) .* radius(mask));
    else
        child = 0.70 .* parent_a + 0.30 .* best_x;
        collision = parent_a - parent_b;
        child = child + get_option(options, 'cro_collision_scale', 0.22) .* rand(1, D) .* collision;
    end
    immigration = min(0.86, get_option(options, 'bbo_restart_immigration_base', 0.20) + ...
        get_option(options, 'bbo_restart_immigration_span', 0.55) * rank);
    mutation = min(0.42, get_option(options, 'bbo_restart_mutation_base', 0.06) + ...
        get_option(options, 'bbo_restart_mutation_span', 0.14) * rank);
    for d = 1:D
        if rand() < immigration
            donor = find(cum_prob >= rand(), 1, 'first');
            child(d) = elite_population(donor, d);
        end
        if rand() < mutation
            child(d) = child(d) + randn() .* get_option(options, 'bbo_restart_mutation_scale', 0.18) .* radius(d);
        end
    end
    if rand() < get_option(options, 'cro_bbo_best_pull_rate', 0.30)
        child = child + get_option(options, 'cro_bbo_best_pull', 0.16) .* rand(1, D) .* (best_x - child);
    end
    initial_population(row, :) = min(max(child, lb), ub);
end
initial_population = min(max(initial_population, lb), ub);
end

function pattern = pattern_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
batch = get_option(options, 'pattern_samples', 96);
sigma = get_option(options, 'pattern_sigma', 0.0015) .* span;
min_sigma = get_option(options, 'pattern_min_sigma', 1e-9) .* max(1, span);
block_rate = get_option(options, 'pattern_block_rate', min(0.12, max(0.03, 6 / D)));
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / batch)), 1);
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = repmat(best_x, count, 1);
    for i = 1:count
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        noise = randn(1, D);
        if rand() < 0.3
            cauchy_noise = tan(pi * (rand(1, D) - 0.5));
            noise(mask) = min(max(cauchy_noise(mask), -8), 8);
        end
        candidates(i, mask) = candidates(i, mask) + noise(mask) .* sigma(mask);
    end
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
        sigma = max(0.985 .* sigma, min_sigma);
    else
        sigma = max(0.82 .* sigma, min_sigma);
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
pattern = struct();
pattern.best_value = best_raw;
pattern.record_value = SOP_cec_record_value(best_raw, problem);
pattern.best_position = best_x;
pattern.raw_convergence_curve = curve;
pattern.convergence_curve = SOP_cec_record_value(curve, problem);
pattern.evaluation_count = eval_count;
pattern.iteration = iter;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
