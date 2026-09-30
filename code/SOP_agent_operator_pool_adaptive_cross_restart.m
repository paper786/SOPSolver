function result = SOP_agent_operator_pool_adaptive_cross_restart(problem, seed, options)
% Operator-pool basin search with adaptive component-cross local restarts.
%
% Agent1 candidate for near-line CEC2017 F20 cases. The first stage keeps
% the DE/GSK/RIME/CMA operator pool as basin finder. The second stage adapts
% the local restart radius from the final elite basin geometry, then crosses
% L-SHADE-CMA and jSO local components. A final objective-only elite
% difference/block pattern refinement uses the same adaptive radius.
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
NP = get_option(options, 'population_num', 180);

base_options = options;
base_options.profile = get_option(options, 'profile', 'de_gsk_cma');
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 205));
base_options.max_fes = min(max_fes, get_option(options, 'base_max_fes', floor(0.82 * max_fes)));
base_options.max_iter = max(1, floor((base_options.max_fes - NP) / NP));
base_options.verbose = false;
base = SOP_agent_operator_pool(problem, double(seed), base_options);

best = base;
total_eval = base.evaluation_count;
total_iter = base.iteration;
raw_curve = get_curve(base, true);
record_curve = get_curve(base, false);

radius = estimate_elite_radius(base, options, problem);
min_radius = get_option(options, 'radius_min', 0.00035);
max_radius = get_option(options, 'radius_max', 0.010);
components = component_sequence(get_option(options, 'component_order', 'cma_jso'));
previous_best = best.best_value;

for phase = 1:numel(components)
    remaining_time = max_runtime_sec - toc(t_start);
    remaining_fes = max_fes - total_eval;
    if remaining_time <= 2 || remaining_fes < max(12000, 35 * NP)
        break;
    end

    local_options = options;
    local_options.population_num = get_option(options, 'local_population_num', max(64, round(0.36 * NP)));
    local_options.max_runtime_sec = max(1, 0.72 * remaining_time);
    reserve_fes = min(max(20000, round(0.12 * remaining_fes)), max(0, remaining_fes - local_options.population_num));
    local_options.max_fes = max(local_options.population_num, floor(remaining_fes - reserve_fes));
    if phase < numel(components)
        local_options.max_fes = max(local_options.population_num, floor(0.55 * local_options.max_fes));
    end
    local_options.initial_point = best.best_position;
    local_options.initial_radius = radius;
    local_options.initial_cauchy = true;
    local_options.initial_population = build_cross_initial_population(best.best_position, base, ...
        local_options.population_num, radius, problem.lb, problem.ub);
    local_options.cma_rate = get_option(options, 'cma_rate', 0.15);
    local_options.elite_rate = get_option(options, 'elite_rate', 0.22);
    local_options.cma_interval = get_option(options, 'cma_interval', 10);
    local_options.ranked_r1 = true;
    local_options.rank_pressure = get_option(options, 'rank_pressure', 1.65);
    local_options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
    local_options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
    local_options.p_rate_start = get_option(options, 'p_rate_start', 0.18);
    local_options.p_rate_end = get_option(options, 'p_rate_end', 0.045);
    local_options.archive_factor_start = get_option(options, 'archive_factor_start', 1.6);
    local_options.archive_factor_end = get_option(options, 'archive_factor_end', 3.0);
    local_options.verbose = false;

    local = run_component(problem, double(seed) + 65537 * phase, local_options, components(phase));
    total_eval = total_eval + local.evaluation_count;
    total_iter = total_iter + local.iteration;
    raw_curve = [raw_curve(:); get_curve(local, true)]; %#ok<AGROW>
    record_curve = [record_curve(:); get_curve(local, false)]; %#ok<AGROW>

    if local.record_value < best.record_value
        improvement = max(0, previous_best - local.best_value);
        best = local;
        previous_best = best.best_value;
        if improvement > get_option(options, 'large_improvement_raw', 1e-6)
            radius = max(min_radius, min(max_radius, radius * get_option(options, 'radius_success_shrink', 0.58)));
        else
            radius = max(min_radius, min(max_radius, radius * get_option(options, 'radius_small_success_shrink', 0.72)));
        end
    else
        radius = max(min_radius, min(max_radius, radius * get_option(options, 'radius_failure_shrink', 0.42)));
    end
end

extra_eval = 0;
extra_iter = 0;
if get_option(options, 'cross_refine_after', true) && toc(t_start) < max_runtime_sec
    remaining_fes = max(0, max_fes - total_eval);
    remaining_time = max(1, max_runtime_sec - toc(t_start));
    if remaining_fes > 0
        refine = cross_block_refine(problem, best.best_position, best.best_value, base, ...
            remaining_fes, remaining_time, radius, options);
        if refine.record_value < best.record_value
            best.best_value = refine.best_value;
            best.record_value = refine.record_value;
            best.best_position = refine.best_position;
        end
        extra_eval = refine.evaluation_count;
        extra_iter = refine.iteration;
        raw_curve = [raw_curve(:); refine.raw_convergence_curve(:)];
        record_curve = [record_curve(:); refine.convergence_curve(:)];
    end
end

result = best;
result.runtime = toc(t_start);
result.evaluation_count = total_eval + extra_eval;
result.iteration = total_iter + extra_iter;
result.raw_convergence_curve = raw_curve(:);
if isempty(record_curve)
    result.convergence_curve = SOP_cec_record_value(raw_curve(:), problem);
else
    result.convergence_curve = record_curve(:);
end
result.algorithm_combination = sprintf('Adaptive operator-pool metaheuristic\nDE/GSK/RIME/elite covariance basin finder\nAdaptive-radius L-SHADE-CMA and jSO component crossover\nElite-difference stochastic block refinement');
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function local = run_component(problem, seed, options, component)
switch lower(string(component))
    case "jso"
        local = SOP_agent_lshade_jso(problem, seed, options);
    case "lshade"
        local = SOP_agent_lshade(problem, seed, options);
    otherwise
        local = SOP_agent_lshade_cma(problem, seed, options);
end
end

function components = component_sequence(order_value)
order_value = lower(string(order_value));
switch order_value
    case "jso_cma"
        components = ["jso", "lshade_cma"];
    case "cma_lshade"
        components = ["lshade_cma", "lshade"];
    otherwise
        components = ["lshade_cma", "jso"];
end
end

function radius = estimate_elite_radius(base, options, problem)
default_radius = get_option(options, 'local_radius', 0.004);
radius = default_radius;
if ~isfield(base, 'final_population') || isempty(base.final_population) || ...
        ~isfield(base, 'final_fitness') || isempty(base.final_fitness)
    return;
end
population = base.final_population;
fitness = base.final_fitness;
[~, order] = sort(fitness);
elite_count = min(numel(order), max(8, round(get_option(options, 'radius_elite_rate', 0.12) * size(population, 1))));
elites = population(order(1:elite_count), :);
span = max(problem.ub - problem.lb, eps);
distances = sqrt(mean(((elites - base.best_position) ./ span) .^ 2, 2));
distances = distances(isfinite(distances) & distances > 0);
if isempty(distances)
    return;
end
estimated = median(distances) * get_option(options, 'radius_scale', 1.35);
radius = 0.55 * default_radius + 0.45 * estimated;
radius = min(get_option(options, 'radius_max', 0.010), max(get_option(options, 'radius_min', 0.00035), radius));
end

function initial_population = build_cross_initial_population(best_x, base, NP, radius_scale, lb, ub)
D = numel(best_x);
span = ub - lb;
radius = radius_scale .* span;
initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
initial_population(1, :) = best_x;
if isfield(base, 'final_population') && ~isempty(base.final_population)
    elites = base.final_population;
    if isfield(base, 'final_fitness') && ~isempty(base.final_fitness)
        [~, order] = sort(base.final_fitness);
        elites = elites(order, :);
    end
    elite_count = min(size(elites, 1), max(4, floor(0.35 * NP)));
    initial_population(2:elite_count + 1, :) = elites(1:elite_count, :);
    rows = elite_count + 2:NP;
    for k = rows
        a = randi(elite_count);
        b = randi(elite_count);
        if a == b
            b = mod(b, elite_count) + 1;
        end
        if rand() < 0.45
            x = best_x + 0.35 * (elites(a, :) - elites(b, :));
        else
            x = 0.72 * best_x + 0.28 * elites(a, :);
        end
        initial_population(k, :) = x + randn(1, D) .* (0.35 .* radius);
    end
end
initial_population = min(max(initial_population, lb), ub);
end

function refine = cross_block_refine(problem, best_x, best_raw, base, max_fes, max_runtime_sec, radius_scale, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
sigma = max(get_option(options, 'cross_sigma_min', 1e-7), radius_scale) .* span;
sigma = max(sigma, get_option(options, 'cross_sigma_floor', 1e-9) .* max(1, span));
block_rate = get_option(options, 'cross_block_rate', min(0.08, max(0.025, 4 / D)));
batch = get_option(options, 'cross_samples', 128);
elite_population = [];
if isfield(base, 'final_population') && ~isempty(base.final_population)
    elite_population = base.final_population;
    if isfield(base, 'final_fitness') && ~isempty(base.final_fitness)
        [~, order] = sort(base.final_fitness);
        elite_population = elite_population(order, :);
    end
    elite_population = elite_population(1:min(size(elite_population, 1), max(6, round(0.18 * size(elite_population, 1)))), :);
end
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);
stall = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
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
        if ~isempty(elite_population) && rand() < 0.55
            a = randi(size(elite_population, 1));
            b = randi(size(elite_population, 1));
            if a == b
                b = mod(b, size(elite_population, 1)) + 1;
            end
            diff_vec = elite_population(a, :) - elite_population(b, :);
            x(mask) = x(mask) + get_option(options, 'cross_diff_weight', 0.18) .* diff_vec(mask);
        end
        candidates(i, :) = min(max(x, lb), ub);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        sigma = max(0.93 .* sigma, get_option(options, 'cross_sigma_floor', 1e-9) .* max(1, span));
        stall = 0;
    else
        sigma = max(0.74 .* sigma, get_option(options, 'cross_sigma_floor', 1e-9) .* max(1, span));
        stall = stall + 1;
    end
    if stall >= 8
        sigma = max(sigma, get_option(options, 'cross_reset_sigma', 0.00035) .* span);
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
refine = struct();
refine.best_value = best_raw;
refine.record_value = SOP_cec_record_value(best_raw, problem);
refine.best_position = best_x;
refine.raw_convergence_curve = curve;
refine.convergence_curve = SOP_cec_record_value(curve, problem);
refine.evaluation_count = eval_count;
refine.iteration = iter;
end

function curve = get_curve(result, raw)
if raw
    if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
        curve = result.raw_convergence_curve(:);
    else
        curve = result.best_value;
    end
else
    if isfield(result, 'convergence_curve') && ~isempty(result.convergence_curve)
        curve = result.convergence_curve(:);
    else
        curve = result.record_value;
    end
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
