function result = SOP_agent_lsrtde_rsp_cma(problem, seed, options)
% LSRTDE-RSP-CMA style adaptive DE.
%
% This candidate follows the temporary CEC2017 design recommendation:
% success-history L-SHADE/L-SRTDE backbone, rank pressure, external archive,
% success-rate strategy selection, eigen-coordinate crossover, partial
% restart, and an optional short CMA-ES evolutionary tail.
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
NP_init = get_option(options, 'population_num', max(180, 12 * D));
NP_min = get_option(options, 'min_population_num', 4);
max_fes = get_option(options, 'max_fes', 10000 * D);
H = get_option(options, 'memory_size', 8);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

cma_tail_fraction = get_option(options, 'cma_tail_fraction', 0.06);
cma_reserve = min(max_fes - NP_init, max(0, round(cma_tail_fraction * max_fes)));
main_max_fes = max(NP_init + 1, max_fes - cma_reserve);

population = lb + rand(NP_init, D) .* span;
initial_population = get_option(options, 'initial_population', []);
if ~isempty(initial_population)
    rows = min(NP_init, size(initial_population, 1));
    population(1:rows, :) = min(max(initial_population(1:rows, :), lb), ub);
end
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);

archive = zeros(0, D);
mu_F = get_option(options, 'mu_F_init', 0.48) * ones(1, H);
mu_CR = get_option(options, 'mu_CR_init', 0.72) * ones(1, H);
memory_index = 1;
strategy_success = ones(1, 5);
strategy_trials = 5 * ones(1, 5);
strategy_delta = ones(1, 5) * eps;
curve = zeros(max(1, ceil(max_fes / NP_min)), 1);
iter = 0;
stall_count = 0;

eig_rate = get_option(options, 'eig_rate', 0.28);
eig_interval = get_option(options, 'eig_interval', 14);
elite_rate = get_option(options, 'elite_rate', 0.24);
eig_basis = eye(D);
eig_center = mean(population, 1);
restart_after = get_option(options, 'restart_after', 34);

while eval_count < main_max_fes && size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    progress = eval_count / max(1, max_fes);
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if eig_rate > 0 && (iter == 1 || mod(iter, eig_interval) == 0)
        [eig_basis, eig_center] = elite_eigen_basis(population, span, elite_rate);
    end

    combined = [population; archive];
    pool_size = size(combined, 1);
    p_max = get_option(options, 'p_rate', get_option(options, 'p_max', 0.20));
    p_min = 2 / NP;
    p_rate = p_min + rand() * max(0, p_max - p_min);
    p_num = max(2, min(NP, round(p_rate * NP)));
    strategy_probs = strategy_probabilities(strategy_success, strategy_trials, strategy_delta, progress, options);
    trial_population = population;
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);
    used_strategy = zeros(NP, 1);

    for i = 1:NP
        mem = randi(H);
        F = sample_F(mu_F(mem), progress, options);
        CR = sample_CR(mu_CR(mem), progress, options);
        sampled_F(i) = F;
        sampled_CR(i) = CR;
        strategy = sample_discrete(strategy_probs);
        if NP < 6 && strategy > 2
            strategy = 1;
        end
        used_strategy(i) = strategy;
        mutant = make_mutant(strategy, population, combined, pool_size, fitness, i, F, p_num, progress, options);
        if eig_rate > 0 && rand() < eig_rate
            trial = eigen_crossover(population(i, :), mutant, CR, eig_basis, eig_center);
        else
            trial = binomial_crossover(population(i, :), mutant, CR);
        end
        trial_population(i, :) = repair_bounds(trial, population(i, :), lb, ub, get_option(options, 'boundary_mode', 'midpoint'));
    end

    remaining = main_max_fes - eval_count;
    if remaining <= 0
        break;
    end
    if remaining < NP
        trial_population = trial_population(1:remaining, :);
        sampled_F = sampled_F(1:remaining);
        sampled_CR = sampled_CR(1:remaining);
        used_strategy = used_strategy(1:remaining);
    end
    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end

    success_F = [];
    success_CR = [];
    success_delta = [];
    rows = numel(trial_fitness);
    for i = 1:rows
        strategy_trials(used_strategy(i)) = strategy_trials(used_strategy(i)) + 1;
        if trial_fitness(i) <= fitness(i)
            delta = max(0, fitness(i) - trial_fitness(i));
            archive = [archive; population(i, :)]; %#ok<AGROW>
            population(i, :) = trial_population(i, :);
            fitness(i) = trial_fitness(i);
            success_delta(end + 1, 1) = delta; %#ok<AGROW>
            success_F(end + 1, 1) = sampled_F(i); %#ok<AGROW>
            success_CR(end + 1, 1) = sampled_CR(i); %#ok<AGROW>
            strategy_success(used_strategy(i)) = strategy_success(used_strategy(i)) + 1;
            strategy_delta(used_strategy(i)) = strategy_delta(used_strategy(i)) + delta;
        end
    end

    archive_factor = get_option(options, 'archive_factor', 1.7);
    archive_limit = max(0, round(archive_factor * NP));
    if size(archive, 1) > archive_limit
        archive = archive(randperm(size(archive, 1), archive_limit), :);
    end

    if ~isempty(success_delta) && sum(success_delta) > 0
        weights = success_delta ./ sum(success_delta);
        mu_F(memory_index) = sum(weights .* (success_F .^ 2)) / max(eps, sum(weights .* success_F));
        mu_CR(memory_index) = sum(weights .* success_CR);
        memory_index = memory_index + 1;
        if memory_index > H
            memory_index = 1;
        end
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
        stall_count = 0;
    else
        stall_count = stall_count + 1;
    end
    curve(iter) = best_raw;

    if get_option(options, 'partial_restart', true) && stall_count >= restart_after && ...
            eval_count < main_max_fes && toc(t_start) < max_runtime_sec
        [population, fitness, extra_eval, local_best, local_x] = partial_restart(problem, population, fitness, ...
            best_position, lb, ub, span, main_max_fes - eval_count, progress, options);
        eval_count = eval_count + extra_eval;
        if local_best < best_raw
            best_raw = local_best;
            best_position = local_x;
        end
        stall_count = max(0, floor(0.35 * stall_count));
    end

    target_NP = target_population_size(NP_init, NP_min, eval_count, max_fes, options);
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness);
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
end

[final_fitness, final_order] = sort(fitness);
final_population = population(final_order, :);

if cma_reserve > 0 && eval_count < max_fes && toc(t_start) < max_runtime_sec
    cma_options = struct();
    cma_options.population_num = get_option(options, 'cma_population_num', max(18, round(0.10 * NP_init)));
    cma_options.max_fes = min(cma_reserve, max_fes - eval_count);
    cma_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
    cma_options.initial_point = best_position;
    cma_options.initial_population = final_population;
    cma_options.seed_covariance = true;
    cma_options.covariance_seed_count = min(size(final_population, 1), get_option(options, 'covariance_seed_count', max(12, 2 * D)));
    cma_options.seed_covariance_blend = get_option(options, 'seed_covariance_blend', 0.52);
    cma_options.seed_covariance_ridge = get_option(options, 'seed_covariance_ridge', 0.08);
    cma_options.local_sigma = get_option(options, 'local_sigma', 0.00045);
    cma_options.restart_sigma = get_option(options, 'restart_sigma', 0.00018);
    cma_options.restart_limit = get_option(options, 'restart_limit', 1);
    cma_options.verbose = false;
    cma_result = SOP_agent_cma_es(problem, double(seed) + 7919, cma_options);
    eval_count = eval_count + cma_result.evaluation_count;
    if cma_result.record_value < SOP_cec_record_value(best_raw, problem)
        best_raw = cma_result.best_value;
        best_position = cma_result.best_position;
    end
end

curve = curve(1:iter);
runtime = toc(t_start);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = iter;
result.population_num = NP_init;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf(['L-SRTDE/L-SHADE-RSP adaptive DE\n' ...
    'Success-history F/CR, LPSR, external archive\n' ...
    'Success-rate multi-mutation pool with rank-based selective pressure\n' ...
    'Eigen-coordinate crossover, partial restart, optional CMA-ES tail']);
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('LSRTDE-RSP-CMA finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function mutant = make_mutant(strategy, population, combined, pool_size, fitness, i, F, p_num, progress, options)
NP = size(population, 1);
rank_pressure = get_option(options, 'rank_pressure', 1.8);
pbest = rank_weighted_index(p_num, rank_pressure + 0.8 * progress);
switch strategy
    case 1
        r1 = rank_weighted_index_except(NP, i, rank_pressure, progress);
        r2 = random_pool_index(pool_size, [i r1]);
        mutant = population(i, :) + F .* (population(pbest, :) - population(i, :)) + ...
            F .* (population(r1, :) - combined(r2, :));
    case 2
        ids = random_indices(NP, 2, i);
        rand_pull = 0.45 + 0.35 * rand();
        mutant = population(i, :) + rand_pull .* (population(ids(1), :) - population(i, :)) + ...
            F .* (population(ids(2), :) - combined(random_pool_index(pool_size, [i ids(1) ids(2)]), :));
    case 3
        ids = random_indices(NP, 3, i);
        mutant = population(ids(1), :) + F .* (population(ids(2), :) - population(ids(3), :));
    case 4
        r1 = rank_weighted_index_except(NP, i, rank_pressure + 0.5, progress);
        r2 = random_pool_index(pool_size, [i r1]);
        weight = get_option(options, 'weight_start', 0.68) + ...
            (get_option(options, 'weight_end', 1.28) - get_option(options, 'weight_start', 0.68)) * progress;
        mutant = population(i, :) + weight .* F .* (population(pbest, :) - population(i, :)) + ...
            F .* (population(r1, :) - combined(r2, :));
    otherwise
        ids = random_indices(NP, 3, i);
        winner_count = max(2, round(0.18 * NP));
        winner = rank_weighted_index(winner_count, 2.2 + progress);
        loser_weight = min(0.55, max(0.10, rank_index(fitness, i) / NP));
        mutant = population(i, :) + F .* (population(1, :) - population(i, :)) + ...
            (0.35 + 0.45 * loser_weight) .* F .* (population(winner, :) - population(ids(1), :)) + ...
            0.50 .* F .* (population(ids(2), :) - population(ids(3), :));
end
end

function probs = strategy_probabilities(success, trials, delta, progress, options)
rate = success ./ max(trials, 1);
gain = delta ./ max(sum(delta), eps);
adaptive = 0.72 .* rate ./ max(sum(rate), eps) + 0.28 .* gain;
base = get_option(options, 'strategy_base', [0.40, 0.13, 0.12, 0.25, 0.10]);
base = base ./ sum(base);
base(3) = base(3) * (1.25 - 0.85 * progress);
base(4) = base(4) * (0.85 + 0.35 * progress);
base(5) = base(5) * (0.80 + 0.55 * progress);
epsilon = get_option(options, 'strategy_epsilon', 0.035);
probs = 0.54 .* base + 0.46 .* adaptive + epsilon;
probs = probs ./ sum(probs);
end

function [population, fitness, eval_count, best_raw, best_position] = partial_restart(problem, population, fitness, best_position, lb, ub, span, remaining_fes, progress, options)
eval_count = 0;
[fitness, order] = sort(fitness);
population = population(order, :);
best_raw = fitness(1);
best_position = population(1, :);
NP = size(population, 1);
D = size(population, 2);
elite_keep = max(2, min(NP - 2, round(get_option(options, 'restart_elite_rate', 0.12) * NP)));
restart_count = min(NP - elite_keep, remaining_fes);
if restart_count <= 0
    return;
end
radius_scale = get_option(options, 'restart_radius', 0.055) * max(0.08, (1 - progress));
radius = radius_scale .* span;
candidates = repmat(best_position, restart_count, 1);
elite_count = max(elite_keep, min(NP, round(0.24 * NP)));
center = mean(population(1:elite_count, :), 1);
for k = 1:restart_count
    mode = rand();
    if mode < 0.45
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -8), 8);
        candidates(k, :) = best_position + noise .* radius;
    elseif mode < 0.78
        anchor = population(randi(elite_count), :);
        candidates(k, :) = anchor + randn(1, D) .* (0.65 .* radius);
    else
        candidates(k, :) = lb + rand(1, D) .* span;
        blend = 0.35 + 0.35 * rand();
        candidates(k, :) = blend .* candidates(k, :) + (1 - blend) .* center;
    end
end
candidates = min(max(candidates, lb), ub);
candidate_fitness = SOP_cec_evaluate(candidates, problem);
eval_count = numel(candidate_fitness);
replace_rows = elite_keep + (1:restart_count);
population(replace_rows, :) = candidates;
fitness(replace_rows) = candidate_fitness;
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
end

function target_NP = target_population_size(NP_init, NP_min, eval_count, max_fes, options)
progress = eval_count / max(1, max_fes);
alpha = get_option(options, 'lpsr_alpha', 1.0);
if abs(alpha - 1.0) < 1e-12
    target_NP = round(NP_init - (NP_init - NP_min) * progress);
else
    target_NP = round(NP_min + (NP_init - NP_min) * (1 - progress) ^ alpha);
end
target_NP = max(NP_min, target_NP);
end

function F = sample_F(mu, progress, options)
if progress < 0.45
    mu = max(mu, get_option(options, 'early_F_floor', 0.34));
end
F = mu + 0.1 * tan(pi * (rand() - 0.5));
tries = 0;
while F <= 0 && tries < 20
    F = mu + 0.1 * tan(pi * (rand() - 0.5));
    tries = tries + 1;
end
if F <= 0
    F = 0.5;
end
F = min(get_option(options, 'F_max', 1.0), max(0.05, F));
end

function CR = sample_CR(mu, progress, options)
if progress < 0.30
    mu = max(mu, get_option(options, 'early_CR_floor', 0.62));
end
CR = min(1, max(0, mu + 0.1 * randn()));
end

function [basis, center] = elite_eigen_basis(population, span, elite_rate)
NP = size(population, 1);
D = size(population, 2);
elite_count = max(4, min(NP, round(elite_rate * NP)));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.0025 * span) .^ 2 + 1e-14);
cov_matrix = (cov_matrix + cov_matrix') / 2;
[basis, values] = eig(cov_matrix, 'vector');
if ~isvector(values)
    eig_values = diag(values);
else
    eig_values = values;
end
[~, order] = sort(real(eig_values), 'descend');
basis = real(basis(:, order));
if any(~isfinite(basis), 'all') || size(basis, 1) ~= D
    basis = eye(D);
    center = mean(population, 1);
end
end

function trial = eigen_crossover(parent, mutant, CR, basis, center)
D = numel(parent);
parent_coord = (parent - center) * basis;
mutant_coord = (mutant - center) * basis;
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial_coord = parent_coord;
trial_coord(mask) = mutant_coord(mask);
trial = center + trial_coord * basis';
end

function trial = binomial_crossover(parent, mutant, CR)
D = numel(parent);
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial = parent;
trial(mask) = mutant(mask);
end

function trial = repair_bounds(trial, parent, lb, ub, mode)
mode = lower(string(mode));
low = trial < lb;
high = trial > ub;
switch mode
    case "random"
        reset = low | high;
        if any(reset)
            trial(reset) = lb(reset) + rand(1, sum(reset)) .* (ub(reset) - lb(reset));
        end
    case "reflect"
        trial(low) = lb(low) + (lb(low) - trial(low));
        trial(high) = ub(high) - (trial(high) - ub(high));
    otherwise
        trial(low) = 0.5 * (parent(low) + lb(low));
        trial(high) = 0.5 * (parent(high) + ub(high));
end
trial = min(max(trial, lb), ub);
end

function idx = rank_weighted_index(limit, pressure)
limit = max(1, limit);
power = max(1.0, pressure);
idx = min(limit, max(1, floor(limit * (rand() ^ power)) + 1));
end

function idx = rank_weighted_index_except(NP, banned, pressure, progress)
idx = rank_weighted_index(NP, pressure + progress);
tries = 0;
while idx == banned && tries < 20
    idx = rank_weighted_index(NP, pressure + progress);
    tries = tries + 1;
end
if idx == banned
    idx = random_index_except(NP, banned);
end
end

function idx = random_pool_index(pool_size, banned)
idx = randi(pool_size);
tries = 0;
while any(idx == banned) && tries < 30
    idx = randi(pool_size);
    tries = tries + 1;
end
end

function ids = random_indices(NP, count, banned)
pool = setdiff(1:NP, banned, 'stable');
if numel(pool) < count
    ids = pool(randperm(numel(pool)));
    while numel(ids) < count
        ids(end + 1) = random_index_except(NP, banned); %#ok<AGROW>
    end
else
    ids = pool(randperm(numel(pool), count));
end
ids = ids(:)';
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function rank = rank_index(fitness, i)
[~, order] = sort(fitness);
positions = zeros(size(fitness));
positions(order) = 1:numel(fitness);
rank = positions(i);
end

function idx = sample_discrete(probs)
idx = find(cumsum(probs) >= rand(), 1, 'first');
if isempty(idx)
    idx = numel(probs);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
