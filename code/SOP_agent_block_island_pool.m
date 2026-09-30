function result = SOP_agent_block_island_pool(problem, seed, options)
% Cooperative block island operator pool for CEC hybrid/composition cases.
%
% Agent2 direction: several islands emphasize GSK, RIME, TLBO, and MPA/WSO
% style moves. A small block-DE component is only used as a subspace
% perturbation inside the pool. The method is derivative-free and calls only
% SOP_cec_evaluate.
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
NP = get_option(options, 'population_num', 220);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
island_count = get_option(options, 'island_count', 4);
block_rate_start = get_option(options, 'block_rate_start', 0.18);
block_rate_end = get_option(options, 'block_rate_end', 0.045);
migration_interval = get_option(options, 'migration_interval', 18);
max_iter = get_option(options, 'max_iter', max(1, floor((max_fes - NP) / NP)));

population = lb + rand(NP, D) .* span;
initial_population = get_option(options, 'initial_population', []);
initial_rows = 0;
if ~isempty(initial_population)
    initial_rows = min(NP, size(initial_population, 1));
    population(1:initial_rows, :) = min(max(initial_population(1:initial_rows, :), lb), ub);
end
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
initial_point = get_option(options, 'initial_point', []);
if ~isempty(initial_point)
    center = min(max(initial_point(:)', lb), ub);
    initial_radius = get_option(options, 'initial_radius', 0.006);
    radius = make_radius(initial_radius, span, D);
    population = repmat(center, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
    if get_option(options, 'initial_cauchy', false)
        cauchy_noise = tan(pi * (rand(NP, D) - 0.5));
        cauchy_noise = min(max(cauchy_noise, -8), 8);
        cauchy_rows = rand(NP, 1) < 0.35;
        population(cauchy_rows, :) = repmat(center, sum(cauchy_rows), 1) + ...
            cauchy_noise(cauchy_rows, :) .* repmat(radius, sum(cauchy_rows), 1);
    end
    population = min(max(population, lb), ub);
    if get_option(options, 'preserve_initial_population_after_radius', true) && initial_rows > 0
        population(1:initial_rows, :) = min(max(initial_population(1:initial_rows, :), lb), ub);
    end
    population(1, :) = center;
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
island_id = mod((1:NP)' - 1, island_count) + 1;
island_id = island_id(randperm(NP));
velocity = 0.006 .* (2 * rand(NP, D) - 1) .* span;
operator_success = ones(island_count, 5);
operator_trials = 2 * ones(island_count, 5);
curve = zeros(max_iter, 1);
actual_iter = 0;

for iter = 1:max_iter
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = min(1, eval_count / max_fes);
    [fitness, order] = sort(fitness);
    population = population(order, :);
    island_id = island_id(order);
    velocity = velocity(order, :);
    [best_raw, best_idx] = min(fitness);
    best_position = population(best_idx, :);
    combined = [population; archive];

    block_rate = block_rate_start + (block_rate_end - block_rate_start) * progress;
    trial = population;
    op_used = zeros(NP, 1);
    for i = 1:NP
        island = island_id(i);
        probs = island_probabilities(island, operator_success(island, :), operator_trials(island, :), progress, options);
        op = choose_operator(probs);
        op_used(i) = op;
        mask = make_block_mask(D, block_rate, get_option(options, 'wide_block_probability', 0.12));
        switch op
            case 1
                candidate = gsk_block(population, fitness, i, mask, lb, ub, progress);
            case 2
                candidate = rime_block(population(i, :), fitness(i), fitness, best_position, mask, lb, ub, span, progress);
            case 3
                candidate = tlbo_block(population, fitness, i, mask, lb, ub);
            case 4
                [candidate, velocity(i, :)] = mpa_wso_block(population, fitness, velocity(i, :), i, best_position, mask, lb, ub, span, progress);
            otherwise
                candidate = de_block(population, combined, i, mask, lb, ub, progress);
        end
        trial(i, :) = candidate;
    end

    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
        op_used = op_used(1:remaining);
    end
    trial_fitness = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end

    rows = numel(trial_fitness);
    improved = trial_fitness <= fitness(1:rows);
    if any(improved)
        archive = [archive; population(improved, :)]; %#ok<AGROW>
        if size(archive, 1) > 2 * NP
            archive = archive(randperm(size(archive, 1), 2 * NP), :);
        end
    end
    for i = 1:rows
        island = island_id(i);
        op = op_used(i);
        operator_trials(island, op) = operator_trials(island, op) + 1;
        if improved(i)
            operator_success(island, op) = operator_success(island, op) + 1;
            population(i, :) = trial(i, :);
            fitness(i) = trial_fitness(i);
        end
    end

    if mod(iter, migration_interval) == 0 && eval_count < max_fes && toc(t_start) < max_runtime_sec
        [population, fitness, eval_delta] = migrate_islands(problem, population, fitness, island_id, best_position, lb, ub, span, max_fes - eval_count, progress, options);
        eval_count = eval_count + eval_delta;
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    curve(iter) = best_raw;
end
curve = curve(1:actual_iter);

runtime = toc(t_start);
[final_fitness, final_order] = sort(fitness);
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.final_population = final_population;
result.final_fitness = final_fitness;
result.algorithm_combination = sprintf('Cooperative island model\nGSK/RIME/TLBO/MPA-WSO operator pool\nRandom block subspace perturbation and migration');
result.combination_number = 5;
result.agent_id = 'Agent2';
result.problem = problem;
end

function probs = island_probabilities(island, success, trials, progress, options)
if get_option(options, 'gsk_only', false)
    probs = [1, 0, 0, 0, 0];
    return;
end
de_weight = get_option(options, 'block_de_weight', 0.10);
switch island
    case 1
        base = [0.42, 0.20, 0.16, 0.16, de_weight];
    case 2
        base = [0.20, 0.42, 0.14, 0.18, de_weight];
    case 3
        base = [0.16, 0.16, 0.42, 0.18, de_weight];
    otherwise
        base = [0.18, 0.16, 0.16, 0.42, de_weight];
end
base = base ./ sum(base);
adaptive = success ./ max(eps, trials);
adaptive = adaptive ./ sum(adaptive);
probs = (0.72 - 0.20 * progress) .* base + (0.28 + 0.20 * progress) .* adaptive;
probs(2) = probs(2) * (1.12 - 0.25 * progress);
probs(3) = probs(3) * (0.85 + 0.35 * progress);
probs(5) = probs(5) * (0.75 + 0.45 * progress);
probs = probs ./ sum(probs);
end

function op = choose_operator(probs)
cum_probs = cumsum(probs);
op = find(cum_probs >= rand(), 1, 'first');
end

function mask = make_block_mask(D, block_rate, wide_probability)
rate = block_rate * (0.75 + 0.50 * rand());
if rand() < wide_probability
    rate = min(0.38, rate * (1.8 + 0.8 * rand()));
end
count = max(1, min(D, round(rate * D)));
mask = false(1, D);
mask(randperm(D, count)) = true;
end

function candidate = gsk_block(population, fitness, i, mask, lb, ub, progress)
NP = size(population, 1);
candidate = population(i, :);
top_count = max(2, round(0.18 * NP));
mid_start = max(top_count + 1, round(0.45 * NP));
bottom_start = max(mid_start + 1, round(0.75 * NP));
KF = min(0.92, max(0.10, 0.60 * (1 - 0.25 * progress) + 0.08 * randn()));
if rand() < 0.55 * (1 - progress) + 0.18
    better = randi(top_count);
    worse = bottom_start - 1 + randi(NP - bottom_start + 1);
    r1 = random_index_except(NP, i);
    r2 = random_index_except(NP, r1);
    step = KF .* (population(better, :) - population(worse, :)) + 0.35 * KF .* (population(r1, :) - population(r2, :));
else
    senior_top = randi(top_count);
    senior_mid = mid_start - 1 + randi(max(1, bottom_start - mid_start));
    senior_bottom = bottom_start - 1 + randi(NP - bottom_start + 1);
    step = KF .* (population(1, :) - population(i, :)) + KF .* (population(senior_top, :) - population(senior_bottom, :)) + 0.15 * KF .* (population(senior_mid, :) - population(i, :));
end
if rank_index(fitness, i) > 0.65 * NP && rand() < 0.12
    step = step + 0.006 * randn(size(step)) .* (ub - lb);
end
candidate(mask) = candidate(mask) + step(mask);
candidate = repair(candidate, population(i, :), lb, ub);
end

function candidate = rime_block(x, fit, fitness, best, mask, lb, ub, span, progress)
candidate = x;
norm_fit = (max(fitness) - fit + eps) ./ (max(fitness) - min(fitness) + eps);
block = find(mask);
soft = (0.020 * (1 - progress) + 0.0015) .* span(block);
candidate(block) = best(block) + randn(1, numel(block)) .* soft;
hard_rate = min(0.72, 0.20 + norm_fit + 0.18 * progress);
hard_dims = block(rand(1, numel(block)) < hard_rate);
candidate(hard_dims) = best(hard_dims);
candidate = repair(candidate, x, lb, ub);
end

function candidate = tlbo_block(population, fitness, i, mask, lb, ub)
NP = size(population, 1);
candidate = population(i, :);
teacher = population(1, :);
mean_pop = mean(population, 1);
if rand() < 0.55
    teaching_factor = randi(2);
    step = rand(size(candidate)) .* (teacher - teaching_factor .* mean_pop);
else
    partner = random_index_except(NP, i);
    if fitness(i) < fitness(partner)
        step = rand(size(candidate)) .* (population(i, :) - population(partner, :));
    else
        step = rand(size(candidate)) .* (population(partner, :) - population(i, :));
    end
end
candidate(mask) = candidate(mask) + step(mask);
candidate = repair(candidate, population(i, :), lb, ub);
end

function [candidate, velocity] = mpa_wso_block(population, fitness, velocity, i, best, mask, lb, ub, span, progress)
NP = size(population, 1);
candidate = population(i, :);
block = find(mask);
if rand() < 0.50
    if progress < 0.33
        step = randn(1, numel(block)) .* (population(i, block) - best(block));
    elseif progress < 0.70
        step = levy_step(1, numel(block)) .* (best(block) - population(i, block));
    else
        step = randn(1, numel(block)) .* (0.003 .* span(block));
    end
    candidate(block) = population(i, block) + 0.48 .* step;
else
    r1 = random_index_except(NP, i);
    r2 = random_index_except(NP, r1);
    pressure = exp(-rank_index(fitness, i) / max(1, NP));
    velocity(block) = (0.70 - 0.40 * progress) .* velocity(block) + ...
        pressure .* rand(1, numel(block)) .* (best(block) - population(i, block)) + ...
        0.22 .* rand(1, numel(block)) .* (population(r1, block) - population(r2, block));
    candidate(block) = population(i, block) + velocity(block);
end
candidate = repair(candidate, population(i, :), lb, ub);
end

function candidate = de_block(population, combined, i, mask, lb, ub, progress)
NP = size(population, 1);
candidate = population(i, :);
p_num = max(2, round((0.10 + 0.10 * progress) * NP));
F = min(0.82, max(0.18, 0.45 + 0.16 * tan(pi * (rand() - 0.5))));
pbest = randi(p_num);
r1 = random_index_except(NP, i);
r2 = randi(size(combined, 1));
step = F .* (population(pbest, :) - population(i, :)) + F .* (population(r1, :) - combined(r2, :));
candidate(mask) = candidate(mask) + step(mask);
candidate = repair(candidate, population(i, :), lb, ub);
end

function [population, fitness, eval_count] = migrate_islands(problem, population, fitness, island_id, best, lb, ub, span, remaining_fes, progress, options)
islands = unique(island_id(:))';
migration_count = min(numel(islands), remaining_fes);
eval_count = 0;
if migration_count <= 0
    return;
end
candidates = zeros(migration_count, size(population, 2));
target_rows = zeros(migration_count, 1);
for k = 1:migration_count
    rows = find(island_id == islands(k));
    [~, local_order] = sort(fitness(rows));
    elite = population(rows(local_order(1)), :);
    target_rows(k) = rows(local_order(end));
    sigma = (0.018 * (1 - progress) + get_option(options, 'migration_sigma', 0.0018)) .* span;
    candidate = 0.62 .* best + 0.38 .* elite + randn(size(best)) .* sigma;
    if rand() < 0.35
        mask = make_block_mask(numel(best), get_option(options, 'migration_block_rate', 0.12), 0.20);
        candidate(~mask) = population(target_rows(k), ~mask);
    end
    candidates(k, :) = min(max(candidate, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
for k = 1:numel(values)
    row = target_rows(k);
    if values(k) <= fitness(row)
        population(row, :) = candidates(k, :);
        fitness(row) = values(k);
    end
end
end

function step = levy_step(rows, cols)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
    (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
u = randn(rows, cols) .* sigma;
v = randn(rows, cols);
step = u ./ (abs(v) .^ (1 / beta) + eps);
step = min(max(step, -8), 8);
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function r = rank_index(fitness, i)
[~, order] = sort(fitness);
ranks = zeros(numel(fitness), 1);
ranks(order) = 1:numel(fitness);
r = ranks(i);
end

function candidate = repair(candidate, parent, lb, ub)
low = candidate < lb;
high = candidate > ub;
candidate(low) = 0.5 * (parent(low) + lb(low));
candidate(high) = 0.5 * (parent(high) + ub(high));
candidate = min(max(candidate, lb), ub);
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
else
    radius = reshape(radius, 1, []);
    if numel(radius) ~= D
        radius = mean(radius(:)) .* ones(1, D);
    end
end
radius = max(radius, 1e-12 .* max(1, span));
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
