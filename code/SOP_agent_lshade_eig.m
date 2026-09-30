function result = SOP_agent_lshade_eig(problem, seed, options)
% L-SHADE with eigen-coordinate crossover and elite covariance sampling.
%
% Literature basis: Differential Evolution/L-SHADE plus covariance/eigen
% coordinate adaptation. The eigen phase performs crossover in an elite
% covariance coordinate system, which is useful for rotated nonseparable
% functions while still using only objective-function feedback.
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
NP_init = get_option(options, 'population_num', max(80, 5 * D));
NP_min = get_option(options, 'min_population_num', 4);
max_fes = get_option(options, 'max_fes', 10000 * D);
H = get_option(options, 'memory_size', 8);
p_rate = get_option(options, 'p_rate', 0.10);
eig_rate = get_option(options, 'eig_rate', 0.55);
eig_interval = get_option(options, 'eig_interval', 12);
elite_rate = get_option(options, 'elite_rate', 0.28);
cma_rate = get_option(options, 'cma_rate', 0.12);
cma_interval = get_option(options, 'cma_interval', 24);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP_init, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
initial_point = get_option(options, 'initial_point', []);
if ~isempty(initial_point)
    center = min(max(initial_point(:)', lb), ub);
    initial_radius = get_option(options, 'initial_radius', []);
    if ~isempty(initial_radius)
        radius = make_radius(initial_radius, span, D);
        center_matrix = repmat(center, NP_init, 1);
        radius_matrix = repmat(radius, NP_init, 1);
        population = center_matrix + randn(NP_init, D) .* radius_matrix;
        if get_option(options, 'initial_cauchy', false)
            cauchy_noise = tan(pi * (rand(NP_init, D) - 0.5));
            cauchy_noise = min(max(cauchy_noise, -8), 8);
            cauchy_mask = rand(NP_init, D) < 0.30;
            population(cauchy_mask) = center_matrix(cauchy_mask) + cauchy_noise(cauchy_mask) .* radius_matrix(cauchy_mask);
        end
        population = min(max(population, lb), ub);
    end
    population(1, :) = center;
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
mu_F = 0.5 * ones(1, H);
mu_CR = 0.5 * ones(1, H);
memory_index = 1;
curve = zeros(max(1, ceil(max_fes / max(1, NP_min))), 1);
iter = 0;
eig_basis = eye(D);
eig_center = mean(population, 1);

while eval_count < max_fes && size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if iter == 1 || mod(iter, eig_interval) == 0
        [eig_basis, eig_center] = elite_eigen_basis(population, span, elite_rate);
    end
    combined = [population; archive];
    pool_size = size(combined, 1);
    p_num = max(2, round(p_rate * NP));
    trial_population = population;
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);

    for i = 1:NP
        mem = randi(H);
        F = sample_F(mu_F(mem));
        CR = min(1, max(0, mu_CR(mem) + 0.1 * randn()));
        sampled_F(i) = F;
        sampled_CR(i) = CR;
        pbest = randi(p_num);
        r1 = random_index_except(NP, i);
        r2 = random_pool_index(pool_size, [i r1]);
        mutant = population(i, :) ...
            + F .* (population(pbest, :) - population(i, :)) ...
            + F .* (population(r1, :) - combined(r2, :));
        if rand() < eig_rate
            trial = eigen_crossover(population(i, :), mutant, CR, eig_basis, eig_center);
        else
            trial = binomial_crossover(population(i, :), mutant, CR);
        end
        trial_population(i, :) = repair_bounds(trial, population(i, :), lb, ub);
    end

    remaining = max_fes - eval_count;
    if remaining <= 0
        break;
    end
    if remaining < NP
        trial_population = trial_population(1:remaining, :);
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
        if trial_fitness(i) <= fitness(i)
            archive = [archive; population(i, :)]; %#ok<AGROW>
            success_delta(end + 1, 1) = max(0, fitness(i) - trial_fitness(i)); %#ok<AGROW>
            success_F(end + 1, 1) = sampled_F(i); %#ok<AGROW>
            success_CR(end + 1, 1) = sampled_CR(i); %#ok<AGROW>
            population(i, :) = trial_population(i, :);
            fitness(i) = trial_fitness(i);
        end
    end
    if size(archive, 1) > NP
        archive = archive(randperm(size(archive, 1), NP), :);
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

    if eval_count < max_fes && mod(iter, cma_interval) == 0
        [population, fitness, extra_eval] = covariance_elite_step(population, fitness, problem, lb, ub, span, ...
            eval_count, max_fes, cma_rate, elite_rate);
        eval_count = eval_count + extra_eval;
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    curve(iter) = best_raw;

    target_NP = round(NP_init - (NP_init - NP_min) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness);
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
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
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nEigen-coordinate crossover\nElite covariance evolutionary sampling');
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;

if verbose
    fprintf('L-SHADE-EIG finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
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
cov_matrix = centered' * (centered .* weights') + diag((0.002 * span) .^ 2 + 1e-14);
cov_matrix = (cov_matrix + cov_matrix') / 2;
[basis, flag] = eig(cov_matrix, 'vector');
if ~isvector(flag)
    eig_values = diag(flag);
else
    eig_values = flag;
end
[~, order] = sort(eig_values, 'descend');
basis = basis(:, order);
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

function [population, fitness, eval_count] = covariance_elite_step(population, fitness, problem, lb, ub, span, eval_count_so_far, max_fes, cma_rate, elite_rate)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
remaining = max_fes - eval_count_so_far;
sample_count = min(remaining, max(2, round(cma_rate * NP)));
if sample_count <= 0
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = max(4, min(NP, round(elite_rate * NP)));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.0025 * span) .^ 2 + 1e-14);
cov_matrix = (cov_matrix + cov_matrix') / 2;
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), eps)));
end
candidates = center + randn(sample_count, D) * R;
if rand() < 0.45
    candidates = 0.70 * candidates + 0.30 * repmat(population(1, :), sample_count, 1);
end
candidates = min(max(candidates, lb), ub);
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
[~, worst_order] = sort(fitness, 'descend');
replace_idx = worst_order(1:sample_count);
for i = 1:sample_count
    if values(i) < fitness(replace_idx(i))
        population(replace_idx(i), :) = candidates(i, :);
        fitness(replace_idx(i)) = values(i);
    end
end
end

function F = sample_F(mu)
F = mu + 0.1 * tan(pi * (rand() - 0.5));
tries = 0;
while F <= 0 && tries < 20
    F = mu + 0.1 * tan(pi * (rand() - 0.5));
    tries = tries + 1;
end
if F <= 0
    F = 0.5;
end
F = min(F, 1);
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function idx = random_pool_index(pool_size, banned)
idx = randi(pool_size);
tries = 0;
while any(idx == banned) && tries < 20
    idx = randi(pool_size);
    tries = tries + 1;
end
end

function trial = binomial_crossover(parent, mutant, CR)
D = numel(parent);
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial = parent;
trial(mask) = mutant(mask);
end

function trial = repair_bounds(trial, parent, lb, ub)
low = trial < lb;
high = trial > ub;
trial(low) = 0.5 * (parent(low) + lb(low));
trial(high) = 0.5 * (parent(high) + ub(high));
trial = min(max(trial, lb), ub);
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
else
    radius = radius(:)';
end
if numel(radius) ~= D
    radius = repmat(radius(1), 1, D);
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
