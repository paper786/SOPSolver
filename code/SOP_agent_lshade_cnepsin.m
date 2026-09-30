function result = SOP_agent_lshade_cnepsin(problem, seed, options)
% L-SHADE with ensemble sinusoidal scaling and neighborhood eigen crossover.
%
% The implementation combines three literature components in one original
% solver: success-history adaptive DE, linear population reduction, and a
% Euclidean-neighborhood covariance coordinate system for crossover.
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
NP_init = get_option(options, 'population_num', 18 * D);
NP_min = get_option(options, 'min_population_num', 4);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
H = get_option(options, 'memory_size', 5);
p_rate = get_option(options, 'p_rate', 0.11);
archive_rate = get_option(options, 'archive_rate', 1.4);
eigen_rate = get_option(options, 'eigen_rate', 0.4);
neighbor_rate = get_option(options, 'neighbor_rate', 0.5);
learning_period = get_option(options, 'learning_period', 20);
initial_frequency = get_option(options, 'initial_frequency', 0.5);
generation_scale = generation_scale_for_dimension(D);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP_init, D) .* span;
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);

memory_F = 0.5 * ones(H, 1);
memory_CR = 0.5 * ones(H, 1);
memory_frequency = initial_frequency * ones(H, 1);
memory_index = 1;
archive = zeros(0, D);
curve = zeros(max(1, ceil(max_fes / NP_min)), 1);
strategy_success = ones(2, max(learning_period, generation_scale + learning_period));
strategy_failure = ones(2, max(learning_period, generation_scale + learning_period));
iter = 0;

while eval_count < max_fes && size(population, 1) >= NP_min && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    [fitness, order] = sort(fitness);
    population = population(order, :);

    memory_rows = randi(H, NP, 1);
    sampled_CR = memory_CR(memory_rows) + 0.1 * randn(NP, 1);
    sampled_CR(memory_CR(memory_rows) < 0) = 0;
    sampled_CR = min(1, max(0, sampled_CR));
    sampled_F = positive_cauchy(memory_F(memory_rows), 0.1);
    sampled_frequency = positive_cauchy(memory_frequency(memory_rows), 0.1);

    active_strategy = 0;
    if eval_count <= max_fes / 2
        if iter <= learning_period
            active_strategy = 1 + (rand() >= 0.5);
        else
            first = max(1, iter - learning_period);
            last = iter - 1;
            success_rate_1 = sum(strategy_success(1, first:last)) / ...
                max(1, sum(strategy_success(1, first:last)) + sum(strategy_failure(1, first:last)));
            success_rate_2 = sum(strategy_success(2, first:last)) / ...
                max(1, sum(strategy_success(2, first:last)) + sum(strategy_failure(2, first:last)));
            active_strategy = 1 + (success_rate_2 >= success_rate_1);
        end
        if active_strategy == 1
            decay = max(0, (generation_scale - iter) / generation_scale);
            fixed_F = 0.5 * (sin(2 * pi * initial_frequency * iter + pi) * decay + 1);
            sampled_F(:) = min(1, max(eps, fixed_F));
        else
            growth = min(1, iter / generation_scale);
            sampled_F = 0.5 * (sin(2 * pi .* sampled_frequency .* iter) .* growth + 1);
            sampled_F = min(1, max(eps, sampled_F));
        end
    end

    p_count = max(2, min(NP, round(p_rate * NP)));
    pbest_rows = orderless_pbest_rows(NP, p_count);
    combined = [population; archive];
    [r1, r2] = distinct_donor_rows(NP, size(combined, 1));
    mutant = population + sampled_F .* ...
        (population(pbest_rows, :) - population + population(r1, :) - combined(r2, :));
    mutant = repair_bounds(mutant, population, lb, ub);

    crossover_mask = rand(NP, D) < sampled_CR;
    forced_columns = randi(D, NP, 1);
    crossover_mask(sub2ind([NP, D], (1:NP)', forced_columns)) = true;
    if rand() < eigen_rate
        basis = neighborhood_basis(population, best_position, neighbor_rate, span);
        parent_coordinates = population * basis;
        mutant_coordinates = mutant * basis;
        trial_coordinates = parent_coordinates;
        trial_coordinates(crossover_mask) = mutant_coordinates(crossover_mask);
        trial_population = trial_coordinates * basis';
        trial_population = repair_bounds(trial_population, population, lb, ub);
    else
        trial_population = population;
        trial_population(crossover_mask) = mutant(crossover_mask);
    end

    remaining = max_fes - eval_count;
    rows = min(NP, remaining);
    trial_population = trial_population(1:rows, :);
    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);

    parent_fitness = fitness(1:rows);
    successful = trial_fitness < parent_fitness;
    improvements = parent_fitness(successful) - trial_fitness(successful);
    if active_strategy > 0
        ensure_column = iter > size(strategy_success, 2);
        if ensure_column
            strategy_success(:, iter) = 1;
            strategy_failure(:, iter) = 1;
        end
        other_strategy = 3 - active_strategy;
        strategy_success(other_strategy, iter) = 1;
        strategy_failure(other_strategy, iter) = 1;
        strategy_success(active_strategy, iter) = nnz(successful);
        strategy_failure(active_strategy, iter) = rows - nnz(successful);
    end

    if any(successful)
        successful_rows = find(successful);
        archive = [archive; population(successful_rows, :)]; %#ok<AGROW>
        weights = improvements ./ max(eps, sum(improvements));
        good_F = sampled_F(successful_rows);
        good_CR = sampled_CR(successful_rows);
        good_frequency = sampled_frequency(successful_rows);
        memory_F(memory_index) = weighted_lehmer(good_F, weights, memory_F(memory_index));
        if all(good_CR == 0) || memory_CR(memory_index) < 0
            memory_CR(memory_index) = -1;
        else
            memory_CR(memory_index) = weighted_lehmer(good_CR, weights, memory_CR(memory_index));
        end
        memory_frequency(memory_index) = weighted_lehmer( ...
            good_frequency, weights, memory_frequency(memory_index));
        memory_index = memory_index + 1;
        if memory_index > H
            memory_index = 1;
        end
        population(successful_rows, :) = trial_population(successful_rows, :);
        fitness(successful_rows) = trial_fitness(successful_rows);
    end

    [generation_best, generation_best_idx] = min(trial_fitness);
    if generation_best < best_raw
        best_raw = generation_best;
        best_position = trial_population(generation_best_idx, :);
    end

    target_NP = round(NP_init + (NP_min - NP_init) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
    if target_NP < size(population, 1)
        [fitness, keep_order] = sort(fitness);
        population = population(keep_order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
    archive_limit = max(1, round(archive_rate * size(population, 1)));
    if size(archive, 1) > archive_limit
        archive = archive(randperm(size(archive, 1), archive_limit), :);
    end
    curve(iter) = best_raw;
end

runtime = toc(t_start);
curve = curve(1:iter);
[final_fitness, final_order] = sort(fitness);
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
result.algorithm_combination = sprintf(['Differential Evolution (DE)\n' ...
    'L-SHADE success-history adaptation and linear population reduction\n' ...
    'Ensemble sinusoidal scaling-factor schedule\n' ...
    'Euclidean-neighborhood covariance coordinate crossover']);
result.combination_number = 4;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = population(final_order, :);
result.final_fitness = final_fitness;

if verbose
    fprintf('L-SHADE-cnEpSin finished %s %dD F%d: best %.12g, time %.3f, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function rows = orderless_pbest_rows(NP, p_count)
rows = randi(p_count, NP, 1);
end

function [r1, r2] = distinct_donor_rows(NP, combined_count)
targets = (1:NP)';
r1 = randi(NP, NP, 1);
bad = r1 == targets;
while any(bad)
    r1(bad) = randi(NP, nnz(bad), 1);
    bad = r1 == targets;
end
r2 = randi(combined_count, NP, 1);
bad = r2 == targets | r2 == r1;
while any(bad)
    r2(bad) = randi(combined_count, nnz(bad), 1);
    bad = r2 == targets | r2 == r1;
end
end

function values = positive_cauchy(centers, scale)
values = centers + scale .* tan(pi * (rand(size(centers)) - 0.5));
bad = values <= 0;
tries = 0;
while any(bad) && tries < 100
    values(bad) = centers(bad) + scale .* tan(pi * (rand(nnz(bad), 1) - 0.5));
    bad = values <= 0;
    tries = tries + 1;
end
values(bad) = 0.5;
values = min(values, 1);
end

function basis = neighborhood_basis(population, best, neighbor_rate, span)
[NP, D] = size(population);
count = max(2, min(NP, round(neighbor_rate * NP)));
distances = sum((population - best) .^ 2, 2);
[~, order] = sort(distances);
neighbors = population(order(1:count), :);
centered = neighbors - mean(neighbors, 1);
covariance = (centered' * centered) ./ max(1, count - 1);
ridge = max(eps, 1e-14 * mean(span .^ 2));
covariance = (covariance + covariance') ./ 2 + ridge * eye(D);
[basis, eigenvalues] = eig(covariance, 'vector');
if any(~isfinite(basis), 'all') || any(~isfinite(eigenvalues))
    basis = eye(D);
end
end

function repaired = repair_bounds(candidate, parent, lb, ub)
repaired = candidate;
low = repaired < lb;
high = repaired > ub;
lower_mid = 0.5 * (parent + lb);
upper_mid = 0.5 * (parent + ub);
repaired(low) = lower_mid(low);
repaired(high) = upper_mid(high);
repaired = min(max(repaired, lb), ub);
end

function value = weighted_lehmer(samples, weights, fallback)
denominator = sum(weights .* samples);
if denominator <= eps
    value = fallback;
else
    value = sum(weights .* (samples .^ 2)) / denominator;
end
end

function scale = generation_scale_for_dimension(D)
switch D
    case 10
        scale = 2163;
    case 30
        scale = 2745;
    case 50
        scale = 3022;
    case 100
        scale = 3401;
    otherwise
        scale = max(1000, round(34 * D));
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
