function result = SOP_agent1_adaptive_de(problem, seed, options)
% Agent1: self-adaptive differential evolution for continuous CEC SOPs.
%
% Literature basis: Differential Evolution (DE), self-adaptive DE/SaDE,
% and JADE-style current-to-pbest guidance. The implementation is original
% and only uses the public benchmark wrapper.
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

NP = get_option(options, 'population_num', default_population(D, problem.func_num));
max_iter = get_option(options, 'max_iter', default_iterations(D, NP));
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

H = 6;
mu_F = 0.55 * ones(1, H);
mu_CR = 0.65 * ones(1, H);
memory_index = 1;
p_rate = get_option(options, 'p_rate', profile_p_rate(problem.func_num));
stagnation_limit = max(25, round(0.08 * max_iter));

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
convergence_raw = zeros(max_iter, 1);
archive = zeros(0, D);
stagnation = 0;
actual_iter = 0;

for iter = 1:max_iter
    if toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    p_num = max(2, round(p_rate * NP));
    success_F = zeros(NP, 1);
    success_CR = zeros(NP, 1);
    success_delta = zeros(NP, 1);
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);
    trial_population = population;
    combined_pool = [population; archive];
    pool_size = size(combined_pool, 1);

    for i = 1:NP
        mem = randi(H);
        F = sample_scale_factor(mu_F(mem));
        CR = min(1, max(0, mu_CR(mem) + 0.1 * randn()));
        sampled_F(i) = F;
        sampled_CR(i) = CR;

        pbest_idx = randi(p_num);
        r1 = random_index_except(NP, i);
        r2 = random_pool_index(pool_size, i);

        mutant = population(i, :) ...
            + F .* (population(pbest_idx, :) - population(i, :)) ...
            + F .* (population(r1, :) - combined_pool(r2, :));

        trial = binomial_crossover(population(i, :), mutant, CR);
        trial = repair_bounds(trial, population(i, :), lb, ub);
        trial_population(i, :) = trial;
    end

    candidate_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(candidate_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = candidate_fitness <= fitness;

    if any(improved)
        parents = population(improved, :);
        archive = [archive; parents]; %#ok<AGROW>
        if size(archive, 1) > NP
            archive = archive(randperm(size(archive, 1), NP), :);
        end
    end

    success_count = 0;
    for i = 1:NP
        if improved(i)
            old_fit = fitness(i);
            population(i, :) = trial_population(i, :);
            fitness(i) = candidate_fitness(i);
            success_count = success_count + 1;
            success_delta(success_count) = max(0, old_fit - candidate_fitness(i));
            success_F(success_count) = sampled_F(i);
            success_CR(success_count) = sampled_CR(i);
        end
    end

    if success_count > 0 && sum(success_delta(1:success_count)) > 0
        weights = success_delta(1:success_count) ./ sum(success_delta(1:success_count));
        sf = success_F(1:success_count);
        scr = success_CR(1:success_count);
        mu_F(memory_index) = sum(weights .* (sf .^ 2)) / max(eps, sum(weights .* sf));
        mu_CR(memory_index) = sum(weights .* scr);
        memory_index = memory_index + 1;
        if memory_index > H
            memory_index = 1;
        end
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
        stagnation = 0;
    else
        stagnation = stagnation + 1;
    end

    if stagnation >= stagnation_limit
        [population, fitness, eval_extra] = restart_worst(population, fitness, problem, lb, ub, span);
        eval_count = eval_count + eval_extra;
        [current_best, current_idx] = min(fitness);
        if current_best < best_raw
            best_raw = current_best;
            best_position = population(current_idx, :);
        end
        stagnation = 0;
    end

    convergence_raw(iter) = best_raw;
end

runtime = toc(t_start);
convergence_raw = convergence_raw(1:actual_iter);
record_curve = SOP_cec_record_value(convergence_raw, problem);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = record_curve;
result.raw_convergence_curve = convergence_raw;
result.runtime = runtime;
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Self-Adaptive Differential Evolution (SaDE)\nDifferential Evolution current-to-pbest mutation');
result.combination_number = 1;
result.agent_id = 'Agent1';
result.problem = problem;

if verbose
    fprintf('Agent1 finished %s %dD F%d: best %.12g, report %.12g, runtime %.4f s.\n', ...
        problem.suite, D, problem.func_num, result.best_value, result.record_value, runtime);
end
end

function NP = default_population(D, func_num)
if D <= 10
    NP = 60;
elseif D <= 30
    NP = 70;
elseif D <= 50
    NP = 90;
else
    NP = 110;
end
if func_num >= 11
    NP = NP + 10;
end
end

function max_iter = default_iterations(D, NP)
target_fe = min(10000 * D, 120000);
max_iter = max(60, floor((target_fe - NP) / NP));
end

function p_rate = profile_p_rate(func_num)
if func_num <= 3
    p_rate = 0.10;
elseif func_num <= 10
    p_rate = 0.15;
else
    p_rate = 0.22;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end

function F = sample_scale_factor(mu)
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
if pool_size <= 1
    idx = 1;
    return;
end
idx = randi(pool_size);
tries = 0;
while idx == banned && tries < 10
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

function [population, fitness, eval_count] = restart_worst(population, fitness, problem, lb, ub, span)
NP = size(population, 1);
D = size(population, 2);
count = max(1, ceil(0.15 * NP));
[~, order] = sort(fitness, 'descend');
replace_idx = order(1:count);
opposite = lb + ub - population(replace_idx, :);
random_points = lb + rand(count, D) .* span;
new_points = 0.55 * opposite + 0.45 * random_points;
new_points = min(max(new_points, lb), ub);
new_fitness = SOP_cec_evaluate(new_points, problem);
eval_count = numel(new_fitness);
accept = new_fitness < fitness(replace_idx);
population(replace_idx(accept), :) = new_points(accept, :);
fitness(replace_idx(accept)) = new_fitness(accept);
end
