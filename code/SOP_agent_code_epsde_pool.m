function result = SOP_agent_code_epsde_pool(problem, seed, options)
% CoDE/EPSDE-style multi-strategy differential evolution pool.
%
% The population competes among several DE mutation/crossover strategies and
% parameter pools. Successful strategy-parameter pairs receive higher future
% sampling probability. This is a metaheuristic-only selector, not a
% gradient or deterministic numerical solver.
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
NP = get_option(options, 'population_num', max(120, 8 * D));
max_fes = get_option(options, 'max_fes', 10000 * D);
max_iter = get_option(options, 'max_iter', max(1, floor((max_fes - NP) / NP)));
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
profile = lower(string(get_option(options, 'profile', 'balanced')));

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
strategy_success = ones(1, 4);
strategy_trials = 3 * ones(1, 4);
param_success = ones(1, 6);
param_trials = 3 * ones(1, 6);
curve = zeros(max_iter, 1);
actual_iter = 0;

for iter = 1:max_iter
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = iter / max_iter;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    best_position = population(1, :);
    best_raw = fitness(1);
    combined = [population; archive];
    strategy_probs = pool_probabilities(profile, strategy_success, strategy_trials, progress);
    param_probs = adaptive_probabilities(param_success, param_trials);
    trial = population;
    used_strategy = zeros(NP, 1);
    used_param = zeros(NP, 1);
    for i = 1:NP
        strategy = sample_discrete(strategy_probs);
        param_id = sample_discrete(param_probs);
        [F, CR] = parameter_pair(param_id, progress);
        used_strategy(i) = strategy;
        used_param(i) = param_id;
        mutant = make_mutant(strategy, population, combined, fitness, i, F, progress);
        trial(i, :) = repair_bounds(binomial_crossover(population(i, :), mutant, CR), population(i, :), lb, ub);
    end

    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
        used_strategy = used_strategy(1:remaining);
        used_param = used_param(1:remaining);
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
        if size(archive, 1) > round((1.2 + 0.8 * progress) * NP)
            archive = archive(randperm(size(archive, 1), round((1.2 + 0.8 * progress) * NP)), :);
        end
    end
    for i = 1:rows
        strategy_trials(used_strategy(i)) = strategy_trials(used_strategy(i)) + 1;
        param_trials(used_param(i)) = param_trials(used_param(i)) + 1;
        if improved(i)
            strategy_success(used_strategy(i)) = strategy_success(used_strategy(i)) + 1;
            param_success(used_param(i)) = param_success(used_param(i)) + 1;
            population(i, :) = trial(i, :);
            fitness(i) = trial_fitness(i);
        end
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
result.algorithm_combination = sprintf('CoDE/EPSDE multi-strategy Differential Evolution\nStrategy pool: rand/1, current-to-pbest/1, current-to-rand/1, best/2\nAdaptive parameter-pair selection with archive');
result.combination_number = 4;
result.agent_id = 'Agent1';
result.problem = problem;
end

function mutant = make_mutant(strategy, population, combined, fitness, i, F, progress)
NP = size(population, 1);
p_num = max(2, min(NP, round((0.05 + 0.20 * (1 - 0.5 * progress)) * NP)));
switch strategy
    case 1
        ids = random_indices(NP, 3, i);
        mutant = population(ids(1), :) + F .* (population(ids(2), :) - population(ids(3), :));
    case 2
        r1 = random_index_except(NP, i);
        r2 = randi(size(combined, 1));
        pbest = randi(p_num);
        mutant = population(i, :) + F .* (population(pbest, :) - population(i, :)) + F .* (population(r1, :) - combined(r2, :));
    case 3
        ids = random_indices(NP, 2, i);
        pbest = randi(p_num);
        mutant = population(i, :) + rand() .* (population(pbest, :) - population(i, :)) + F .* (population(ids(1), :) - population(ids(2), :));
    otherwise
        ids = random_indices(NP, 4, i);
        mutant = population(1, :) + F .* (population(ids(1), :) - population(ids(2), :)) + 0.5 * F .* (population(ids(3), :) - population(ids(4), :));
        if rank_index(fitness, i) > 0.65 * NP
            mutant = 0.75 * mutant + 0.25 * population(i, :);
        end
end
end

function probs = pool_probabilities(profile, success, trials, progress)
switch profile
    case "exploit"
        base = [0.14, 0.52, 0.12, 0.22];
    case "center_exploit"
        base = [0.10, 0.58, 0.08, 0.24];
    case "diverse"
        base = [0.34, 0.30, 0.24, 0.12];
    otherwise
        base = [0.25, 0.38, 0.20, 0.17];
end
adaptive = adaptive_probabilities(success, trials);
probs = 0.55 * base + 0.45 * adaptive;
probs(1) = probs(1) * (1.15 - 0.50 * progress);
probs(2) = probs(2) * (0.90 + 0.35 * progress);
probs = probs ./ sum(probs);
end

function probs = adaptive_probabilities(success, trials)
rate = success ./ trials;
probs = rate ./ sum(rate);
end

function [F, CR] = parameter_pair(param_id, progress)
F_pool = [1.0, 1.0, 0.8, 0.8, 0.6, 0.45];
CR_pool = [0.1, 0.9, 0.2, 0.8, 0.5, 0.95];
F = F_pool(param_id);
CR = CR_pool(param_id);
if progress > 0.65
    F = 0.82 * F + 0.18 * (0.35 + 0.25 * rand());
    CR = 0.80 * CR + 0.20 * min(1, max(0, 0.75 + 0.15 * randn()));
end
F = min(1, max(0.1, F + 0.05 * randn()));
CR = min(1, max(0, CR + 0.05 * randn()));
end

function idx = sample_discrete(probs)
idx = find(cumsum(probs) >= rand(), 1, 'first');
end

function ids = random_indices(NP, count, banned)
pool = setdiff(1:NP, banned, 'stable');
perm = pool(randperm(numel(pool), count));
ids = perm(:)';
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
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

function rank = rank_index(fitness, i)
[~, order] = sort(fitness);
positions = zeros(size(fitness));
positions(order) = 1:numel(fitness);
rank = positions(i);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
