function result = SOP_agent_de_pso_hybrid(problem, seed, options)
% Coupled DE-PSO metaheuristic for curved continuous landscapes.
%
% Literature basis: Differential Evolution (DE) and Particle Swarm
% Optimization (PSO). Each generation evaluates a PSO-guided candidate and
% a DE/current-to-pbest candidate for every individual, retaining the best
% trial by objective value.
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
NP = get_option(options, 'population_num', 120);
max_iter = get_option(options, 'max_iter', 2500);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
velocity = 0.1 * (2 * rand(NP, D) - 1) .* span;
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
pbest = population;
pbest_fit = fitness;
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
convergence_raw = zeros(max_iter, 1);
actual_iter = 0;

H = 6;
mu_F = 0.55 * ones(1, H);
mu_CR = 0.65 * ones(1, H);
memory_index = 1;

for iter = 1:max_iter
    if toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = iter / max_iter;
    inertia = 0.82 - 0.48 * progress;
    c1 = 1.60 - 0.40 * progress;
    c2 = 1.20 + 0.60 * progress;

    [fitness, order] = sort(fitness);
    population = population(order, :);
    velocity = velocity(order, :);
    combined_pool = [population; archive];
    pool_size = size(combined_pool, 1);
    p_num = max(2, round((0.10 + 0.10 * progress) * NP));

    pso_candidates = zeros(NP, D);
    de_candidates = zeros(NP, D);
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);

    for i = 1:NP
        velocity(i, :) = inertia * velocity(i, :) ...
            + c1 * rand(1, D) .* (pbest(i, :) - population(i, :)) ...
            + c2 * rand(1, D) .* (best_position - population(i, :));
        velocity(i, :) = min(max(velocity(i, :), -0.25 * span), 0.25 * span);
        pso_candidates(i, :) = min(max(population(i, :) + velocity(i, :), lb), ub);

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
        de_candidates(i, :) = repair_bounds(binomial_crossover(population(i, :), mutant, CR), population(i, :), lb, ub);
    end

    pso_fit = SOP_cec_evaluate(pso_candidates, problem);
    de_fit = SOP_cec_evaluate(de_candidates, problem);
    eval_count = eval_count + numel(pso_fit) + numel(de_fit);
    if toc(t_start) >= max_runtime_sec
        break;
    end

    success_F = zeros(NP, 1);
    success_CR = zeros(NP, 1);
    success_delta = zeros(NP, 1);
    success_count = 0;
    for i = 1:NP
        old_fit = fitness(i);
        if pso_fit(i) <= de_fit(i)
            trial = pso_candidates(i, :);
            trial_fit = pso_fit(i);
        else
            trial = de_candidates(i, :);
            trial_fit = de_fit(i);
        end
        if trial_fit <= fitness(i)
            archive = [archive; population(i, :)]; %#ok<AGROW>
            population(i, :) = trial;
            fitness(i) = trial_fit;
            success_count = success_count + 1;
            success_delta(success_count) = max(0, old_fit - trial_fit);
            success_F(success_count) = sampled_F(i);
            success_CR(success_count) = sampled_CR(i);
        end
        if trial_fit < pbest_fit(i)
            pbest(i, :) = trial;
            pbest_fit(i) = trial_fit;
        end
    end
    if size(archive, 1) > NP
        archive = archive(randperm(size(archive, 1), NP), :);
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
    end
    convergence_raw(iter) = best_raw;
end

runtime = toc(t_start);
convergence_raw = convergence_raw(1:actual_iter);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(convergence_raw, problem);
result.raw_convergence_curve = convergence_raw;
result.runtime = runtime;
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nParticle Swarm Optimization (PSO)');
result.combination_number = 3;
result.agent_id = 'Agent1';
result.problem = problem;

if verbose
    fprintf('DE-PSO finished %s %dD F%d: best %.12g, report %.12g, runtime %.4f s.\n', ...
        problem.suite, D, problem.func_num, result.best_value, result.record_value, runtime);
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
