function result = SOP_agent2_elite_swarm(problem, seed, options)
% Agent2: coupled elite swarm search using GWO, WOA, and HHO operators.
%
% Literature basis: Grey Wolf Optimizer leadership hierarchy, Whale
% Optimization Algorithm spiral/encircling search, and Harris Hawks
% Optimizer exploration/exploitation switching. The implementation is
% original and only uses the public benchmark wrapper.
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

population = lb + rand(NP, D) .* span;
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
convergence_raw = zeros(max_iter, 1);
actual_iter = 0;

for iter = 1:max_iter
    if toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = iter / max_iter;
    [fitness, order] = sort(fitness);
    population = population(order, :);

    alpha = population(1, :);
    beta = population(min(2, NP), :);
    delta = population(min(3, NP), :);
    alpha_fit = fitness(1);
    center = mean(population, 1);
    a = 2 * (1 - progress);
    escape_energy = 2 * (1 - progress) * (2 * rand(NP, 1) - 1);

    new_population = population;
    for i = 1:NP
        r = rand();
        if r < operator_probability(progress, 'gwo')
            candidate = gwo_candidate(population(i, :), alpha, beta, delta, a);
        elseif r < operator_probability(progress, 'gwo') + operator_probability(progress, 'woa')
            candidate = woa_candidate(population(i, :), alpha, a, lb, ub);
        else
            candidate = hho_candidate(population, population(i, :), alpha, center, escape_energy(i), progress, lb, ub);
        end

        if rand() < 0.18 + 0.22 * progress
            candidate = local_elite_step(candidate, alpha, beta, span, progress);
        end

        new_population(i, :) = min(max(candidate, lb), ub);
    end

    new_fitness = SOP_cec_evaluate(new_population, problem);
    eval_count = eval_count + numel(new_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = new_fitness <= fitness;
    population(improved, :) = new_population(improved, :);
    fitness(improved) = new_fitness(improved);

    if mod(iter, max(8, round(0.05 * max_iter))) == 0
        [population, fitness, extra_eval] = elite_diversification(population, fitness, problem, alpha, lb, ub, span, progress);
        eval_count = eval_count + extra_eval;
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
    elseif alpha_fit < best_raw
        best_raw = alpha_fit;
        best_position = alpha;
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
result.algorithm_combination = sprintf('Grey Wolf Optimizer (GWO)\nWhale Optimization Algorithm (WOA)\nHarris Hawks Optimizer (HHO)');
result.combination_number = 5;
result.agent_id = 'Agent2';
result.problem = problem;

if verbose
    fprintf('Agent2 finished %s %dD F%d: best %.12g, report %.12g, runtime %.4f s.\n', ...
        problem.suite, D, problem.func_num, result.best_value, result.record_value, runtime);
end
end

function NP = default_population(D, func_num)
if D <= 10
    NP = 55;
elseif D <= 30
    NP = 65;
elseif D <= 50
    NP = 85;
else
    NP = 105;
end
if func_num >= 11
    NP = NP + 10;
end
end

function max_iter = default_iterations(D, NP)
target_fe = min(9000 * D, 100000);
max_iter = max(60, floor((target_fe - NP) / NP));
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end

function p = operator_probability(progress, name)
switch name
    case 'gwo'
        p = 0.42 - 0.10 * progress;
    case 'woa'
        p = 0.25 + 0.10 * progress;
    otherwise
        p = 0.33;
end
end

function candidate = gwo_candidate(x, alpha, beta, delta, a)
D = numel(x);
A1 = 2 * a * rand(1, D) - a;
C1 = 2 * rand(1, D);
A2 = 2 * a * rand(1, D) - a;
C2 = 2 * rand(1, D);
A3 = 2 * a * rand(1, D) - a;
C3 = 2 * rand(1, D);
X1 = alpha - A1 .* abs(C1 .* alpha - x);
X2 = beta - A2 .* abs(C2 .* beta - x);
X3 = delta - A3 .* abs(C3 .* delta - x);
candidate = (X1 + X2 + X3) / 3;
end

function candidate = woa_candidate(x, alpha, a, lb, ub)
D = numel(x);
p = rand();
A = 2 * a * rand(1, D) - a;
C = 2 * rand(1, D);
if p < 0.5
    if norm(A, 2) / sqrt(D) < 1
        candidate = alpha - A .* abs(C .* alpha - x);
    else
        random_point = lb + rand(1, D) .* (ub - lb);
        candidate = random_point - A .* abs(C .* random_point - x);
    end
else
    b = 1;
    ell = -1 + 2 * rand(1, D);
    distance = abs(alpha - x);
    candidate = distance .* exp(b .* ell) .* cos(2 * pi .* ell) + alpha;
end
end

function candidate = hho_candidate(population, x, alpha, center, E, progress, lb, ub)
D = numel(x);
q = rand();
J = 2 * (1 - rand(1, D));
if abs(E) >= 1
    random_hawk = population(randi(size(population, 1)), :);
    if q < 0.5
        candidate = random_hawk - rand(1, D) .* abs(random_hawk - 2 * rand(1, D) .* x);
    else
        candidate = (alpha - center) - rand(1, D) .* (lb + rand(1, D) .* (ub - lb));
    end
else
    if q >= 0.5 && abs(E) >= 0.5
        candidate = alpha - E .* abs(J .* alpha - x);
    elseif q >= 0.5 && abs(E) < 0.5
        candidate = alpha - E .* abs(alpha - x);
    else
        Y = alpha - E .* abs(J .* alpha - x);
        levy = levy_step(D);
        S = 0.01 * (1 - progress) .* (ub - lb);
        candidate = Y + randn(1, D) .* S .* levy;
    end
end
end

function candidate = local_elite_step(candidate, alpha, beta, span, progress)
radius = (0.12 * (1 - progress) + 0.01) .* span;
candidate = 0.70 * candidate + 0.20 * alpha + 0.10 * beta + randn(size(candidate)) .* radius;
end

function L = levy_step(D)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
    (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
u = randn(1, D) * sigma;
v = randn(1, D);
L = u ./ (abs(v) .^ (1 / beta) + eps);
end

function [population, fitness, eval_count] = elite_diversification(population, fitness, problem, alpha, lb, ub, span, progress)
NP = size(population, 1);
D = size(population, 2);
count = max(1, ceil(0.12 * NP));
[~, order] = sort(fitness, 'descend');
replace_idx = order(1:count);
radius = (0.18 * (1 - progress) + 0.015) .* span;
new_points = alpha + randn(count, D) .* radius;
mask = rand(count, D) < (0.12 + 0.10 * (1 - progress));
random_points = lb + rand(count, D) .* span;
new_points(mask) = random_points(mask);
new_points = min(max(new_points, lb), ub);
new_fitness = SOP_cec_evaluate(new_points, problem);
eval_count = numel(new_fitness);
accept = new_fitness < fitness(replace_idx);
population(replace_idx(accept), :) = new_points(accept, :);
fitness(replace_idx(accept)) = new_fitness(accept);
end
