function result = SOP_agent_operator_pool(problem, seed, options)
% Adaptive operator-pool metaheuristic.
%
% A single population is updated by a pool of literature-grounded operators:
% DE/current-to-pbest, GSK knowledge sharing, RIME hard-rime puncture, and
% elite covariance sampling. Operators compete through greedy selection.
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
NP = get_option(options, 'population_num', 180);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_iter = get_option(options, 'max_iter', max(1, floor((max_fes - NP) / NP)));
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
profile = lower(string(get_option(options, 'profile', 'de_gsk_cma')));

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
curve = zeros(max_iter, 1);
operator_success = ones(1, 4);
operator_trials = 2 * ones(1, 4);
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
    trial = population;
    op_used = zeros(NP, 1);
    probs = operator_probabilities(profile, operator_success, operator_trials, progress);
    cum_probs = cumsum(probs);
    combined = [population; archive];
    for i = 1:NP
        op = find(cum_probs >= rand(), 1, 'first');
        op_used(i) = op;
        switch op
            case 1
                trial(i, :) = de_candidate(population, combined, i, lb, ub, progress);
            case 2
                trial(i, :) = gsk_candidate(population, fitness, i, lb, ub, progress);
            case 3
                trial(i, :) = rime_candidate(population(i, :), fitness(i), fitness, best_position, lb, ub, progress);
            otherwise
                trial(i, :) = cma_candidate(population, fitness, lb, ub, span, progress);
        end
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
        if size(archive, 1) > NP
            archive = archive(randperm(size(archive, 1), NP), :);
        end
    end
    for i = 1:rows
        op = op_used(i);
        operator_trials(op) = operator_trials(op) + 1;
        if improved(i)
            operator_success(op) = operator_success(op) + 1;
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
result.final_population = population;
result.final_fitness = fitness;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nGaining-Sharing Knowledge (GSK)\nRIME/MVO-style puncture\nElite covariance sampling');
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;
end

function probs = operator_probabilities(profile, success, trials, progress)
rate = success ./ trials;
switch profile
    case "gsk_rime_de"
        base = [0.22, 0.36, 0.30, 0.12];
    case "de_gsk_cma"
        base = [0.36, 0.28, 0.10, 0.26];
    case "de_cma_exploit"
        base = [0.56, 0.06, 0.04, 0.34];
    case "de_cma_balanced"
        base = [0.48, 0.10, 0.07, 0.35];
    otherwise
        base = [0.30, 0.30, 0.20, 0.20];
end
adaptive = rate ./ sum(rate);
probs = 0.65 * base + 0.35 * adaptive;
probs(3) = probs(3) * (1.15 - 0.30 * progress);
probs(4) = probs(4) * (0.75 + 0.65 * progress);
probs = probs ./ sum(probs);
end

function candidate = de_candidate(population, combined, i, lb, ub, progress)
NP = size(population, 1);
D = size(population, 2);
p_num = max(2, round((0.08 + 0.12 * progress) * NP));
F = min(1, max(0.15, 0.55 + 0.18 * tan(pi * (rand() - 0.5))));
CR = min(1, max(0.05, 0.75 + 0.15 * randn()));
pbest = randi(p_num);
r1 = random_index_except(NP, i);
r2 = randi(size(combined, 1));
mutant = population(i, :) + F .* (population(pbest, :) - population(i, :)) + F .* (population(r1, :) - combined(r2, :));
mask = rand(1, D) < CR;
mask(randi(D)) = true;
candidate = population(i, :);
candidate(mask) = mutant(mask);
candidate = repair(candidate, population(i, :), lb, ub);
end

function candidate = gsk_candidate(population, fitness, i, lb, ub, progress)
NP = size(population, 1);
top_count = max(2, round(0.18 * NP));
mid_start = max(top_count + 1, round(0.45 * NP));
bottom_start = max(mid_start + 1, round(0.75 * NP));
KF = min(0.95, max(0.12, 0.62 * (1 - 0.30 * progress) + 0.08 * randn()));
if rand() < 0.55 * (1 - progress) + 0.20
    better = randi(top_count);
    worse = bottom_start - 1 + randi(NP - bottom_start + 1);
    r1 = random_index_except(NP, i);
    r2 = random_index_except(NP, r1);
    candidate = population(i, :) + KF .* (population(better, :) - population(worse, :)) + 0.45 * KF .* (population(r1, :) - population(r2, :));
else
    senior_top = randi(top_count);
    senior_mid = mid_start - 1 + randi(max(1, bottom_start - mid_start));
    senior_bottom = bottom_start - 1 + randi(NP - bottom_start + 1);
    candidate = population(i, :) + KF .* (population(1, :) - population(i, :)) + KF .* (population(senior_top, :) - population(senior_bottom, :)) + 0.20 * KF .* (population(senior_mid, :) - population(i, :));
end
if rank_of(fitness, i) > 0.65 * NP && rand() < 0.18
    candidate = candidate + 0.02 * randn(size(candidate)) .* (ub - lb);
end
candidate = min(max(candidate, lb), ub);
end

function candidate = rime_candidate(x, fit, fitness, best, lb, ub, progress)
norm_fit = (max(fitness) - fit + eps) ./ (max(fitness) - min(fitness) + eps);
candidate = x;
soft = (1 - progress) * cos(pi * progress / 2);
soft_mask = rand(size(x)) < (0.18 + 0.38 * (1 - progress));
candidate(soft_mask) = best(soft_mask) + soft .* randn(1, sum(soft_mask)) .* (ub(soft_mask) - lb(soft_mask));
hard_mask = rand(size(x)) < min(0.65, norm_fit);
candidate(hard_mask) = best(hard_mask);
candidate = min(max(candidate, lb), ub);
end

function candidate = cma_candidate(population, fitness, lb, ub, span, progress)
NP = size(population, 1);
elite_count = max(4, round(0.20 * NP));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.0025 * span) .^ 2 + 1e-14);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(diag(cov_matrix)));
end
scale = 0.85 - 0.45 * progress;
candidate = center + scale .* randn(1, numel(center)) * R;
if rand() < 0.35
    candidate = 0.75 * candidate + 0.25 * population(1, :);
end
candidate = min(max(candidate, lb), ub);
end

function candidate = repair(candidate, parent, lb, ub)
low = candidate < lb;
high = candidate > ub;
candidate(low) = 0.5 * (parent(low) + lb(low));
candidate(high) = 0.5 * (parent(high) + ub(high));
candidate = min(max(candidate, lb), ub);
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function rank = rank_of(fitness, i)
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
