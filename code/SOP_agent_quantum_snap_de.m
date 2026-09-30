function result = SOP_agent_quantum_snap_de(problem, seed, options)
% Quantum/opposition/snap Differential Evolution for stepped multimodal cases.
%
% This Agent1 candidate is designed for CEC2017 100D F5/F8 style Rastrigin
% landscapes. It keeps the search derivative-free and objective-only, with
% restart-free population injection instead of RSP/MTS/CMAES post-processing.
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
NP_init = get_option(options, 'population_num', max(180, 10 * D));
NP_min = get_option(options, 'min_population_num', max(80, round(0.16 * NP_init)));
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
profile = lower(string(get_option(options, 'profile', 'balanced')));
H = get_option(options, 'memory_size', 6);
p_input = get_option(options, 'p_rate', 0.08);

population = initialize_population(NP_init, D, lb, ub, span, options);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
mu_F = get_option(options, 'mu_F_init', 0.42) * ones(1, H);
mu_CR = get_option(options, 'mu_CR_init', 0.82) * ones(1, H);
memory_index = 1;
operator_success = ones(1, 4);
operator_trials = 2 * ones(1, 4);
max_iter = get_option(options, 'max_iter', max(1, floor((max_fes - NP_init) / NP_min)));
curve = zeros(max_iter + 512, 1);
iter = 0;
stagnation = 0;
stagnation_limit = get_option(options, 'stagnation_limit', max(45, round(0.018 * max_iter)));
inject_interval = get_option(options, 'inject_interval', max(35, round(0.012 * max_iter)));

while eval_count < max_fes && size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    progress = min(1, eval_count / max(1, max_fes));
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_position = population(1, :);
        stagnation = 0;
    end

    rows = min(NP, max_fes - eval_count);
    if rows <= 0
        break;
    end
    combined = [population; archive];
    operator_probs = operator_probabilities(profile, operator_success, operator_trials, progress);
    trial_population = population(1:rows, :);
    used_operator = zeros(rows, 1);
    sampled_F = zeros(rows, 1);
    sampled_CR = zeros(rows, 1);
    p_rate = scheduled_p_rate(options, p_input, progress);
    p_num = max(2, min(NP, round(p_rate * NP)));
    elite_center = weighted_elite_center(population, max(4, round(0.16 * NP)));
    mean_best = mean(population(1:max(4, round(0.22 * NP)), :), 1);

    for i = 1:rows
        op = sample_discrete(operator_probs);
        mem = randi(H);
        F = sample_F(mu_F(mem));
        CR = sample_CR(mu_CR(mem));
        used_operator(i) = op;
        sampled_F(i) = F;
        sampled_CR(i) = CR;
        switch op
            case 1
                trial = current_to_pbest_candidate(population, combined, i, p_num, F, CR, lb, ub);
            case 2
                trial = quantum_candidate(population(i, :), population(1, :), elite_center, mean_best, progress, lb, ub, options);
            case 3
                trial = orthogonal_opposition_candidate(population, i, elite_center, progress, lb, ub, span, options);
            otherwise
                trial = snap_de_candidate(population, combined, i, p_num, F, CR, elite_center, progress, lb, ub, options);
        end
        trial_population(i, :) = trial;
    end

    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end

    improved = trial_fitness <= fitness(1:rows);
    if any(improved)
        archive = [archive; population(improved, :)]; %#ok<AGROW>
        archive_limit = max(NP, round((get_option(options, 'archive_factor_start', 1.4) + ...
            (get_option(options, 'archive_factor_end', 2.4) - get_option(options, 'archive_factor_start', 1.4)) * progress) * NP));
        if size(archive, 1) > archive_limit
            archive = archive(randperm(size(archive, 1), archive_limit), :);
        end
    end

    success_F = [];
    success_CR = [];
    success_delta = [];
    for i = 1:rows
        op = used_operator(i);
        operator_trials(op) = operator_trials(op) + 1;
        if improved(i)
            operator_success(op) = operator_success(op) + 1;
            success_delta(end + 1, 1) = max(0, fitness(i) - trial_fitness(i)); %#ok<AGROW>
            success_F(end + 1, 1) = sampled_F(i); %#ok<AGROW>
            success_CR(end + 1, 1) = sampled_CR(i); %#ok<AGROW>
            population(i, :) = trial_population(i, :);
            fitness(i) = trial_fitness(i);
        end
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
        stagnation = 0;
    else
        stagnation = stagnation + 1;
    end

    if eval_count < max_fes && toc(t_start) < max_runtime_sec && ...
            (mod(iter, inject_interval) == 0 || stagnation >= stagnation_limit)
        [population, fitness, extra_eval, injected_better] = inject_population(population, fitness, best_position, ...
            problem, archive, max_fes - eval_count, options, progress);
        eval_count = eval_count + extra_eval;
        if injected_better
            [best_raw, best_idx] = min(fitness);
            best_position = population(best_idx, :);
            stagnation = 0;
        else
            stagnation = max(0, round(0.45 * stagnation));
        end
    end

    if iter > numel(curve)
        curve(end + 1024, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;

    target_NP = round(NP_init - (NP_init - NP_min) * progress ^ get_option(options, 'lpsr_power', 1.35));
    target_NP = max(NP_min, target_NP);
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness);
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
end

curve = curve(1:iter);
runtime = toc(t_start);
[final_fitness, order] = sort(fitness);
final_population = population(order, :);
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
result.final_population = final_population;
result.final_fitness = final_fitness;
result.algorithm_combination = sprintf('Quantum-inspired DE and jSO archive adaptation\nOrthogonal opposition learning\nElite-anchored integer-step snapping\nRestart-free population injection');
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;
end

function population = initialize_population(NP, D, lb, ub, span, options)
center = 0.5 * (lb + ub);
population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', true)
    population(1, :) = center;
end
pair_count = floor(0.28 * NP / 2);
for k = 1:pair_count
    row = 2 * k;
    if row + 1 > NP
        break;
    end
    x = lb + rand(1, D) .* span;
    q = 0.55 + 0.90 * rand(1, D);
    population(row, :) = x;
    population(row + 1, :) = center + q .* (center - x);
end
center_count = min(NP, max(4, round(get_option(options, 'center_seed_rate', 0.10) * NP)));
radius = get_option(options, 'center_seed_radius', 0.16) .* span;
for row = 2:center_count
    if rand() < 0.55
        x = center + randn(1, D) .* radius;
    else
        x = center + tan(pi * (rand(1, D) - 0.5)) .* (0.35 .* radius);
    end
    population(row, :) = x;
end
population = min(max(population, lb), ub);
end

function probs = operator_probabilities(profile, success, trials, progress)
switch profile
    case "snap_focus"
        base = [0.34, 0.18, 0.10, 0.38];
    case "quantum_focus"
        base = [0.32, 0.30, 0.18, 0.20];
    otherwise
        base = [0.42, 0.20, 0.13, 0.25];
end
adaptive = success ./ trials;
adaptive = adaptive ./ sum(adaptive);
probs = 0.62 * base + 0.38 * adaptive;
probs(2) = probs(2) * (1.10 - 0.35 * progress);
probs(4) = probs(4) * (0.78 + 0.62 * progress);
probs = probs ./ sum(probs);
end

function p_rate = scheduled_p_rate(options, p_input, progress)
p_start = get_option(options, 'p_rate_start', max(0.16, 2.2 * p_input));
p_end = get_option(options, 'p_rate_end', min(0.055, max(0.030, 0.70 * p_input)));
p_rate = p_start + (p_end - p_start) * progress;
end

function trial = current_to_pbest_candidate(population, combined, i, p_num, F, CR, lb, ub)
NP = size(population, 1);
pbest = randi(p_num);
r1 = random_index_except(NP, i);
r2 = random_pool_index(size(combined, 1), [i, r1]);
mutant = population(i, :) + F .* (population(pbest, :) - population(i, :)) + ...
    F .* (population(r1, :) - combined(r2, :));
trial = repair_bounds(binomial_crossover(population(i, :), mutant, CR), population(i, :), lb, ub);
end

function trial = quantum_candidate(x, best, elite_center, mean_best, progress, lb, ub, options)
phi = rand();
attractor = phi .* best + (1 - phi) .* elite_center;
if rand() < 0.35
    attractor = 0.50 .* attractor + 0.50 .* mean_best;
end
beta = get_option(options, 'quantum_beta_start', 0.82) + ...
    (get_option(options, 'quantum_beta_end', 0.16) - get_option(options, 'quantum_beta_start', 0.82)) * progress;
u = max(rand(size(x)), 1e-12);
direction = sign(rand(size(x)) - 0.5);
radius = abs(mean_best - x);
trial = attractor + beta .* direction .* radius .* log(1 ./ u);
trial = min(max(trial, lb), ub);
end

function trial = orthogonal_opposition_candidate(population, i, elite_center, progress, lb, ub, span, options)
D = size(population, 2);
x = population(i, :);
domain_center = 0.5 * (lb + ub);
center = (0.78 - 0.24 * progress) .* elite_center + (0.22 + 0.24 * progress) .* domain_center;
scale = get_option(options, 'opposition_scale_start', 0.95) + ...
    (get_option(options, 'opposition_scale_end', 0.35) - get_option(options, 'opposition_scale_start', 0.95)) * progress;
mask = mod((1:D) + i + floor(17 * progress * D), 2) == 0;
random_mask = rand(1, D) < get_option(options, 'opposition_random_rate', 0.18);
mask = xor(mask, random_mask);
if ~any(mask)
    mask(randi(D)) = true;
end
trial = x;
trial(mask) = center(mask) + scale .* (center(mask) - x(mask));
trial(~mask) = trial(~mask) + randn(1, sum(~mask)) .* (get_option(options, 'opposition_jitter', 0.003) .* span(~mask));
trial = min(max(trial, lb), ub);
end

function trial = snap_de_candidate(population, combined, i, p_num, F, CR, elite_center, progress, lb, ub, options)
trial = current_to_pbest_candidate(population, combined, i, p_num, F, CR, lb, ub);
if rand() <= get_option(options, 'snap_rate', 0.58)
    if rand() < get_option(options, 'snap_best_anchor_rate', 0.72)
        anchor = population(1, :);
    else
        anchor = elite_center;
    end
    trial = snap_to_elite_lattice(trial, anchor, progress, lb, ub, options);
end
end

function trial = snap_to_elite_lattice(trial, anchor, progress, lb, ub, options)
D = numel(trial);
steps = get_option(options, 'snap_steps', [1.0, 0.5, 0.25, 2.0]);
if progress < 0.45
    weights = [0.40, 0.24, 0.10, 0.26];
elseif progress < 0.78
    weights = [0.36, 0.34, 0.18, 0.12];
else
    weights = [0.28, 0.40, 0.26, 0.06];
end
step = steps(sample_discrete(weights ./ sum(weights)));
mask = rand(1, D) < get_option(options, 'snap_block_rate', max(0.04, 6 / D));
if ~any(mask)
    mask(randi(D)) = true;
end
delta = (trial(mask) - anchor(mask)) ./ step;
trial(mask) = anchor(mask) + round(delta) .* step;
trial(mask) = trial(mask) + get_option(options, 'snap_jitter', 0.025) .* step .* randn(1, sum(mask));
trial = min(max(trial, lb), ub);
end

function [population, fitness, eval_count, injected_better] = inject_population(population, fitness, best_x, problem, archive, max_fes, options, progress)
eval_count = 0;
injected_better = false;
if max_fes <= 0
    return;
end
NP = size(population, 1);
D = size(population, 2);
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
count = min(max_fes, max(8, round(get_option(options, 'inject_rate', 0.075) * NP)));
count = min(count, NP);
[fitness, order] = sort(fitness);
population = population(order, :);
elite_center = weighted_elite_center(population, max(4, round(0.16 * NP)));
mean_best = mean(population(1:max(4, round(0.20 * NP)), :), 1);
candidates = repmat(best_x, count, 1);
for k = 1:count
    mode = mod(k - 1, 4) + 1;
    switch mode
        case 1
            candidates(k, :) = quantum_candidate(population(min(k, NP), :), best_x, elite_center, mean_best, progress, lb, ub, options);
        case 2
            x = best_x + randn(1, D) .* (get_option(options, 'inject_sigma', 0.012) .* span);
            candidates(k, :) = snap_to_elite_lattice(x, best_x, progress, lb, ub, options);
        case 3
            candidates(k, :) = orthogonal_opposition_candidate(population, min(k, NP), elite_center, progress, lb, ub, span, options);
        otherwise
            donor = population(randi(max(4, round(0.25 * NP))), :);
            if ~isempty(archive)
                donor2 = archive(randi(size(archive, 1)), :);
            else
                donor2 = population(randi(NP), :);
            end
            x = best_x + get_option(options, 'inject_diff_weight', 0.22) .* (donor - donor2) + ...
                randn(1, D) .* (get_option(options, 'inject_jitter', 0.0025) .* span);
            candidates(k, :) = min(max(x, lb), ub);
    end
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
[values, order] = sort(values);
candidates = candidates(order, :);
for k = 1:count
    row = NP - k + 1;
    if values(k) < fitness(row)
        population(row, :) = candidates(k, :);
        fitness(row) = values(k);
        injected_better = true;
    end
end
end

function center = weighted_elite_center(population, elite_count)
elite_count = min(size(population, 1), elite_count);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * population(1:elite_count, :);
end

function F = sample_F(mu)
F = mu + 0.1 * tan(pi * (rand() - 0.5));
tries = 0;
while F <= 0 && tries < 20
    F = mu + 0.1 * tan(pi * (rand() - 0.5));
    tries = tries + 1;
end
if F <= 0
    F = 0.45;
end
F = min(1, F);
end

function CR = sample_CR(mu)
CR = min(1, max(0, mu + 0.08 * randn()));
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

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function idx = random_pool_index(pool_size, banned)
if numel(banned) >= pool_size
    idx = randi(pool_size);
    return;
end
idx = randi(pool_size);
tries = 0;
while any(idx == banned) && tries < 20
    idx = randi(pool_size);
    tries = tries + 1;
end
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
