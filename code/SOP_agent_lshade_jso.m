function result = SOP_agent_lshade_jso(problem, seed, options)
% jSO-style enhanced L-SHADE Differential Evolution.
%
% This candidate keeps L-SHADE's success-history adaptation and archive,
% then adds staged parameter control and dynamic p-best pressure inspired by
% competition-grade DE variants for difficult CEC functions.
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
NP_min = get_option(options, 'min_population_num', 4);
max_fes = get_option(options, 'max_fes', 10000 * D);
H = get_option(options, 'memory_size', 6);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);
lpsr_hold_progress = get_option(options, 'lpsr_hold_progress', inf);
lpsr_hold_min_rate = get_option(options, 'lpsr_hold_min_rate', 0);

population = lb + rand(NP_init, D) .* span;
initial_population = get_option(options, 'initial_population', []);
if ~isempty(initial_population)
    rows = min(NP_init, size(initial_population, 1));
    population(1:rows, :) = min(max(initial_population(1:rows, :), lb), ub);
end
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
if get_option(options, 'opposition_init', false)
    [population, fitness, eval_count] = opposition_initialize(problem, population, lb, ub, get_option(options, 'quasi_opposition', true));
else
    fitness = SOP_cec_evaluate(population, problem);
    eval_count = numel(fitness);
end
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
archive = zeros(0, D);
archive_age = zeros(0, 1);
mu_F = get_option(options, 'mu_F_init', 0.3) * ones(1, H);
mu_CR = get_option(options, 'mu_CR_init', 0.8) * ones(1, H);
memory_index = 1;
curve = zeros(max(1, ceil(max_fes / NP_min)), 1);
iter = 0;
eig_rate = get_option(options, 'eig_rate', 0);
eig_interval = get_option(options, 'eig_interval', 12);
elite_rate = get_option(options, 'elite_rate', 0.25);
eig_basis = eye(D);
eig_center = mean(population, 1);
injection_interval = get_option(options, 'injection_interval', 0);
stall_count = 0;
adaptive_rime_rate = get_option(options, 'rime_puncture_rate', 0);
success_grouping_inloop = get_option(options, 'success_grouping_inloop', false);
dim_success_scores = ones(1, D);

while eval_count < max_fes && size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    progress = eval_count / max_fes;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if ~isempty(archive_age)
        archive_age = archive_age + 1;
    end
    if eig_rate > 0 && (iter == 1 || mod(iter, eig_interval) == 0)
        [eig_basis, eig_center] = elite_eigen_basis(population, span, elite_rate);
    end
    combined = [population; archive];
    pool_size = size(combined, 1);
    p_start = get_option(options, 'p_rate_start', 0.25);
    p_end = get_option(options, 'p_rate_end', 0.07);
    p_rate = p_start + (p_end - p_start) * progress;
    p_num = max(2, min(NP, round(p_rate * NP)));
    trial_population = population;
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);
    rime_used = false(NP, 1);
    success_group_masks = false(NP, D);
    current_rime_rate = adaptive_rime_rate;

    for i = 1:NP
        if get_option(options, 'fixed_high_memory', false) && rand() < get_option(options, 'fixed_high_memory_rate', 1 / (H + 1))
            F = sample_F(get_option(options, 'fixed_high_F', 0.9), progress);
            CR = sample_CR(get_option(options, 'fixed_high_CR', 0.9), progress);
        else
            mem = randi(H);
            F = sample_F(mu_F(mem), progress);
            CR = sample_CR(mu_CR(mem), progress);
        end
        if get_option(options, 'official_F_cap', false) && progress < 0.60
            F = min(F, 0.70);
        end
        if get_option(options, 'stage_limit_params', false)
            F = min(get_option(options, 'stage_F_max', 0.92), max(get_option(options, 'stage_F_min', 0.18 + 0.10 * progress), F));
            CR = min(get_option(options, 'stage_CR_max', 0.96), max(get_option(options, 'stage_CR_min', 0.18), CR));
        end
        if get_option(options, 'sinusoidal_parameter_rate', 0) > 0 && ...
                rand() < get_option(options, 'sinusoidal_parameter_rate', 0)
            phase = 2 * pi * (get_option(options, 'sinusoidal_frequency', 1.0) * progress + rand());
            F = min(1.0, max(0.05, F + get_option(options, 'sinusoidal_F_amp', 0.18) * sin(phase)));
            CR = min(1.0, max(0.02, CR + get_option(options, 'sinusoidal_CR_amp', 0.14) * cos(phase)));
        end
        sampled_F(i) = F;
        sampled_CR(i) = CR;
        pbest = randi(p_num);
        if get_option(options, 'ranked_r1', false)
            r1 = ranked_index_except(NP, i, progress, get_option(options, 'rank_pressure', 2.0));
        else
            r1 = random_index_except(NP, i);
        end
        if lower(string(get_option(options, 'archive_sampling_mode', 'uniform'))) == "stratified" && ...
                ~isempty(archive) && rand() < get_option(options, 'stratified_archive_rate', 0.52)
            r2 = NP + stratified_archive_index(archive, archive_age, best_position, progress, options);
        else
            r2 = random_pool_index(pool_size, [i r1]);
        end
        if get_option(options, 'official_stage_weight', false)
            if progress < 0.20
                weight = 0.70;
            elseif progress < 0.40
                weight = 0.80;
            else
                weight = 1.20;
            end
        else
            weight = get_option(options, 'weight_start', 0.7) + ...
                (get_option(options, 'weight_end', 1.3) - get_option(options, 'weight_start', 0.7)) * progress;
        end
        mutant = population(i, :) ...
            + weight .* F .* (population(pbest, :) - population(i, :)) ...
            + F .* (population(r1, :) - combined(r2, :));
        if eig_rate > 0 && rand() < eig_rate
            trial = eigen_crossover(population(i, :), mutant, CR, eig_basis, eig_center);
        elseif get_option(options, 'uncrossed_cauchy_rate', 0) > 0
            trial = rde_cauchy_crossover(population(i, :), mutant, CR, span, ...
                get_option(options, 'uncrossed_cauchy_rate', 0), get_option(options, 'uncrossed_cauchy_scale', 0.0025), progress);
        else
            trial = binomial_crossover(population(i, :), mutant, CR);
        end
        if success_grouping_inloop && progress >= get_option(options, 'success_group_start_progress', 0.16)
            group_rate = get_option(options, 'success_group_rate_start', 0.18) + ...
                (get_option(options, 'success_group_rate_end', 0.055) - get_option(options, 'success_group_rate_start', 0.18)) * progress;
            group_mask = sample_success_group(dim_success_scores, group_rate, ...
                get_option(options, 'success_group_min_dims', max(3, ceil(0.035 * D))), ...
                get_option(options, 'success_group_random_rate', 0.22));
            trial(~group_mask) = population(i, ~group_mask);
            if all(abs(trial - population(i, :)) <= 1e-14 .* max(1, span))
                [~, force_order] = sort(dim_success_scores .* (0.35 + rand(1, D)), 'descend');
                force_dim = force_order(1);
                trial(force_dim) = mutant(force_dim);
                group_mask(force_dim) = true;
            end
            success_group_masks(i, :) = group_mask;
        end
        if get_option(options, 'elite_blx_pulse_rate', 0) > 0 && ...
                progress >= get_option(options, 'elite_blx_start_progress', 0.18) && ...
                rand() < get_option(options, 'elite_blx_pulse_rate', 0)
            trial = elite_blx_pulse_trial(trial, population, best_position, span, lb, ub, options, progress);
        end
        if current_rime_rate > 0 && ...
                progress >= get_option(options, 'rime_start_progress', 0.35) && ...
                rand() < current_rime_rate
            trial = rime_puncture_trial(population(i, :), fitness(i), fitness, best_position, ...
                lb, ub, span, progress, options);
            rime_used(i) = true;
        end
        trial_population(i, :) = repair_bounds(trial, population(i, :), lb, ub);
    end

    remaining = max_fes - eval_count;
    if remaining < NP
        trial_population = trial_population(1:remaining, :);
        rime_used = rime_used(1:remaining);
        success_group_masks = success_group_masks(1:remaining, :);
    end
    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end

    success_F = [];
    success_CR = [];
    success_delta = [];
    rime_attempts = 0;
    rime_successes = 0;
    rows = numel(trial_fitness);
    for i = 1:rows
        if rime_used(i)
            rime_attempts = rime_attempts + 1;
        end
        if trial_fitness(i) <= fitness(i)
            improvement = max(0, fitness(i) - trial_fitness(i));
            if success_grouping_inloop
                group_mask = success_group_masks(i, :);
                if any(group_mask)
                    scaled_step = abs(trial_population(i, group_mask) - population(i, group_mask)) ./ max(span(group_mask), eps);
                    dim_success_scores(group_mask) = dim_success_scores(group_mask) + ...
                        improvement .* (0.04 + min(1.0, scaled_step));
                end
            end
            archive = [archive; population(i, :)]; %#ok<AGROW>
            archive_age = [archive_age; 0]; %#ok<AGROW>
            success_delta(end + 1, 1) = improvement; %#ok<AGROW>
            success_F(end + 1, 1) = sampled_F(i); %#ok<AGROW>
            success_CR(end + 1, 1) = sampled_CR(i); %#ok<AGROW>
            if rime_used(i)
                rime_successes = rime_successes + 1;
            end
            population(i, :) = trial_population(i, :);
            fitness(i) = trial_fitness(i);
        end
    end
    if success_grouping_inloop
        dim_success_scores = max(1e-9, get_option(options, 'success_group_decay', 0.985) .* dim_success_scores);
        mean_score = mean(dim_success_scores);
        if mean_score > 0
            dim_success_scores = dim_success_scores ./ mean_score;
        end
    end
    if get_option(options, 'rime_success_gate', false) && rime_attempts > 0
        success_rate = rime_successes / rime_attempts;
        target_rate = get_option(options, 'rime_gate_success_target', 0.05);
        if success_rate < target_rate
            adaptive_rime_rate = max(get_option(options, 'rime_rate_floor', 0), ...
                adaptive_rime_rate * get_option(options, 'rime_gate_decay', 0.62));
        else
            adaptive_rime_rate = min(get_option(options, 'rime_puncture_rate', 0), ...
                adaptive_rime_rate * get_option(options, 'rime_gate_boost', 1.10));
        end
    end
    archive_start = get_option(options, 'archive_factor_start', 1.4);
    archive_end = get_option(options, 'archive_factor_end', 2.6);
    archive_limit = round((archive_start + (archive_end - archive_start) * progress) * NP);
    if size(archive, 1) > archive_limit
        keep = randperm(size(archive, 1), archive_limit);
        archive = archive(keep, :);
        archive_age = archive_age(keep);
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
    improved_best = current_best < best_raw;
    if improved_best
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    if improved_best
        stall_count = 0;
    else
        stall_count = stall_count + 1;
    end
    if injection_interval > 0 && stall_count >= injection_interval && eval_count < max_fes && toc(t_start) < max_runtime_sec
        inject_count = max(4, round(get_option(options, 'injection_rate', 0.08) * NP));
        inject_count = min(inject_count, max_fes - eval_count);
        if inject_count > 0
            inject_radius = get_option(options, 'injection_radius', 0.012) * (1 - 0.60 * progress);
            candidates = build_elite_injection(population, best_position, lb, ub, span, inject_count, inject_radius, progress);
            candidate_fitness = SOP_cec_evaluate(candidates, problem);
            eval_count = eval_count + numel(candidate_fitness);
            [fitness, order] = sort(fitness);
            population = population(order, :);
            replace_count = min(numel(candidate_fitness), size(population, 1));
            for k = 1:replace_count
                row = size(population, 1) - k + 1;
                if candidate_fitness(k) < fitness(row)
                    archive = [archive; population(row, :)]; %#ok<AGROW>
                    archive_age = [archive_age; 0]; %#ok<AGROW>
                    population(row, :) = candidates(k, :);
                    fitness(row) = candidate_fitness(k);
                end
            end
            [current_best, current_idx] = min(fitness);
            if current_best < best_raw
                best_raw = current_best;
                best_position = population(current_idx, :);
                stall_count = 0;
            else
                stall_count = max(0, floor(stall_count / 2));
            end
        end
    end
    curve(iter) = best_raw;

    target_NP = round(NP_init - (NP_init - NP_min) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
    if eval_count / max_fes >= lpsr_hold_progress
        target_NP = max(target_NP, round(lpsr_hold_min_rate * NP_init));
    end
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness);
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
end
curve = curve(1:iter);

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
result.iteration = iter;
result.population_num = NP_init;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Differential Evolution (DE)\njSO-style enhanced L-SHADE staged parameter adaptation');
if success_grouping_inloop
    result.algorithm_combination = sprintf('%s\nSuccess-history in-loop variable grouping crossover', result.algorithm_combination);
end
if lower(string(get_option(options, 'archive_sampling_mode', 'uniform'))) == "stratified"
    result.algorithm_combination = sprintf('%s\nStratified external archive donor sampling', result.algorithm_combination);
end
result.combination_number = 1;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('L-SHADE-jSO finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function F = sample_F(mu, progress)
if progress < 0.6
    mu = max(mu, 0.35);
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
F = min(F, 0.9);
end

function CR = sample_CR(mu, progress)
if progress < 0.25
    mu = max(mu, 0.7);
elseif progress < 0.5
    mu = max(mu, 0.6);
end
CR = min(1, max(0, mu + 0.1 * randn()));
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function idx = ranked_index_except(NP, banned, progress, pressure)
power = max(1.05, pressure + 1.2 * progress);
idx = min(NP, max(1, floor(NP * (rand() ^ power)) + 1));
tries = 0;
while idx == banned && tries < 20
    idx = min(NP, max(1, floor(NP * (rand() ^ power)) + 1));
    tries = tries + 1;
end
if idx == banned
    idx = random_index_except(NP, banned);
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

function idx = stratified_archive_index(archive, archive_age, best, progress, options)
count = size(archive, 1);
if count <= 1
    idx = 1;
    return;
end
recent_rate = get_option(options, 'stratified_recent_rate', 0.42) * max(0, 1 - 0.55 * progress);
stale_rate = get_option(options, 'stratified_stale_rate', 0.24) * (0.35 + 0.65 * progress);
draw = rand();
if draw < recent_rate
    [~, order] = sort(archive_age, 'ascend');
    pool_count = max(1, min(count, round(get_option(options, 'stratified_recent_pool_rate', 0.28) * count)));
    pool = order(1:pool_count);
elseif draw < recent_rate + stale_rate
    [~, order] = sort(archive_age, 'descend');
    pool_count = max(1, min(count, round(get_option(options, 'stratified_stale_pool_rate', 0.30) * count)));
    pool = order(1:pool_count);
else
    distance = sum((archive - best) .^ 2, 2);
    [~, order] = sort(distance, 'ascend');
    pool_count = max(1, min(count, round(get_option(options, 'stratified_elite_near_pool_rate', 0.24) * count)));
    pool = order(1:pool_count);
end
idx = pool(randi(numel(pool)));
end

function mask = sample_success_group(scores, rate, min_dims, random_rate)
D = numel(scores);
count = min(D, max(min_dims, round(rate * D)));
if rand() < random_rate
    idx = randperm(D, count);
else
    scores = max(scores(:)', 1e-12);
    jitter = 0.35 + rand(1, D);
    [~, order] = sort(scores .* jitter, 'descend');
    idx = order(1:count);
    if rand() < 0.35 && count < D
        replace_count = max(1, round(0.20 * count));
        replace_rows = randperm(count, replace_count);
        random_idx = randperm(D, replace_count);
        idx(replace_rows) = random_idx;
        idx = unique(idx, 'stable');
        if numel(idx) < count
            fill = setdiff(randperm(D), idx, 'stable');
            idx = [idx, fill(1:count - numel(idx))]; %#ok<AGROW>
        end
    end
end
mask = false(1, D);
mask(idx) = true;
end

function trial = binomial_crossover(parent, mutant, CR)
D = numel(parent);
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial = parent;
trial(mask) = mutant(mask);
end

function trial = elite_blx_pulse_trial(trial, population, best, span, lb, ub, options, progress)
[NP, D] = size(population);
elite_count = min(NP, max(3, round(get_option(options, 'elite_blx_elite_rate', 0.18) * NP)));
parent_a = population(randi(elite_count), :);
parent_b = population(randi(elite_count), :);
lo = min(parent_a, parent_b);
hi = max(parent_a, parent_b);
width = max(hi - lo, 1e-12 .* span);
alpha = get_option(options, 'elite_blx_alpha', 0.22);
child = lo - alpha .* width + rand(1, D) .* ((1 + 2 * alpha) .* width);
if rand() < get_option(options, 'elite_blx_best_pull_rate', 0.36)
    child = child + get_option(options, 'elite_blx_best_pull', 0.14) .* rand() .* (best - child);
end
block_rate = get_option(options, 'elite_blx_block_rate', max(0.035, 5 / D));
block_rate = block_rate * (1 - get_option(options, 'elite_blx_block_decay', 0.30) * progress);
mask = rand(1, D) < block_rate;
if ~any(mask)
    mask(randi(D)) = true;
end
trial(mask) = child(mask);
if rand() < get_option(options, 'elite_blx_noise_rate', 0.16)
    noise_mask = rand(1, D) < max(0.025, 0.55 * block_rate);
    trial(noise_mask) = trial(noise_mask) + randn(1, sum(noise_mask)) .* ...
        (get_option(options, 'elite_blx_noise_scale', 0.0012) .* span(noise_mask));
end
trial = min(max(trial, lb), ub);
end

function trial = rde_cauchy_crossover(parent, mutant, CR, span, jump_rate, jump_scale, progress)
D = numel(parent);
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial = parent;
trial(mask) = mutant(mask);
free_mask = ~mask & (rand(1, D) < jump_rate);
if any(free_mask)
    noise = tan(pi * (rand(1, sum(free_mask)) - 0.5));
    noise = min(max(noise, -8), 8);
    scale = jump_scale * (1 - 0.65 * progress);
    trial(free_mask) = parent(free_mask) + noise .* scale .* span(free_mask);
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
[basis, values] = eig(cov_matrix, 'vector');
if ~isvector(values)
    eig_values = diag(values);
else
    eig_values = values;
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

function candidates = build_elite_injection(population, best, lb, ub, span, count, radius_scale, progress)
[NP, D] = size(population);
elite_count = max(4, min(NP, round(0.18 * NP)));
center = mean(population(1:elite_count, :), 1);
candidates = repmat(best, count, 1);
radius = max(1e-8, radius_scale) .* span;
for i = 1:count
    mode = rand();
    if mode < 0.40
        a = randi(elite_count);
        b = randi(elite_count);
        while b == a
            b = randi(elite_count);
        end
        F = min(0.9, max(0.15, 0.42 + 0.16 * randn()));
        x = best + F .* (population(a, :) - population(b, :));
    elseif mode < 0.75
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -7), 7);
        mask = rand(1, D) < max(0.03, 5 / D);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        x = best;
        x(mask) = x(mask) + noise(mask) .* radius(mask);
    else
        donor = population(randi(elite_count), :);
        mirror = 2 .* center - donor;
        x = best + (0.25 + 0.35 * rand()) .* (mirror - best);
        if rand() < 0.50 * (1 - progress)
            x = x + randn(1, D) .* (0.25 .* radius);
        end
    end
    candidates(i, :) = min(max(x, lb), ub);
end
end

function trial = rime_puncture_trial(parent, parent_fitness, fitness, best, lb, ub, span, progress, options)
D = numel(parent);
trial = parent;
norm_fit = (max(fitness) - parent_fitness + eps) ./ (max(fitness) - min(fitness) + eps);
block_rate = get_option(options, 'rime_block_rate', min(0.10, max(0.025, 5 / D)));
if rand() < get_option(options, 'rime_hard_probability', 0.55)
    mask = rand(1, D) < min(0.80, max(block_rate, norm_fit));
    if ~any(mask)
        mask(randi(D)) = true;
    end
    trial(mask) = best(mask);
else
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randi(D)) = true;
    end
    soft = (1 - progress) * cos(pi * progress / 2);
    noise = randn(1, D);
    cauchy_mask = rand(1, D) < 0.35;
    cauchy_noise = tan(pi * (rand(1, D) - 0.5));
    cauchy_noise = min(max(cauchy_noise, -7), 7);
    noise(cauchy_mask) = cauchy_noise(cauchy_mask);
    radius = get_option(options, 'rime_radius', 0.004) .* span;
    trial(mask) = best(mask) + soft .* noise(mask) .* radius(mask);
end
if rand() < get_option(options, 'rime_elite_blend_rate', 0.25)
    blend = get_option(options, 'rime_elite_blend_weight', 0.35);
    trial = (1 - blend) .* trial + blend .* best;
end
trial = min(max(trial, lb), ub);
end

function [population, fitness, eval_count] = opposition_initialize(problem, population, lb, ub, use_quasi)
NP = size(population, 1);
opposite = repmat(lb + ub, NP, 1) - population;
if use_quasi
    center = repmat(0.5 .* (lb + ub), NP, 1);
    quasi = center + rand(NP, size(population, 2)) .* (opposite - center);
    candidates = [population; opposite; quasi];
else
    candidates = [population; opposite];
end
candidates = min(max(candidates, lb), ub);
values = SOP_cec_evaluate(candidates, problem);
[fitness_all, order] = sort(values);
population = candidates(order(1:NP), :);
fitness = fitness_all(1:NP);
eval_count = numel(values);
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
