function result = SOP_agent_lshade_cma(problem, seed, options)
% L-SHADE with elite covariance sampling.
%
% Literature basis: Differential Evolution/L-SHADE and covariance-adaptive
% evolutionary sampling. The added covariance phase samples along elite
% population directions only from public objective feedback.
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
max_iter = get_option(options, 'max_iter', inf);
H = get_option(options, 'memory_size', 8);
p_rate = get_option(options, 'p_rate', 0.11);
cma_rate = get_option(options, 'cma_rate', 0.18);
elite_rate = get_option(options, 'elite_rate', 0.22);
cma_interval = get_option(options, 'cma_interval', 20);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);
lpsr_hold_progress = get_option(options, 'lpsr_hold_progress', inf);
lpsr_hold_min_rate = get_option(options, 'lpsr_hold_min_rate', 0);

population = lb + rand(NP_init, D) .* span;
initial_population = get_option(options, 'initial_population', []);
initial_rows = 0;
if ~isempty(initial_population)
    rows = min(NP_init, size(initial_population, 1));
    population(1:rows, :) = min(max(initial_population(1:rows, :), lb), ub);
    initial_rows = rows;
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
        if get_option(options, 'preserve_initial_population_after_radius', false) && initial_rows > 0
            population(1:initial_rows, :) = min(max(initial_population(1:initial_rows, :), lb), ub);
        end
    end
    population(1, :) = center;
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
anchor_population = [];
anchor_fitness = [];
if get_option(options, 'retain_initial_anchors', false) && initial_rows > 0
    anchor_limit = min(initial_rows, max(2, round(get_option(options, 'anchor_retain_pool_rate', 0.32) * initial_rows)));
    [initial_sorted_fitness, initial_order] = sort(fitness(1:initial_rows));
    anchor_population = population(initial_order(1:anchor_limit), :);
    anchor_fitness = initial_sorted_fitness(1:anchor_limit);
end
archive = zeros(0, D);
external_anchor_archive = get_option(options, 'external_anchor_archive', []);
if ~isempty(external_anchor_archive)
    external_anchor_archive = min(max(external_anchor_archive(:, 1:D), lb), ub);
end
mu_F = 0.5 * ones(1, H);
mu_CR = 0.5 * ones(1, H);
memory_index = 1;
curve = zeros(max(1, ceil(max_fes / max(1, NP_min))), 1);
iter = 0;
stall_iter = 0;

while iter < max_iter && eval_count < max_fes && ...
        size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    progress = eval_count / max_fes;
    [fitness, order] = sort(fitness);
    population = population(order, :);
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
        use_anchor = ~isempty(external_anchor_archive) && ...
            progress >= get_option(options, 'anchor_archive_start_progress', 0.10) && ...
            progress <= get_option(options, 'anchor_archive_end_progress', 0.88) && ...
            rand() < get_option(options, 'anchor_archive_rate', 0);
        if use_anchor
            donor = external_anchor_archive(randi(size(external_anchor_archive, 1)), :);
        else
            r2 = random_pool_index(pool_size, [i r1]);
            donor = combined(r2, :);
        end
        mutant = population(i, :) ...
            + F .* (population(pbest, :) - population(i, :)) ...
            + F .* (population(r1, :) - donor);
        if get_option(options, 'epsde_strategy_rate', 0) > 0 && ...
                progress >= get_option(options, 'epsde_start_progress', 0.18) && ...
                progress <= get_option(options, 'epsde_end_progress', 0.86) && ...
                rand() < get_option(options, 'epsde_strategy_rate', 0)
            [mutant, F, CR] = epsde_strategy_mutant(population, combined, fitness, i, progress, options);
            sampled_F(i) = F;
            sampled_CR(i) = CR;
        end
        trial_population(i, :) = repair_bounds(binomial_crossover(population(i, :), mutant, CR), population(i, :), lb, ub);
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
    if get_option(options, 'elite_direction_rate', 0) > 0 && eval_count < max_fes && ...
            mod(iter, get_option(options, 'elite_direction_interval', 9)) == 0
        [population, fitness, extra_eval] = elite_direction_step(population, fitness, problem, lb, ub, span, ...
            eval_count, max_fes, options);
        eval_count = eval_count + extra_eval;
    end
    if get_option(options, 'inline_eda_rate', 0) > 0 && eval_count < max_fes && ...
            mod(iter, get_option(options, 'inline_eda_interval', 11)) == 0
        [population, fitness, extra_eval] = inline_eda_step(population, fitness, problem, lb, ub, span, ...
            eval_count, max_fes, options);
        eval_count = eval_count + extra_eval;
    end
    if get_option(options, 'inline_abc_rime_rate', 0) > 0 && eval_count < max_fes && ...
            mod(iter, get_option(options, 'inline_abc_rime_interval', 13)) == 0
        [population, fitness, extra_eval] = inline_abc_rime_step(population, fitness, problem, lb, ub, span, ...
            eval_count, max_fes, options);
        eval_count = eval_count + extra_eval;
    end

    [current_best, current_idx] = min(fitness);
    improved = false;
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
        improved = true;
    end
    if improved
        stall_iter = 0;
    else
        stall_iter = stall_iter + 1;
    end
    if get_option(options, 'stagnation_stepping_stone', false) && eval_count < max_fes && ...
            stall_iter >= get_option(options, 'stagnation_stall_iter', 36) && ...
            progress >= get_option(options, 'stagnation_start_progress', 0.18) && ...
            progress <= get_option(options, 'stagnation_end_progress', 0.90)
        [population, fitness, extra_eval] = stagnation_stepping_stone_step(population, fitness, problem, lb, ub, span, ...
            eval_count, max_fes, options, initial_population, best_position);
        eval_count = eval_count + extra_eval;
        [current_best, current_idx] = min(fitness);
        if current_best < best_raw
            best_raw = current_best;
            best_position = population(current_idx, :);
        end
        stall_iter = 0;
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
        if ~isempty(anchor_population) && eval_count / max_fes <= get_option(options, 'anchor_retain_until_progress', 0.82)
            [population, fitness] = retain_initial_anchor_rows(population, fitness, anchor_population, ...
                anchor_fitness, get_option(options, 'anchor_retain_count', max(3, round(0.06 * target_NP))), ...
                get_option(options, 'anchor_retain_duplicate_tol', 1e-10));
        end
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
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nElite covariance evolutionary sampling');
if ~isempty(external_anchor_archive)
    result.algorithm_combination = sprintf('%s\nExternal scout-anchor archive donor sampling', result.algorithm_combination);
end
if get_option(options, 'inline_abc_rime_rate', 0) > 0
    result.algorithm_combination = sprintf('%s\nArtificial Bee Colony/RIME sparse inline pulse', result.algorithm_combination);
end
if get_option(options, 'epsde_strategy_rate', 0) > 0
    result.algorithm_combination = sprintf('%s\nCoDE/EPSDE low-frequency strategy-pool trial pulse', result.algorithm_combination);
end
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('L-SHADE-CMA finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function [population, fitness] = retain_initial_anchor_rows(population, fitness, anchor_population, anchor_fitness, retain_count, duplicate_tol)
NP = size(population, 1);
if NP < 4 || isempty(anchor_population)
    return;
end
retain_count = min([retain_count, size(anchor_population, 1), max(0, NP - 3)]);
if retain_count <= 0
    return;
end
chosen = zeros(0, size(population, 2));
chosen_fit = zeros(0, 1);
for i = 1:size(anchor_population, 1)
    if size(chosen, 1) >= retain_count
        break;
    end
    anchor = anchor_population(i, :);
    if min(sqrt(mean((population - anchor) .^ 2, 2))) <= duplicate_tol
        continue;
    end
    chosen(end + 1, :) = anchor; %#ok<AGROW>
    chosen_fit(end + 1, 1) = anchor_fitness(i); %#ok<AGROW>
end
if isempty(chosen)
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
replace_count = min(size(chosen, 1), NP - 3);
replace_rows = (NP - replace_count + 1):NP;
population(replace_rows, :) = chosen(1:replace_count, :);
fitness(replace_rows) = chosen_fit(1:replace_count);
[fitness, order] = sort(fitness);
population = population(order, :);
end

function [population, fitness, eval_count] = stagnation_stepping_stone_step(population, fitness, problem, lb, ub, span, eval_count_so_far, max_fes, options, initial_population, best_position)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
if NP < 4
    return;
end
remaining = max_fes - eval_count_so_far;
sample_count = min(remaining, max(2, round(get_option(options, 'stagnation_pulse_rate', 0.10) * NP)));
if sample_count <= 0
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = max(4, min(NP, round(get_option(options, 'stagnation_elite_rate', 0.28) * NP)));
elites = population(1:elite_count, :);
best = population(1, :);
if ~isempty(best_position)
    best = best_position(:)';
end
if ~isempty(initial_population)
    anchor_count = min(size(initial_population, 1), max(4, round(get_option(options, 'stagnation_anchor_rate', 0.36) * size(initial_population, 1))));
    anchors = initial_population(1:anchor_count, 1:D);
else
    anchors = elites;
end
anchors = min(max(anchors, lb), ub);
radius = get_option(options, 'stagnation_radius', 0.0038) .* span;
block_rate = get_option(options, 'stagnation_block_rate', max(0.050, 5 / D));
candidates = repmat(best, sample_count, 1);
for i = 1:sample_count
    anchor = anchors(randi(size(anchors, 1)), :);
    a = elites(randi(elite_count), :);
    b = elites(randi(elite_count), :);
    c = elites(randi(elite_count), :);
    op = mod(i - 1, 3) + 1;
    switch op
        case 1
            alpha = -0.10 + 1.22 * rand(1, D);
            child = anchor + alpha .* (a - anchor);
            child = child + (0.10 + 0.28 * rand()) .* (best - child);
            child = child + randn(1, D) .* (0.20 .* radius);
        case 2
            F = 0.28 + 0.40 * rand();
            G = 0.08 + 0.20 * rand();
            H = 0.06 + 0.16 * rand();
            child = a + F .* (best - a) + G .* (anchor - b) + H .* (b - c);
        otherwise
            child = 0.55 .* anchor + 0.45 .* a;
            mask = rand(1, D) < block_rate;
            if ~any(mask)
                mask(randi(D)) = true;
            end
            noise = tan(pi * (rand(1, D) - 0.5));
            noise = min(max(noise, -6), 6);
            child(mask) = best(mask) + noise(mask) .* radius(mask);
    end
    if rand() < get_option(options, 'stagnation_micro_blx_rate', 0.32)
        mask = rand(1, D) < block_rate;
        if any(mask)
            lo = min(anchor(mask), a(mask));
            hi = max(anchor(mask), a(mask));
            width = max(hi - lo, 1e-12 .* span(mask));
            alpha = get_option(options, 'stagnation_blx_alpha', 0.18);
            child(mask) = lo - alpha .* width + rand(1, nnz(mask)) .* ((1 + 2 * alpha) .* width);
        end
    end
    candidates(i, :) = min(max(child, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
[~, worst_order] = sort(fitness, 'descend');
replace_count = min(sample_count, numel(worst_order));
for i = 1:replace_count
    idx = worst_order(i);
    if values(i) < fitness(idx)
        population(idx, :) = candidates(i, :);
        fitness(idx) = values(i);
    end
end
end

function [population, fitness, eval_count] = inline_eda_step(population, fitness, problem, lb, ub, span, eval_count_so_far, max_fes, options)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
if NP < 2
    return;
end
remaining = max_fes - eval_count_so_far;
sample_count = min(remaining, max(2, round(get_option(options, 'inline_eda_rate', 0.08) * NP)));
if sample_count <= 0
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = min(NP, max(2, round(get_option(options, 'inline_eda_elite_rate', 0.18) * NP)));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
best = population(1, :);
center = weights * elites;
center = 0.72 * best + 0.28 * center;
centered = elites - center;
iso_scale = get_option(options, 'inline_eda_iso_scale', 0.00022) .* span;
cov_matrix = centered' * (centered .* weights') + diag((iso_scale .^ 2) + 1e-16);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), 1e-16)));
end
cov_scale = get_option(options, 'inline_eda_cov_scale', 0.035);
block_rate = get_option(options, 'inline_eda_block_rate', max(0.035, 5 / D));
candidates = repmat(best, sample_count, 1);
full_samples = center + randn(sample_count, D) * R * cov_scale;
for i = 1:sample_count
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randi(D)) = true;
    end
    if rand() < get_option(options, 'inline_eda_full_rate', 0.18)
        child = full_samples(i, :);
    else
        child = best;
        child(mask) = full_samples(i, mask);
        if rand() < get_option(options, 'inline_eda_cauchy_rate', 0.16)
            cauchy_noise = tan(pi * (rand(1, D) - 0.5));
            cauchy_noise = min(max(cauchy_noise, -6), 6) .* iso_scale;
            child(mask) = child(mask) + cauchy_noise(mask);
        end
    end
    candidates(i, :) = min(max(child, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
[~, worst_order] = sort(fitness, 'descend');
replace_count = min(sample_count, numel(worst_order));
for i = 1:replace_count
    idx = worst_order(i);
    if values(i) < fitness(idx)
        population(idx, :) = candidates(i, :);
        fitness(idx) = values(i);
    end
end
end

function [population, fitness, eval_count] = inline_abc_rime_step(population, fitness, problem, lb, ub, span, eval_count_so_far, max_fes, options)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
if NP < 4
    return;
end
remaining = max_fes - eval_count_so_far;
sample_count = min(remaining, max(2, round(get_option(options, 'inline_abc_rime_rate', 0.055) * NP)));
if sample_count <= 0
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = max(3, min(NP, round(get_option(options, 'inline_abc_rime_elite_rate', 0.20) * NP)));
parent_pool = max(elite_count, min(NP, round(get_option(options, 'inline_abc_rime_parent_rate', 0.55) * NP)));
best = population(1, :);
block_rate = get_option(options, 'inline_abc_rime_block_rate', max(0.035, 5 / D));
neighbor_scale = get_option(options, 'inline_abc_neighbor_scale', 0.52);
best_pull = get_option(options, 'inline_abc_best_pull', 0.16);
rime_hard_rate = get_option(options, 'inline_rime_hard_rate', 0.30);
rime_noise_scale = get_option(options, 'inline_rime_noise_scale', 0.0010) .* span;
candidates = zeros(sample_count, D);
parent_rows = zeros(sample_count, 1);
for i = 1:sample_count
    parent_rows(i) = randi(parent_pool);
    x = population(parent_rows(i), :);
    neighbor = population(randi(parent_pool), :);
    while isequal(neighbor, x) && parent_pool > 1
        neighbor = population(randi(parent_pool), :);
    end
    elite = population(randi(elite_count), :);
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randi(D)) = true;
    end
    phi = -1 + 2 * rand(1, D);
    child = x;
    if rand() < get_option(options, 'inline_abc_onlooker_rate', 0.42)
        child(mask) = x(mask) + neighbor_scale .* phi(mask) .* (elite(mask) - neighbor(mask)) + ...
            best_pull .* rand(1, nnz(mask)) .* (best(mask) - x(mask));
    else
        child(mask) = x(mask) + neighbor_scale .* phi(mask) .* (x(mask) - neighbor(mask)) + ...
            best_pull .* rand(1, nnz(mask)) .* (best(mask) - x(mask));
    end
    if rand() < rime_hard_rate
        hard_mask = mask & (rand(1, D) < get_option(options, 'inline_rime_mask_rate', 0.42));
        if ~any(hard_mask)
            hard_mask(mask) = rand(1, nnz(mask)) < 0.50;
        end
        child(hard_mask) = best(hard_mask);
    end
    if rand() < get_option(options, 'inline_rime_soft_rate', 0.55)
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -6), 6);
        child(mask) = child(mask) + noise(mask) .* rime_noise_scale(mask);
    end
    candidates(i, :) = min(max(child, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
for i = 1:sample_count
    row = parent_rows(i);
    if values(i) < fitness(row)
        population(row, :) = candidates(i, :);
        fitness(row) = values(i);
    end
end
if get_option(options, 'inline_abc_rime_replace_worst', false)
    [~, worst_order] = sort(fitness, 'descend');
    for i = 1:min(sample_count, numel(worst_order))
        row = worst_order(i);
        if values(i) < fitness(row)
            population(row, :) = candidates(i, :);
            fitness(row) = values(i);
        end
    end
end
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
cov_matrix = centered' * (centered .* weights');
diag_floor = (0.0025 * span) .^ 2 + 1e-14;
cov_matrix = cov_matrix + diag(diag_floor);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(diag(cov_matrix)));
end
candidates = center + randn(sample_count, D) * R;
if rand() < 0.5
    best = population(1, :);
    candidates = 0.75 * candidates + 0.25 * repmat(best, sample_count, 1);
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

function [population, fitness, eval_count] = elite_direction_step(population, fitness, problem, lb, ub, span, eval_count_so_far, max_fes, options)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
remaining = max_fes - eval_count_so_far;
sample_count = min(remaining, max(2, round(get_option(options, 'elite_direction_rate', 0.10) * NP)));
if sample_count <= 0
    return;
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = max(4, min(NP, round(get_option(options, 'elite_direction_elite_rate', 0.20) * NP)));
elites = population(1:elite_count, :);
best = population(1, :);
center = mean(elites, 1);
block_rate = get_option(options, 'elite_direction_block_rate', max(0.05, 6 / D));
pull_scale = get_option(options, 'elite_direction_pull_scale', 0.35);
diff_scale = get_option(options, 'elite_direction_diff_scale', 0.32);
noise_sigma = get_option(options, 'elite_direction_noise_sigma', 0.0012) .* span;
candidates = repmat(best, sample_count, 1);
for i = 1:sample_count
    a = elites(randi(elite_count), :);
    b = elites(randi(elite_count), :);
    c = elites(randi(elite_count), :);
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randi(D)) = true;
    end
    child = best;
    if rand() < 0.55
        step = pull_scale .* rand(1, D) .* (a - best) + diff_scale .* rand(1, D) .* (b - c);
    else
        step = pull_scale .* rand(1, D) .* (center - best) + diff_scale .* randn(1, D) .* (a - b);
    end
    child(mask) = child(mask) + step(mask);
    if rand() < get_option(options, 'elite_direction_cauchy_rate', 0.18)
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -6), 6) .* noise_sigma;
    else
        noise = randn(1, D) .* noise_sigma;
    end
    child(mask) = child(mask) + noise(mask);
    candidates(i, :) = min(max(child, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
[~, worst_order] = sort(fitness, 'descend');
replace_count = min(sample_count, numel(worst_order));
for i = 1:replace_count
    idx = worst_order(i);
    if values(i) < fitness(idx)
        population(idx, :) = candidates(i, :);
        fitness(idx) = values(i);
    end
end
end

function [mutant, F, CR] = epsde_strategy_mutant(population, combined, fitness, i, progress, options)
NP = size(population, 1);
weights = get_option(options, 'epsde_strategy_weights', [0.16 0.46 0.16 0.22]);
weights = weights(:)';
if numel(weights) ~= 4 || sum(weights) <= 0
    weights = [0.16 0.46 0.16 0.22];
end
weights = weights ./ sum(weights);
strategy = find(cumsum(weights) >= rand(), 1, 'first');
[F, CR] = epsde_parameter_pair(randi(6), progress);
p_num = max(2, min(NP, round((0.05 + 0.20 * (1 - 0.5 * progress)) * NP)));
switch strategy
    case 1
        ids = random_indices_except(NP, 3, i);
        mutant = population(ids(1), :) + F .* (population(ids(2), :) - population(ids(3), :));
    case 2
        pbest = randi(p_num);
        r1 = random_index_except(NP, i);
        r2 = randi(size(combined, 1));
        mutant = population(i, :) + F .* (population(pbest, :) - population(i, :)) + ...
            F .* (population(r1, :) - combined(r2, :));
    case 3
        ids = random_indices_except(NP, 2, i);
        pbest = randi(p_num);
        mutant = population(i, :) + rand() .* (population(pbest, :) - population(i, :)) + ...
            F .* (population(ids(1), :) - population(ids(2), :));
    otherwise
        ids = random_indices_except(NP, 4, i);
        mutant = population(1, :) + F .* (population(ids(1), :) - population(ids(2), :)) + ...
            0.5 .* F .* (population(ids(3), :) - population(ids(4), :));
        rank = rank_index(fitness, i);
        if rank > 0.65 * NP
            mutant = 0.75 .* mutant + 0.25 .* population(i, :);
        end
end
end

function [F, CR] = epsde_parameter_pair(param_id, progress)
F_pool = [1.0, 1.0, 0.8, 0.8, 0.6, 0.45];
CR_pool = [0.1, 0.9, 0.2, 0.8, 0.5, 0.95];
F = F_pool(param_id);
CR = CR_pool(param_id);
if progress > 0.65
    F = 0.82 .* F + 0.18 .* (0.35 + 0.25 .* rand());
    CR = 0.80 .* CR + 0.20 .* min(1, max(0, 0.75 + 0.15 .* randn()));
end
F = min(1, max(0.1, F + 0.05 .* randn()));
CR = min(1, max(0, CR + 0.05 .* randn()));
end

function ids = random_indices_except(NP, count, banned)
pool = setdiff(1:NP, banned, 'stable');
if isempty(pool)
    ids = ones(1, count);
elseif numel(pool) >= count
    perm = pool(randperm(numel(pool), count));
    ids = perm(:)';
else
    ids = pool(randi(numel(pool), 1, count));
end
end

function rank = rank_index(fitness, i)
[~, order] = sort(fitness);
positions = zeros(size(fitness));
positions(order) = 1:numel(fitness);
rank = positions(i);
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
