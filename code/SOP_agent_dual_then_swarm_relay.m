function result = SOP_agent_dual_then_swarm_relay(problem, seed, options)
% Strong dual-basin DE/CMA search followed by a short metaheuristic relay.
%
% The first stage is an objective-only dual-basin crossover search. The
% second stage restarts another metaheuristic around the best point found by
% stage one, giving a different operator family a short chance to improve.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if isempty(seed)
    seed = randi(1000000);
end

t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
dual_runtime = min(max_runtime_sec, get_option(options, 'dual_runtime_sec', 245));
dual_fes = min(max_fes, get_option(options, 'dual_max_fes', max_fes));

dual_options = options;
dual_options.max_runtime_sec = dual_runtime;
dual_options.max_fes = dual_fes;
dual_options.verbose = false;
dual = SOP_agent_dual_basin_crossover_refine(problem, double(seed), dual_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - dual.evaluation_count);
relay_method = lower(string(get_option(options, 'relay_method', 'rime')));
relay = [];
if remaining_fes > 0 && remaining_time > 1
    relay_options = options;
    relay_options.population_num = get_option(options, 'relay_population_num', max(36, round(0.28 * get_option(options, 'population_num', 180))));
    relay_options.max_runtime_sec = remaining_time;
    relay_options.max_fes = remaining_fes;
    relay_options.initial_point = dual.best_position;
    relay_options.initial_radius = get_option(options, 'relay_radius', 0.0035);
    relay_options.initial_cauchy = get_option(options, 'relay_cauchy', true);
    relay_options.verbose = false;
    switch relay_method
        case "gsk"
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay_options.knowledge_rate = get_option(options, 'relay_knowledge_rate', 0.70);
            relay_options.knowledge_factor = get_option(options, 'relay_knowledge_factor', 0.34);
            relay = SOP_agent_gsk(problem, double(seed) + 131071, relay_options);
            relay_label = 'Gaining-Sharing Knowledge short relay';
        case "lshade"
            relay = SOP_agent_lshade(problem, double(seed) + 131071, relay_options);
            relay_label = 'L-SHADE short relay';
        case "lshade_cma"
            relay = SOP_agent_lshade_cma(problem, double(seed) + 131071, relay_options);
            relay_label = 'L-SHADE-CMA short relay';
        case "mts"
            relay_options.initial_step = get_option(options, 'relay_mts_step', get_option(options, 'relay_radius', 0.0035));
            relay_options.pattern_sigma = get_option(options, 'relay_pattern_sigma', 0.0008);
            relay_options.restart_step = get_option(options, 'relay_restart_step', 0.0012);
            relay_options.pattern_samples = get_option(options, 'relay_pattern_samples', max(96, 2 * relay_options.population_num));
            relay_options.pair_batch = get_option(options, 'relay_pair_batch', 20);
            relay = SOP_agent_center_mts_refine(problem, double(seed) + 131071, relay_options);
            relay_label = 'Multiple Trajectory Search short relay';
        case "eda"
            relay = eda_tail_refine(problem, dual, double(seed) + 131071, relay_options);
            relay_label = 'Elite EDA covariance/block-sampling short relay';
        case "island"
            if isfield(dual, 'final_population') && ~isempty(dual.final_population)
                relay_options.initial_population = dual.final_population;
                relay_options.preserve_initial_population_after_radius = true;
            end
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay_options.island_count = get_option(options, 'relay_island_count', 4);
            relay_options.block_rate_start = get_option(options, 'relay_block_rate_start', 0.14);
            relay_options.block_rate_end = get_option(options, 'relay_block_rate_end', 0.032);
            relay_options.block_de_weight = get_option(options, 'relay_block_de_weight', 0.10);
            relay_options.wide_block_probability = get_option(options, 'relay_wide_block_probability', 0.08);
            relay_options.migration_interval = get_option(options, 'relay_migration_interval', 12);
            relay_options.migration_sigma = get_option(options, 'relay_migration_sigma', 0.0012);
            relay_options.migration_block_rate = get_option(options, 'relay_migration_block_rate', 0.08);
            relay = SOP_agent_block_island_pool(problem, double(seed) + 131071, relay_options);
            relay_label = 'Cooperative GSK/RIME/TLBO/MPA-WSO block-island short relay';
        case "ccde"
            relay = ccde_tail_refine(problem, dual, double(seed) + 131071, relay_options);
            relay_label = 'Cooperative coevolution random-subspace DE short relay';
        case "linewell"
            relay = line_well_tail_refine(problem, dual, double(seed) + 131071, relay_options);
            relay_label = 'Elite-difference line-well lattice short relay';
        case "nrbo"
            relay_options.mode = 'nrbo';
            if isfield(dual, 'final_population') && ~isempty(dual.final_population)
                relay_options.initial_population = dual.final_population;
            end
            relay = SOP_agent_newton_guided_refine(problem, double(seed) + 131071, relay_options);
            relay_label = 'Newton-Raphson Search Rule guidance relay';
        case "ndo"
            relay_options.mode = 'ndo';
            if isfield(dual, 'final_population') && ~isempty(dual.final_population)
                relay_options.initial_population = dual.final_population;
            end
            relay = SOP_agent_newton_guided_refine(problem, double(seed) + 131071, relay_options);
            relay_label = 'Newton-downhill SSO/HGO guidance relay';
        case {"bfgs", "simplex", "powell"}
            relay_options.mode = char(relay_method);
            if isfield(dual, 'final_population') && ~isempty(dual.final_population)
                relay_options.initial_population = dual.final_population;
            end
            relay = SOP_agent_numeric_guided_refine(problem, double(seed) + 131071, relay_options);
            relay_label = sprintf('%s numerical-optimization guidance relay', upper(char(relay_method)));
        otherwise
            relay_options.method = char(relay_method);
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay = SOP_agent_literature_swarm(problem, double(seed) + 131071, relay_options);
            relay_label = sprintf('%s short literature-swarm relay', upper(char(relay_method)));
    end
else
    relay_label = 'No relay budget remaining';
end

result = dual;
if ~isempty(relay) && relay.record_value < result.record_value
    result = relay;
end
result.runtime = toc(t_start);
if isempty(relay)
    result.evaluation_count = dual.evaluation_count;
    result.iteration = dual.iteration;
    result.convergence_curve = dual.convergence_curve(:);
    result.raw_convergence_curve = raw_curve_for(dual);
else
    result.evaluation_count = dual.evaluation_count + relay.evaluation_count;
    result.iteration = dual.iteration + relay.iteration;
    result.convergence_curve = [dual.convergence_curve(:); relay.convergence_curve(:)];
    result.raw_convergence_curve = [raw_curve_for(dual); raw_curve_for(relay)];
end
result.algorithm_combination = sprintf('%s\n%s', dual.algorithm_combination, relay_label);
result.combination_number = 5;
result.agent_id = 'Agent2';
end

function result = ccde_tail_refine(problem, base, seed, options)
if ~isempty(seed)
    rng(double(seed), 'twister');
end
t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 1000);
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', 72);
best_x = base.best_position;
best_raw = base.best_value;
radius = get_option(options, 'ccde_radius', get_option(options, 'initial_radius', 0.0030)) .* span;
min_radius = get_option(options, 'ccde_min_radius', 1e-8) .* max(1, span);
reset_radius = get_option(options, 'ccde_reset_radius', 0.0045) .* span;
block_rate = get_option(options, 'ccde_block_rate', min(0.18, max(0.035, 6 / D)));
block_min = get_option(options, 'ccde_block_min', 4);
F0 = get_option(options, 'ccde_F', 0.48);
CR0 = get_option(options, 'ccde_CR', 0.34);

population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    rows = min(NP, size(base.final_population, 1));
    population(1:rows, :) = base.final_population(1:rows, :);
end
population(1, :) = best_x;
population = min(max(population, lb), ub);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[trial_raw, idx] = min(fitness);
if trial_raw < best_raw
    best_raw = trial_raw;
    best_x = population(idx, :);
end
curve = zeros(max(1, ceil(max_fes / max(1, NP))), 1);
iter = 0;
stall = 0;
archive = zeros(0, D);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_x = population(1, :);
    end
    combined = [population; archive];
    trial = population;
    for i = 1:NP
        mask = rand(1, D) < block_rate;
        if sum(mask) < block_min
            cols = randperm(D, min(D, block_min));
            mask(cols) = true;
        end
        p_count = max(2, round((0.08 + 0.08 * rand()) * NP));
        pbest = randi(p_count);
        r1 = random_index_except(NP, i);
        r2 = randi(size(combined, 1));
        F = min(0.88, max(0.12, F0 + 0.16 * tan(pi * (rand() - 0.5))));
        CR = min(1, max(0.02, CR0 + 0.14 * randn()));
        mutant = population(i, :);
        mutant(mask) = population(i, mask) + ...
            F .* (population(pbest, mask) - population(i, mask)) + ...
            F .* (population(r1, mask) - combined(r2, mask));
        cross = (rand(1, D) < CR) & mask;
        if ~any(cross)
            mask_idx = find(mask);
            cross(mask_idx(randi(numel(mask_idx)))) = true;
        end
        candidate = population(i, :);
        candidate(cross) = mutant(cross);
        if rand() < get_option(options, 'ccde_noise_rate', 0.16)
            candidate(mask) = candidate(mask) + randn(1, sum(mask)) .* radius(mask);
        end
        low = candidate < lb;
        high = candidate > ub;
        candidate(low) = 0.5 * (population(i, low) + lb(low));
        candidate(high) = 0.5 * (population(i, high) + ub(high));
        trial(i, :) = min(max(candidate, lb), ub);
    end
    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
    end
    values = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    rows = numel(values);
    improved = values(:) <= fitness(1:rows);
    if any(improved)
        archive = [archive; population(improved, :)]; %#ok<AGROW>
        if size(archive, 1) > 2 * NP
            archive = archive(randperm(size(archive, 1), 2 * NP), :);
        end
        population(improved, :) = trial(improved, :);
        fitness(improved) = values(improved);
    end
    [iter_best, iter_idx] = min(fitness);
    if iter_best < best_raw
        best_raw = iter_best;
        best_x = population(iter_idx, :);
        radius = max(0.985 .* radius, min_radius);
        stall = 0;
    else
        radius = max(0.94 .* radius, min_radius);
        stall = stall + 1;
    end
    if stall >= 28
        radius = max(radius, reset_radius .* (0.70 + 0.60 * rand(1, D)));
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = iter;
result.evaluation_count = eval_count;
result.population_num = NP;
result.final_population = population;
result.final_fitness = fitness;
result.algorithm_combination = 'Cooperative coevolution random-subspace DE refinement';
result.combination_number = 5;
result.agent_id = 'Agent2';
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
elseif isfield(result, 'convergence_curve') && ~isempty(result.convergence_curve)
    curve = result.convergence_curve(:);
else
    curve = result.record_value;
end
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function result = line_well_tail_refine(problem, base, seed, options)
if ~isempty(seed)
    rng(double(seed), 'twister');
end
t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 1000);
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = base.best_position;
best_raw = base.best_value;
if isfield(base, 'final_population') && ~isempty(base.final_population)
    population = base.final_population;
else
    population = repmat(best_x, max(24, get_option(options, 'population_num', 64)), 1);
end
population = min(max(population(:, 1:D), lb), ub);
if isfield(base, 'final_fitness') && ~isempty(base.final_fitness) && numel(base.final_fitness) == size(population, 1)
    fitness = base.final_fitness(:);
else
    fitness = SOP_cec_evaluate(population, problem);
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = min(size(population, 1), get_option(options, 'linewell_elite_count', max(10, round(0.22 * size(population, 1)))));
batch = get_option(options, 'linewell_batch', max(72, round(0.40 * get_option(options, 'population_num', size(population, 1)))));
step_scales = get_option(options, 'linewell_step_scales', [0.0018, 0.0035, 0.0065, 0.0100]);
block_rate = get_option(options, 'linewell_block_rate', max(0.035, 5 / D));
shrink = get_option(options, 'linewell_shrink', 0.72);
expand = get_option(options, 'linewell_expand', 1.08);
eval_count = 0;
iter = 0;
stall = 0;
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = linewell_candidates(population, best_x, lb, ub, span, elite_count, count, step_scales, block_rate, options);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    pool = [population; candidates];
    pool_fit = [fitness; values(:)];
    [pool_fit, order] = sort(pool_fit);
    keep = min(size(population, 1), size(pool, 1));
    population = pool(order(1:keep), :);
    fitness = pool_fit(1:keep);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        step_scales = max(step_scales .* shrink, get_option(options, 'linewell_min_step_scale', 0.00010));
        stall = 0;
    else
        stall = stall + 1;
        if stall >= get_option(options, 'linewell_stall_expand_iter', 4)
            step_scales = min(step_scales .* expand, get_option(options, 'linewell_max_step_scale', 0.018));
            stall = 0;
        end
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = iter;
result.evaluation_count = eval_count;
result.population_num = get_option(options, 'population_num', size(population, 1));
result.algorithm_combination = sprintf('Elite-difference line-well lattice refinement');
result.combination_number = 2;
result.agent_id = 'Agent2';
end

function candidates = linewell_candidates(population, best_x, lb, ub, span, elite_count, count, step_scales, block_rate, options)
D = numel(best_x);
elites = population(1:elite_count, :);
center = mean(elites, 1);
centered = elites - center;
directions = zeros(max(8, count), D);
dir_count = 0;
if elite_count >= 3
    [~, ~, V] = svd(centered, 'econ');
    pc_count = min(size(V, 2), get_option(options, 'linewell_pc_count', 5));
    for j = 1:pc_count
        dir_count = dir_count + 1;
        directions(dir_count, :) = V(:, j)';
    end
end
for j = 1:max(6, ceil(0.35 * count))
    a = randi(elite_count);
    b = randi(elite_count);
    d = elites(a, :) - elites(b, :);
    if norm(d) < eps
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        d = zeros(1, D);
        d(mask) = randn(1, sum(mask)) .* span(mask);
    end
    dir_count = dir_count + 1;
    directions(dir_count, :) = d;
end
directions = directions(1:dir_count, :);
candidates = repmat(best_x, count, 1);
for i = 1:count
    d = directions(randi(dir_count), :);
    if rand() < get_option(options, 'linewell_sparse_rate', 0.44)
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        d(~mask) = 0;
    end
    d = d ./ max(norm(d), eps);
    step = step_scales(randi(numel(step_scales))) .* norm(span);
    if rand() < 0.5
        step = -step;
    end
    if rand() < get_option(options, 'linewell_halfstep_rate', 0.35)
        step = step .* (0.35 + 0.45 * rand());
    end
    child = best_x + step .* d;
    if rand() < get_option(options, 'linewell_elite_blend_rate', 0.24)
        donor = elites(randi(elite_count), :);
        blend = get_option(options, 'linewell_elite_blend', 0.18) * rand();
        child = (1 - blend) .* child + blend .* donor;
    end
    candidates(i, :) = min(max(child, lb), ub);
end
end

function result = eda_tail_refine(problem, base, seed, options)
if ~isempty(seed)
    rng(double(seed), 'twister');
end
t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 1000);
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
D = problem.dimension;
best_x = base.best_position;
best_raw = base.best_value;
if isfield(base, 'final_population') && ~isempty(base.final_population)
    population = base.final_population;
else
    population = repmat(best_x, max(20, get_option(options, 'population_num', 60)), 1);
end
if isfield(base, 'final_fitness') && ~isempty(base.final_fitness) && numel(base.final_fitness) == size(population, 1)
    fitness = base.final_fitness(:);
else
    fitness = SOP_cec_evaluate(population, problem);
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = min(size(population, 1), get_option(options, 'eda_elite_count', max(12, round(0.18 * size(population, 1)))));
batch = get_option(options, 'eda_batch', max(96, round(0.55 * get_option(options, 'population_num', size(population, 1)))));
cov_scale = get_option(options, 'eda_cov_scale', 0.055);
iso_scale = get_option(options, 'eda_iso_scale', 0.00045) .* span;
block_rate = get_option(options, 'eda_block_rate', max(0.030, 4 / D));
min_iso = get_option(options, 'eda_min_iso_scale', 1e-8) .* max(1, span);
reset_cov_scale = get_option(options, 'eda_reset_cov_scale', 0.018);
reset_iso = get_option(options, 'eda_reset_iso_scale', 0.00016) .* span;
eval_count = 0;
iter = 0;
stall = 0;
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = sample_eda_candidates(population, best_x, lb, ub, span, elite_count, count, cov_scale, iso_scale, block_rate);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        population = [best_x; population(1:end-1, :)]; %#ok<AGROW>
        fitness = [best_raw; fitness(1:end-1)]; %#ok<AGROW>
        cov_scale = max(0.72 * cov_scale, 0.0025);
        iso_scale = max(0.86 .* iso_scale, min_iso);
        stall = 0;
    else
        if get_option(options, 'eda_accept_worst', false)
            [~, worst_order] = sort(fitness, 'descend');
            replace_count = min(numel(values), numel(worst_order));
            accepted = false;
            for k = 1:replace_count
                row = worst_order(k);
                if values(k) < fitness(row)
                    population(row, :) = candidates(k, :);
                    fitness(row) = values(k);
                    accepted = true;
                end
            end
            if accepted
                [fitness, order] = sort(fitness);
                population = population(order, :);
                elite_count = min(size(population, 1), elite_count);
            end
        end
        cov_scale = max(0.80 * cov_scale, 0.0015);
        iso_scale = max(0.74 .* iso_scale, min_iso);
        stall = stall + 1;
    end
    if stall >= 10
        cov_scale = max(cov_scale, reset_cov_scale);
        iso_scale = max(iso_scale, reset_iso);
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = iter;
result.evaluation_count = eval_count;
result.population_num = get_option(options, 'population_num', size(population, 1));
result.algorithm_combination = sprintf('Elite EDA covariance/block-sampling refinement');
result.combination_number = 2;
result.agent_id = 'Agent2';
end

function candidates = sample_eda_candidates(population, best_x, lb, ub, span, elite_count, count, cov_scale, iso_scale, block_rate)
D = numel(best_x);
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
center = 0.70 * best_x + 0.30 * center;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((iso_scale .^ 2) + 1e-16);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), 1e-16)));
end
candidates = repmat(best_x, count, 1);
for i = 1:count
    if rand() < 0.62
        x = center + cov_scale .* randn(1, D) * R;
    else
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        if rand() < 0.45
            noise = tan(pi * (rand(1, D) - 0.5));
            noise = min(max(noise, -7), 7);
        else
            noise = randn(1, D);
        end
        x = best_x;
        x(mask) = x(mask) + noise(mask) .* iso_scale(mask);
    end
    candidates(i, :) = min(max(x, lb), ub);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
