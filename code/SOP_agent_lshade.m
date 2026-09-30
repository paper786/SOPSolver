function result = SOP_agent_lshade(problem, seed, options)
% L-SHADE style success-history Differential Evolution.
%
% Literature basis: Differential Evolution and success-history adaptive DE
% with linear population size reduction. The implementation is original and
% only uses the public benchmark objective.
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
H = get_option(options, 'memory_size', 8);
p_rate = get_option(options, 'p_rate', 0.11);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

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
archive = zeros(0, D);
archive_age = zeros(0, 1);
mu_F = 0.5 * ones(1, H);
mu_CR = 0.5 * ones(1, H);
memory_index = 1;
curve = zeros(max(1, ceil(max_fes / max(1, NP_min))), 1);
iter = 0;
stall_count = 0;
soft_restart_enabled = get_option(options, 'inloop_soft_restart', false);
eig_gate_enabled = get_option(options, 'eig_success_gate', false);
eig_gate_rate = get_option(options, 'eig_gate_rate', 0.065);
eig_rate_min = get_option(options, 'eig_rate_min', 0.006);
eig_rate_max = get_option(options, 'eig_rate_max', max(0.08, eig_gate_rate));
eig_interval = get_option(options, 'eig_interval', 14);
eig_elite_rate = get_option(options, 'eig_elite_rate', 0.22);
eig_basis = eye(D);
eig_center = mean(population, 1);
stratified_archive = get_option(options, 'stratified_archive', false);
stratified_archive_rate = get_option(options, 'stratified_archive_rate', 0.58);
dual_island_topology = get_option(options, 'dual_island_topology', false);
island_migration_rate = get_option(options, 'island_migration_rate', 0.08);
island_migration_interval = get_option(options, 'island_migration_interval', 22);

while eval_count < max_fes && size(population, 1) >= 4 && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if eig_gate_enabled && (iter == 1 || mod(iter, eig_interval) == 0)
        [eig_basis, eig_center] = elite_eigen_basis(population, span, eig_elite_rate);
    end
    combined = [population; archive];
    pool_size = size(combined, 1);
    p_num = max(2, round(p_rate * NP));
    trial_population = population;
    sampled_F = zeros(NP, 1);
    sampled_CR = zeros(NP, 1);
    used_eig = false(NP, 1);
    island_a = 1:2:NP;
    island_b = 2:2:NP;

    for i = 1:NP
        mem = randi(H);
        F = sample_F(mu_F(mem));
        CR = min(1, max(0, mu_CR(mem) + 0.1 * randn()));
        sampled_F(i) = F;
        sampled_CR(i) = CR;
        if dual_island_topology
            if mod(i, 2) == 1
                island_indices = island_a;
                other_indices = island_b;
            else
                island_indices = island_b;
                other_indices = island_a;
            end
            local_p_num = max(1, min(numel(island_indices), round(p_rate * numel(island_indices))));
            migrate = mod(iter, island_migration_interval) == 0 || rand() < island_migration_rate;
            if migrate && ~isempty(other_indices)
                other_p_num = max(1, min(numel(other_indices), round(p_rate * numel(other_indices))));
                pbest = other_indices(randi(other_p_num));
                r1 = other_indices(randi(numel(other_indices)));
            else
                pbest = island_indices(randi(local_p_num));
                r1 = sample_index_excluding(island_indices, i);
            end
        else
            pbest = randi(p_num);
            r1 = random_index_except(NP, i);
        end
        if stratified_archive && ~isempty(archive) && rand() < stratified_archive_rate
            r2 = NP + stratified_archive_index(archive, archive_age, best_position, eval_count / max_fes, options);
        else
            r2 = random_pool_index(pool_size, [i r1]);
        end
        mutant = population(i, :) ...
            + F .* (population(pbest, :) - population(i, :)) ...
            + F .* (population(r1, :) - combined(r2, :));
        if eig_gate_enabled && rand() < eig_gate_rate
            used_eig(i) = true;
            trial = eigen_crossover(population(i, :), mutant, CR, eig_basis, eig_center);
        else
            trial = binomial_crossover(population(i, :), mutant, CR);
        end
        trial_population(i, :) = repair_bounds(trial, population(i, :), lb, ub);
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
            archive_age = [archive_age; iter]; %#ok<AGROW>
            success_delta(end + 1, 1) = max(0, fitness(i) - trial_fitness(i)); %#ok<AGROW>
            success_F(end + 1, 1) = sampled_F(i); %#ok<AGROW>
            success_CR(end + 1, 1) = sampled_CR(i); %#ok<AGROW>
            population(i, :) = trial_population(i, :);
            fitness(i) = trial_fitness(i);
        end
    end
    if eig_gate_enabled && any(used_eig(1:rows))
        eig_rows = used_eig(1:rows);
        eig_success = nnz((trial_fitness <= fitness(1:rows)) & eig_rows);
        eig_trials = nnz(eig_rows);
        target_rate = eig_gate_rate * (0.35 + 1.75 * eig_success / max(1, eig_trials));
        if eig_success == 0
            target_rate = 0.55 * eig_gate_rate;
        end
        eig_gate_rate = 0.78 * eig_gate_rate + 0.22 * target_rate;
        eig_gate_rate = min(eig_rate_max, max(eig_rate_min, eig_gate_rate));
    end
    if size(archive, 1) > NP
        keep = randperm(size(archive, 1), NP);
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
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
        stall_count = 0;
    else
        stall_count = stall_count + 1;
    end
    if soft_restart_enabled && stall_count >= get_option(options, 'soft_restart_stall_iter', 28) && eval_count < max_fes
        [population, fitness, extra_eval] = soft_restart_worst_step(population, fitness, best_position, problem, ...
            lb, ub, span, eval_count, max_fes, options);
        eval_count = eval_count + extra_eval;
        [current_best, current_idx] = min(fitness);
        if current_best < best_raw
            best_raw = current_best;
            best_position = population(current_idx, :);
        end
        stall_count = 0;
    end
    curve(iter) = best_raw;

    target_NP = round(NP_init - (NP_init - NP_min) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
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
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation');
if soft_restart_enabled
    result.algorithm_combination = sprintf('%s\nIn-loop worst-population soft restart around current best', result.algorithm_combination);
end
if eig_gate_enabled
    result.algorithm_combination = sprintf('%s\nSuccess-gated elite eigen-coordinate crossover', ...
        result.algorithm_combination);
end
if stratified_archive
    result.algorithm_combination = sprintf('%s\nStratified recent, stale, and elite-near archive donors', ...
        result.algorithm_combination);
end
if dual_island_topology
    result.algorithm_combination = sprintf('%s\nInterleaved dual-island donor topology with sparse migration', ...
        result.algorithm_combination);
end
result.combination_number = 1;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('L-SHADE finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
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

function idx = sample_index_excluding(pool, banned)
available = pool(pool ~= banned);
if isempty(available)
    idx = banned;
else
    idx = available(randi(numel(available)));
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
recent_rate = get_option(options, 'stratified_recent_rate', 0.40) * max(0, 1 - 0.50 * progress);
stale_rate = get_option(options, 'stratified_stale_rate', 0.26) * (0.35 + 0.65 * progress);
draw = rand();
if draw < recent_rate
    [~, order] = sort(archive_age, 'descend');
    pool_count = max(1, round(get_option(options, 'stratified_recent_pool_rate', 0.30) * count));
elseif draw < recent_rate + stale_rate
    [~, order] = sort(archive_age, 'ascend');
    pool_count = max(1, round(get_option(options, 'stratified_stale_pool_rate', 0.30) * count));
else
    distance = sum((archive - best) .^ 2, 2);
    [~, order] = sort(distance, 'ascend');
    pool_count = max(1, round(get_option(options, 'stratified_elite_near_pool_rate', 0.26) * count));
end
pool_count = min(count, pool_count);
idx = order(randi(pool_count));
end

function trial = binomial_crossover(parent, mutant, CR)
D = numel(parent);
mask = rand(1, D) <= CR;
mask(randi(D)) = true;
trial = parent;
trial(mask) = mutant(mask);
end

function [basis, center] = elite_eigen_basis(population, span, elite_rate)
elite_count = max(4, min(size(population, 1), round(elite_rate * size(population, 1))));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.002 * span) .^ 2 + 1e-14);
cov_matrix = 0.5 * (cov_matrix + cov_matrix');
[basis, values] = eig(cov_matrix, 'vector');
if ~isvector(values)
    values = diag(values);
end
[~, order] = sort(values, 'descend');
basis = real(basis(:, order));
if any(~isfinite(basis(:)))
    basis = eye(size(population, 2));
end
end

function trial = eigen_crossover(parent, mutant, CR, basis, center)
parent_z = (parent - center) * basis;
mutant_z = (mutant - center) * basis;
mask = rand(1, numel(parent)) <= CR;
mask(randi(numel(parent))) = true;
trial_z = parent_z;
trial_z(mask) = mutant_z(mask);
trial = trial_z * basis' + center;
end

function trial = repair_bounds(trial, parent, lb, ub)
low = trial < lb;
high = trial > ub;
trial(low) = 0.5 * (parent(low) + lb(low));
trial(high) = 0.5 * (parent(high) + ub(high));
trial = min(max(trial, lb), ub);
end

function [population, fitness, eval_count] = soft_restart_worst_step(population, fitness, best, problem, lb, ub, span, eval_count_so_far, max_fes, options)
eval_count = 0;
NP = size(population, 1);
D = size(population, 2);
remaining = max_fes - eval_count_so_far;
if remaining <= 0 || NP < 6
    return;
end
replace_count = max(2, round(get_option(options, 'soft_restart_worst_rate', 0.18) * NP));
replace_count = min([replace_count, remaining, max(1, NP - 2)]);
[~, worst_order] = sort(fitness, 'descend');
rows = worst_order(1:replace_count);
progress = eval_count_so_far / max_fes;
base_radius = get_option(options, 'soft_restart_radius', 0.0060) .* (1 - 0.55 * progress) .* span;
reset_radius = get_option(options, 'soft_restart_reset_radius', 0.014) .* span;
radius = max(base_radius, get_option(options, 'soft_restart_min_radius', 0.00012) .* span);
if rand() < get_option(options, 'soft_restart_wide_rate', 0.18)
    radius = max(radius, reset_radius .* (0.45 + 0.65 * rand(1, D)));
end
candidates = repmat(best, replace_count, 1) + randn(replace_count, D) .* repmat(radius, replace_count, 1);
if get_option(options, 'soft_restart_cauchy_rate', 0.35) > 0
    cauchy_mask = rand(replace_count, D) < get_option(options, 'soft_restart_cauchy_rate', 0.35);
    cauchy_noise = tan(pi * (rand(replace_count, D) - 0.5));
    cauchy_noise = min(max(cauchy_noise, -7), 7);
    radius_matrix = repmat(radius, replace_count, 1);
    best_matrix = repmat(best, replace_count, 1);
    candidates(cauchy_mask) = best_matrix(cauchy_mask) + cauchy_noise(cauchy_mask) .* radius_matrix(cauchy_mask);
end
candidates = min(max(candidates, lb), ub);
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
for k = 1:replace_count
    row = rows(k);
    population(row, :) = candidates(k, :);
    fitness(row) = values(k);
end
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
