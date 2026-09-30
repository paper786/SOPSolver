function result = SOP_agent_lshade_spacma_faithful(problem, seed, options)
% L-SHADE with semi-adaptive DE/CMA offspring generation.
%
% Each generation assigns individuals to either current-to-pbest/1 DE or
% covariance-distribution sampling. Success magnitudes adapt the class mix.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isempty(seed)
    if get_option(options, 'legacy_random_stream', false)
        rng(double(seed), 'v5uniform');
    else
        rng(double(seed), 'twister');
    end
else
    rng('shuffle');
end

t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP_init = get_option(options, 'population_num', 18 * D);
NP_min = get_option(options, 'min_population_num', 4);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
H = get_option(options, 'memory_size', 5);
p_rate = get_option(options, 'p_rate', 0.11);
archive_rate = get_option(options, 'archive_rate', 1.4);
class_learning_rate = get_option(options, 'class_learning_rate', 0.8);
eigen_rate = get_option(options, 'eigen_rate', 0);
neighbor_rate = get_option(options, 'neighbor_rate', 0.5);
ebo_operator_rate = get_option(options, 'ebo_operator_rate', 0);
shaped_crossover_rate = get_option(options, 'shaped_crossover_rate', 0);
shape_strength = get_option(options, 'shape_strength', 0.12);
partial_restart = get_option(options, 'partial_restart', false);
restart_start_progress = get_option(options, 'restart_start_progress', 0.28);
restart_stall_fraction = get_option(options, 'restart_stall_fraction', 0.10);
restart_population_rate = get_option(options, 'restart_population_rate', 0.34);
restart_global_rate = get_option(options, 'restart_global_rate', 0.30);
restart_radius = get_option(options, 'restart_radius', 0.075);
restart_max_count = get_option(options, 'restart_max_count', 2);
deduplicate_archive = get_option(options, 'deduplicate_archive', true);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP_init, D) .* span;
initial_population = get_option(options, 'initial_population', []);
initial_rows = 0;
if ~isempty(initial_population)
    initial_rows = min(NP_init, size(initial_population, 1));
    population(1:initial_rows, :) = min(max(initial_population(1:initial_rows, :), lb), ub);
end
initial_point = get_option(options, 'initial_point', []);
if ~isempty(initial_point)
    center = min(max(initial_point(:)', lb), ub);
    anchor_rate = get_option(options, 'initial_anchor_rate', 0.45);
    anchor_count = max(1, min(NP_init, round(anchor_rate * NP_init)));
    radius = get_option(options, 'initial_radius', 0.08) .* span;
    anchor_population = repmat(center, anchor_count, 1) + randn(anchor_count, D) .* radius;
    if get_option(options, 'initial_cauchy', true)
        use_cauchy = rand(anchor_count, 1) < 0.30;
        cauchy_noise = tan(pi * (rand(nnz(use_cauchy), D) - 0.5));
        cauchy_noise = min(max(cauchy_noise, -8), 8);
        anchor_population(use_cauchy, :) = repmat(center, nnz(use_cauchy), 1) + ...
            cauchy_noise .* radius;
    end
    anchor_population = min(max(anchor_population, lb), ub);
    population(1:anchor_count, :) = anchor_population;
    population(1, :) = center;
    if initial_rows > 0 && get_option(options, 'preserve_initial_population', true)
        preserve_rows = min(initial_rows, NP_init - anchor_count);
        if preserve_rows > 0
            population(anchor_count + (1:preserve_rows), :) = ...
                min(max(initial_population(1:preserve_rows, :), lb), ub);
        end
    end
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);

memory_F = 0.5 * ones(H, 1);
memory_CR = 0.5 * ones(H, 1);
initial_de_probability = get_option(options, 'initial_de_probability', 0.5);
memory_de_probability = initial_de_probability * ones(H, 1);
memory_index = 1;
archive = zeros(0, D);

mu = floor(NP_init / 2);
weights = recombination_weights(mu);
mueff = sum(weights) ^ 2 / sum(weights .^ 2);
cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D);
cs = (mueff + 2) / (D + mueff + 5);
c1 = 2 / ((D + 1.3) ^ 2 + mueff);
cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ^ 2 + mueff));
damps = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cs;
chi_D = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ^ 2));
xmean = rand(D, 1);
if ~isempty(initial_point)
    xmean = center';
end
sigma = get_option(options, 'sigma_init', 0.5);
pc = zeros(D, 1);
ps = zeros(D, 1);
basis = eye(D);
axis_scale = ones(D, 1);
covariance = eye(D);
inverse_sqrt_covariance = eye(D);
last_eigen_eval = 0;
cma_enabled = true;
last_improvement_eval = eval_count;
restart_count = 0;

curve = zeros(max(1, ceil(max_fes / NP_min)), 1);
iter = 0;
while eval_count < max_fes && size(population, 1) >= NP_min && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    NP = size(population, 1);
    [fitness, order] = sort(fitness);
    population = population(order, :);

    memory_rows = randi(H, NP, 1);
    sampled_CR = memory_CR(memory_rows) + 0.1 * randn(NP, 1);
    sampled_CR(memory_CR(memory_rows) < 0) = 0;
    sampled_CR = min(1, max(0, sampled_CR));
    if eval_count <= max_fes / 2
        sampled_F = 0.45 + 0.1 * rand(NP, 1);
    else
        sampled_F = positive_cauchy(memory_F(memory_rows), 0.1);
    end
    use_de = memory_de_probability(memory_rows) >= rand(NP, 1);
    if ~cma_enabled
        use_de(:) = true;
    end

    p_count = max(2, min(NP, round(p_rate * NP)));
    pbest_rows = randi(p_count, NP, 1);
    combined = [population; archive];
    if ebo_operator_rate > 0
        [r1, r2, r3] = distinct_donor_rows(NP, size(combined, 1));
    else
        [r1, r2] = distinct_donor_rows(NP, size(combined, 1));
        r3 = [];
    end
    mutant = zeros(NP, D);
    if any(use_de)
        rows_de = find(use_de);
        mutant(rows_de, :) = population(rows_de, :) + sampled_F(rows_de) .* ...
            (population(pbest_rows(rows_de), :) - population(rows_de, :) + ...
            population(r1(rows_de), :) - combined(r2(rows_de), :));
        if ebo_operator_rate > 0
            use_ebo = rand(numel(rows_de), 1) < ebo_operator_rate;
            ebo_rows = rows_de(use_ebo);
            if ~isempty(ebo_rows)
                mutant(ebo_rows, :) = population(ebo_rows, :) + sampled_F(ebo_rows) .* ...
                    (population(r1(ebo_rows), :) - population(ebo_rows, :) + ...
                    population(r3(ebo_rows), :) - combined(r2(ebo_rows), :));
            end
        end
    end
    if any(~use_de)
        rows_cma = find(~use_de);
        normal_coordinates = randn(numel(rows_cma), D) .* axis_scale';
        mutant(rows_cma, :) = xmean' + sigma .* (normal_coordinates * basis');
    end
    if any(~isfinite(mutant), 'all') || ~isreal(mutant)
        cma_enabled = false;
        use_de(:) = true;
        mutant = population + sampled_F .* ...
            (population(pbest_rows, :) - population + population(r1, :) - combined(r2, :));
    end
    mutant = repair_bounds(mutant, population, lb, ub);

    crossover_mask = rand(NP, D) < sampled_CR;
    if shaped_crossover_rate > 0
        shaped_rows = find(rand(NP, 1) < shaped_crossover_rate);
        for shaped_index = 1:numel(shaped_rows)
            row = shaped_rows(shaped_index);
            center = randi(D);
            distance = abs((1:D) - center);
            distance = min(distance, D - distance);
            shaped_probability = sampled_CR(row) .* ...
                exp(-shape_strength .* distance ./ max(1, D / 2));
            crossover_mask(row, :) = rand(1, D) < shaped_probability;
        end
    end
    forced_columns = randi(D, NP, 1);
    crossover_mask(sub2ind([NP, D], (1:NP)', forced_columns)) = true;
    if eigen_rate > 0 && rand() < eigen_rate
        crossover_basis = neighborhood_basis(population, best_position, neighbor_rate);
        parent_coordinates = population * crossover_basis;
        mutant_coordinates = mutant * crossover_basis;
        trial_coordinates = parent_coordinates;
        trial_coordinates(crossover_mask) = mutant_coordinates(crossover_mask);
        trial_population = trial_coordinates * crossover_basis';
    else
        trial_population = population;
        trial_population(crossover_mask) = mutant(crossover_mask);
    end

    remaining = max_fes - eval_count;
    rows = min(NP, remaining);
    trial_population = trial_population(1:rows, :);
    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);

    parent_fitness = fitness(1:rows);
    successful = trial_fitness < parent_fitness;
    improvements = parent_fitness(successful) - trial_fitness(successful);
    if any(successful)
            successful_rows = find(successful);
            archive = [archive; population(successful_rows, :)]; %#ok<AGROW>
            if deduplicate_archive
                archive = unique(archive, 'rows', 'stable');
            end
        weights_success = improvements ./ max(eps, sum(improvements));
        good_F = sampled_F(successful_rows);
        good_CR = sampled_CR(successful_rows);
        memory_F(memory_index) = weighted_lehmer(good_F, weights_success, memory_F(memory_index));
        if all(good_CR == 0) || memory_CR(memory_index) < 0
            memory_CR(memory_index) = -1;
        else
            memory_CR(memory_index) = weighted_lehmer( ...
                good_CR, weights_success, memory_CR(memory_index));
        end
        de_gain = sum(improvements(use_de(successful_rows)));
        cma_gain = sum(improvements(~use_de(successful_rows)));
        de_share = de_gain / max(eps, de_gain + cma_gain);
        updated_probability = class_learning_rate * memory_de_probability(memory_index) + ...
            (1 - class_learning_rate) * de_share;
        memory_de_probability(memory_index) = min(0.8, max(0.2, updated_probability));
        memory_index = memory_index + 1;
        if memory_index > H
            memory_index = 1;
        end

        population(successful_rows, :) = trial_population(successful_rows, :);
        fitness(successful_rows) = trial_fitness(successful_rows);
    end

    [generation_best, generation_best_idx] = min(trial_fitness);
    if generation_best < best_raw
        best_raw = generation_best;
        best_position = trial_population(generation_best_idx, :);
        last_improvement_eval = eval_count;
    end

    target_NP = round(NP_init + (NP_min - NP_init) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
    if target_NP < size(population, 1)
        [fitness, keep_order] = sort(fitness);
        population = population(keep_order(1:target_NP), :);
        fitness = fitness(1:target_NP);
        mu = max(2, floor(target_NP / 2));
        weights = recombination_weights(mu);
        mueff = sum(weights) ^ 2 / sum(weights .^ 2);
    end
    archive_limit = max(1, round(archive_rate * size(population, 1)));
    if size(archive, 1) > archive_limit
        archive = archive(randperm(size(archive, 1), archive_limit), :);
    end

    progress = eval_count / max_fes;
    stall_limit = max(NP_init, round(restart_stall_fraction * max_fes));
    if partial_restart && restart_count < restart_max_count && ...
            progress >= restart_start_progress && ...
            eval_count - last_improvement_eval >= stall_limit && ...
            eval_count < max_fes
        [population, fitness, extra_evals, restart_best, restart_position] = ...
            multibasin_restart(problem, population, fitness, archive, best_position, ...
            lb, ub, span, max_fes - eval_count, progress, restart_population_rate, ...
            restart_global_rate, restart_radius);
        eval_count = eval_count + extra_evals;
        if restart_best < best_raw
            best_raw = restart_best;
            best_position = restart_position;
        end
        restart_count = restart_count + 1;
        last_improvement_eval = eval_count;
        pc(:) = 0;
        ps(:) = 0;
        covariance = eye(D);
        basis = eye(D);
        axis_scale = ones(D, 1);
        inverse_sqrt_covariance = eye(D);
        sigma = get_option(options, 'sigma_init', 0.5);
    end

    if cma_enabled && size(population, 1) >= mu
        [fitness, cma_order] = sort(fitness);
        population = population(cma_order, :);
        xold = xmean;
        xmean = population(1:mu, :)' * weights;
        normalized_step = (xmean - xold) / max(eps, sigma);
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * ...
            inverse_sqrt_covariance * normalized_step;
        denominator = max(eps, 1 - (1 - cs) ^ (2 * eval_count / max(1, size(population, 1))));
        hsig = (sum(ps .^ 2) / denominator / D) < (2 + 4 / (D + 1));
        pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * normalized_step;
        centered_elites = (population(1:mu, :)' - xold) / max(eps, sigma);
        covariance = (1 - c1 - cmu) * covariance + ...
            c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * covariance) + ...
            cmu * centered_elites * diag(weights) * centered_elites';
        sigma = sigma * exp((cs / damps) * (norm(ps) / chi_D - 1));

        eigen_interval = size(population, 1) / max(eps, c1 + cmu) / D / 10;
        if eval_count - last_eigen_eval > eigen_interval
            last_eigen_eval = eval_count;
            covariance = (covariance + covariance') / 2;
            if any(~isfinite(covariance), 'all') || ~isreal(covariance)
                cma_enabled = false;
            else
                [basis, eigenvalues] = eig(covariance, 'vector');
                if any(~isfinite(eigenvalues)) || any(eigenvalues <= 0)
                    cma_enabled = false;
                else
                    axis_scale = sqrt(eigenvalues);
                    inverse_sqrt_covariance = basis * diag(1 ./ axis_scale) * basis';
                end
            end
        end
    end
    curve(iter) = best_raw;
end

runtime = toc(t_start);
curve = curve(1:iter);
[final_fitness, final_order] = sort(fitness);
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
result.algorithm_combination = sprintf(['Differential Evolution (DE)\n' ...
    'L-SHADE success-history adaptation and linear population reduction\n' ...
    'Semi-adaptive DE/CMA offspring allocation\n' ...
    'Online covariance evolution-path sampling']);
result.combination_number = 4;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = population(final_order, :);
result.final_fitness = final_fitness;

if verbose
    fprintf('L-SHADE-SPACMA finished %s %dD F%d: best %.12g, time %.3f, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function weights = recombination_weights(mu)
weights = log(mu + 0.5) - log((1:mu)');
weights = weights ./ sum(weights);
end

function [r1, r2, r3] = distinct_donor_rows(NP, combined_count)
targets = (1:NP)';
r1 = randi(NP, NP, 1);
bad = r1 == targets;
while any(bad)
    r1(bad) = randi(NP, nnz(bad), 1);
    bad = r1 == targets;
end
r2 = randi(combined_count, NP, 1);
bad = r2 == targets | r2 == r1;
while any(bad)
    r2(bad) = randi(combined_count, nnz(bad), 1);
    bad = r2 == targets | r2 == r1;
end
if nargout >= 3
    r3 = randi(NP, NP, 1);
    bad = r3 == targets | r3 == r1 | r3 == r2;
    while any(bad)
        r3(bad) = randi(NP, nnz(bad), 1);
        bad = r3 == targets | r3 == r1 | r3 == r2;
    end
else
    r3 = [];
end
end

function values = positive_cauchy(centers, scale)
values = centers + scale .* tan(pi * (rand(size(centers)) - 0.5));
bad = values <= 0;
while any(bad)
    values(bad) = centers(bad) + scale .* tan(pi * (rand(nnz(bad), 1) - 0.5));
    bad = values <= 0;
end
values = min(values, 1);
end

function repaired = repair_bounds(candidate, parent, lb, ub)
repaired = candidate;
low = repaired < lb;
high = repaired > ub;
lower_mid = 0.5 * (parent + lb);
upper_mid = 0.5 * (parent + ub);
repaired(low) = lower_mid(low);
repaired(high) = upper_mid(high);
repaired = min(max(repaired, lb), ub);
end

function basis = neighborhood_basis(population, best, neighbor_rate)
[NP, D] = size(population);
count = max(2, min(NP, round(neighbor_rate * NP)));
distances = sum((population - best) .^ 2, 2);
[~, order] = sort(distances);
neighbors = population(order(1:count), :);
centered = neighbors - mean(neighbors, 1);
covariance = (centered' * centered) ./ max(1, count - 1);
covariance = (covariance + covariance') ./ 2;
[basis, eigenvalues] = eig(covariance, 'vector');
if all(isfinite(eigenvalues))
    largest = max(eigenvalues);
    smallest = min(eigenvalues);
    if smallest <= 0 || largest > 1e20 * smallest
        covariance = covariance + max(eps, largest / 1e20 - smallest) * eye(D);
        [basis, eigenvalues] = eig(covariance, 'vector');
    end
end
if any(~isfinite(basis), 'all') || any(~isfinite(eigenvalues))
    basis = eye(D);
end
end

function value = weighted_lehmer(samples, weights, fallback)
denominator = sum(weights .* samples);
if denominator <= eps
    value = fallback;
else
    value = sum(weights .* (samples .^ 2)) / denominator;
end
end

function [population, fitness, eval_count, best_value, best_position] = ...
        multibasin_restart(problem, population, fitness, archive, incumbent, ...
        lb, ub, span, remaining, progress, population_rate, global_rate, radius_rate)
[fitness, order] = sort(fitness);
population = population(order, :);
NP = size(population, 1);
D = size(population, 2);
restart_rows = min(max(1, round(population_rate * NP)), remaining);
if restart_rows <= 0
    eval_count = 0;
    best_value = fitness(1);
    best_position = population(1, :);
    return;
end

candidates = zeros(restart_rows, D);
elite_count = max(2, min(NP, round(0.16 * NP)));
radius = radius_rate * max(0.18, 1 - progress) .* span;
for row = 1:restart_rows
    mode = rand();
    if mode < global_rate
        candidates(row, :) = lb + rand(1, D) .* span;
    elseif mode < global_rate + 0.36 && ~isempty(archive)
        anchor = archive(randi(size(archive, 1)), :);
        candidates(row, :) = anchor + randn(1, D) .* (0.55 .* radius);
    elseif mode < global_rate + 0.72
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -8), 8);
        candidates(row, :) = incumbent + noise .* radius;
    else
        anchor = population(randi(elite_count), :);
        candidates(row, :) = anchor + randn(1, D) .* radius;
    end
end
candidates = min(max(candidates, lb), ub);
candidate_fitness = SOP_cec_evaluate(candidates, problem);
replace_rows = (NP - restart_rows + 1):NP;
population(replace_rows, :) = candidates;
fitness(replace_rows) = candidate_fitness;
[best_value, best_idx] = min(fitness);
best_position = population(best_idx, :);
eval_count = restart_rows;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
