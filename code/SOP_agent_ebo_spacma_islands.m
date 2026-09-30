function result = SOP_agent_ebo_spacma_islands(problem, seed, options)
% Cooperative adaptive-DE and covariance-sampling islands.
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
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
NP_init = get_option(options, 'population_num', 14 * D);
NP_min = get_option(options, 'min_population_num', max(24, round(0.30 * D)));
lambda = get_option(options, 'scout_population_num', 4 + floor(3 * log(D)));
H = get_option(options, 'memory_size', 6);
p_rate = get_option(options, 'p_rate', 0.10);
archive_rate = get_option(options, 'archive_rate', 2.0);
exchange_period = get_option(options, 'exchange_period', 24);
restart_period = get_option(options, 'restart_period', 18);
immigrant_period = get_option(options, 'immigrant_period', 36);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP_init, D) .* span;
fitness = SOP_cec_evaluate(population, problem);
scout_population = lb + rand(lambda, D) .* span;
scout_fitness = SOP_cec_evaluate(scout_population, problem);
eval_count = NP_init + lambda;

[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
[scout_best, scout_idx] = min(scout_fitness);
if scout_best < best_raw
    best_raw = scout_best;
    best_position = scout_population(scout_idx, :);
end

memory_F = 0.7 * ones(H, 1);
memory_CR = 0.5 * ones(H, 1);
memory_shape = 0.10 * ones(H, 1);
memory_index = 1;
operator_probability = [0.5, 0.5];
archive = zeros(0, D);

mu = max(2, floor(lambda / 2));
weights = log(mu + 0.5) - log((1:mu)');
weights = weights / sum(weights);
mueff = sum(weights) ^ 2 / sum(weights .^ 2);
cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D);
cs = (mueff + 2) / (D + mueff + 5);
c1 = 2 / ((D + 1.3) ^ 2 + mueff);
cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ^ 2 + mueff));
damps = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cs;
chi_D = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ^ 2));
scout_norm = (scout_population - lb) ./ span;
[scout_fitness, scout_order] = sort(scout_fitness);
scout_norm = scout_norm(scout_order, :);
xmean = scout_norm(1:mu, :)' * weights;
sigma = get_option(options, 'sigma_init', 0.28);
pc = zeros(D, 1);
ps = zeros(D, 1);
covariance = eye(D);
basis = eye(D);
axis_scale = ones(D, 1);
inverse_sqrt_covariance = eye(D);
scout_stagnation = 0;
last_scout_best = scout_fitness(1);

curve = zeros(max(1, ceil(max_fes / max(1, NP_min + lambda))), 1);
iter = 0;
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    NP = size(population, 1);

    remaining = max_fes - eval_count;
    rows = min(NP, remaining);
    if rows > 0
        memory_rows = randi(H, NP, 1);
        sampled_F = positive_cauchy(memory_F(memory_rows), 0.1);
        sampled_CR = min(1, max(0, memory_CR(memory_rows) + 0.1 * randn(NP, 1)));
        sampled_shape = min(0.5, max(0, memory_shape(memory_rows) + 0.05 * randn(NP, 1)));
        use_operator_one = rand(NP, 1) < operator_probability(1);

        combined = [population; archive];
        [r1, r2, r3] = distinct_donor_rows(NP, size(combined, 1));
        p_count = max(2, min(NP, round(p_rate * NP)));
        pbest_rows = randi(p_count, NP, 1);
        mutant = zeros(NP, D);
        rows_one = find(use_operator_one);
        rows_two = find(~use_operator_one);
        mutant(rows_one, :) = population(rows_one, :) + sampled_F(rows_one) .* ...
            (population(r1(rows_one), :) - population(rows_one, :) + ...
            population(r3(rows_one), :) - combined(r2(rows_one), :));
        mutant(rows_two, :) = population(rows_two, :) + sampled_F(rows_two) .* ...
            (population(pbest_rows(rows_two), :) - population(rows_two, :) + ...
            population(r1(rows_two), :) - combined(r2(rows_two), :));
        mutant = repair_bounds(mutant, population, lb, ub);

        crossover_mask = shaped_crossover_mask(sampled_CR, sampled_shape, D);
        forced_columns = randi(D, NP, 1);
        crossover_mask(sub2ind([NP, D], (1:NP)', forced_columns)) = true;
        trial_population = population;
        trial_population(crossover_mask) = mutant(crossover_mask);
        trial_population = trial_population(1:rows, :);
        trial_fitness = SOP_cec_evaluate(trial_population, problem);
        eval_count = eval_count + rows;

        parent_fitness = fitness(1:rows);
        successful = trial_fitness < parent_fitness;
        improvements = parent_fitness(successful) - trial_fitness(successful);
        if any(successful)
            successful_rows = find(successful);
            archive = [archive; population(successful_rows, :)]; %#ok<AGROW>
            archive = unique(archive, 'rows', 'stable');
            success_weights = improvements / max(eps, sum(improvements));
            good_F = sampled_F(successful_rows);
            good_CR = sampled_CR(successful_rows);
            good_shape = sampled_shape(successful_rows);
            memory_F(memory_index) = weighted_lehmer(good_F, success_weights, memory_F(memory_index));
            memory_CR(memory_index) = weighted_lehmer(good_CR, success_weights, memory_CR(memory_index));
            memory_shape(memory_index) = weighted_lehmer(good_shape, success_weights, memory_shape(memory_index));

            gain_one = sum(improvements(use_operator_one(successful_rows)));
            gain_two = sum(improvements(~use_operator_one(successful_rows)));
            if gain_one + gain_two > 0
                target_probability = [gain_one, gain_two] / (gain_one + gain_two);
                operator_probability = 0.75 * operator_probability + 0.25 * target_probability;
                operator_probability = max(0.12, operator_probability);
                operator_probability = operator_probability / sum(operator_probability);
            end
            memory_index = mod(memory_index, H) + 1;
            population(successful_rows, :) = trial_population(successful_rows, :);
            fitness(successful_rows) = trial_fitness(successful_rows);
        end

        [generation_best, generation_idx] = min(trial_fitness);
        if generation_best < best_raw
            best_raw = generation_best;
            best_position = trial_population(generation_idx, :);
        end
    end

    target_NP = round(NP_init + (NP_min - NP_init) * eval_count / max_fes);
    target_NP = max(NP_min, target_NP);
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness);
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
    archive_limit = max(1, round(archive_rate * size(population, 1)));
    if size(archive, 1) > archive_limit
        archive = archive(randperm(size(archive, 1), archive_limit), :);
    end

    remaining = max_fes - eval_count;
    scout_rows = min(lambda, remaining);
    if scout_rows > 0
        normal_coordinates = randn(scout_rows, D) .* axis_scale';
        candidate_norm = xmean' + sigma .* (normal_coordinates * basis');
        candidate_norm = reflect_unit(candidate_norm);
        candidate_population = lb + candidate_norm .* span;
        candidate_fitness = SOP_cec_evaluate(candidate_population, problem);
        eval_count = eval_count + scout_rows;

        scout_norm = [scout_norm; candidate_norm];
        scout_fitness = [scout_fitness; candidate_fitness];
        [scout_fitness, scout_order] = sort(scout_fitness);
        scout_norm = scout_norm(scout_order, :);
        keep = min(lambda, size(scout_norm, 1));
        scout_norm = scout_norm(1:keep, :);
        scout_fitness = scout_fitness(1:keep);

        if scout_fitness(1) < last_scout_best - 1e-12 * max(1, abs(last_scout_best))
            scout_stagnation = 0;
            last_scout_best = scout_fitness(1);
        else
            scout_stagnation = scout_stagnation + 1;
        end
        if scout_fitness(1) < best_raw
            best_raw = scout_fitness(1);
            best_position = lb + scout_norm(1, :) .* span;
        end

        if keep >= mu
            xold = xmean;
            elite_norm = scout_norm(1:mu, :)';
            xmean = elite_norm * weights;
            normalized_step = (xmean - xold) / max(eps, sigma);
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * ...
                inverse_sqrt_covariance * normalized_step;
            hsig = norm(ps) / sqrt(max(eps, 1 - (1 - cs) ^ (2 * iter))) / chi_D < ...
                (1.4 + 2 / (D + 1));
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * normalized_step;
            elite_steps = (elite_norm - xold) / max(eps, sigma);
            covariance = (1 - c1 - cmu) * covariance + ...
                c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * covariance) + ...
                cmu * elite_steps * diag(weights) * elite_steps';
            covariance = (covariance + covariance') / 2;
            sigma = min(0.8, max(1e-5, sigma * exp((cs / damps) * (norm(ps) / chi_D - 1))));
        end
        if mod(iter, 4) == 0
            [basis, eigenvalues] = stable_eigendecomposition(covariance);
            axis_scale = sqrt(eigenvalues);
            inverse_sqrt_covariance = basis * diag(1 ./ axis_scale) * basis';
        end
    end

    if mod(iter, exchange_period) == 0
        [population, fitness, scout_norm, scout_fitness, xmean, covariance, sigma] = ...
            exchange_islands(population, fitness, scout_norm, scout_fitness, ...
            xmean, covariance, sigma, lb, span, weights);
    end

    if scout_stagnation >= restart_period
        [scout_norm, scout_fitness, xmean, covariance, sigma, added_evals] = ...
            restart_scout(problem, population, fitness, scout_norm, scout_fitness, ...
            lb, span, lambda, weights, max_fes - eval_count);
        eval_count = eval_count + added_evals;
        scout_stagnation = 0;
        last_scout_best = scout_fitness(1);
        pc(:) = 0;
        ps(:) = 0;
        [basis, eigenvalues] = stable_eigendecomposition(covariance);
        axis_scale = sqrt(eigenvalues);
        inverse_sqrt_covariance = basis * diag(1 ./ axis_scale) * basis';
    end

    if mod(iter, immigrant_period) == 0 && eval_count < max_fes
        immigrant_count = min(max(1, round(0.015 * size(population, 1))), max_fes - eval_count);
        immigrants = lb + rand(immigrant_count, D) .* span;
        immigrant_fitness = SOP_cec_evaluate(immigrants, problem);
        eval_count = eval_count + immigrant_count;
        [fitness, order] = sort(fitness);
        population = population(order, :);
        replace_rows = (size(population, 1) - immigrant_count + 1):size(population, 1);
        population(replace_rows, :) = immigrants;
        fitness(replace_rows) = immigrant_fitness;
        [immigrant_best, immigrant_idx] = min(immigrant_fitness);
        if immigrant_best < best_raw
            best_raw = immigrant_best;
            best_position = immigrants(immigrant_idx, :);
        end
    end

    curve(iter) = best_raw;
end

scout_population = lb + scout_norm .* span;
all_population = [population; scout_population];
all_fitness = [fitness; scout_fitness];
[all_fitness, order] = sort(all_fitness);
all_population = all_population(order, :);

result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve(1:iter), problem);
result.raw_convergence_curve = curve(1:iter);
result.runtime = toc(t_start);
result.iteration = iter;
result.population_num = NP_init + lambda;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf(['Adaptive dual-operator Differential Evolution island\n' ...
    'Success-history scale, crossover and dimension-shape adaptation\n' ...
    'SPACMA-style covariance sampling island\n' ...
    'Quality-diversity island exchange and stagnation restart']);
result.combination_number = 5;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = all_population;
result.final_fitness = all_fitness;

if verbose
    fprintf('EBO-SPACMA islands finished %s %dD F%d: best %.12g, time %.3f, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, result.runtime, eval_count);
end
end

function mask = shaped_crossover_mask(cr, shape, D)
NP = numel(cr);
mask = false(NP, D);
for row = 1:NP
    center = randi(D);
    distance = abs((1:D) - center);
    distance = min(distance, D - distance);
    probability = cr(row) .* exp(-shape(row) .* distance ./ max(1, D / 2));
    mask(row, :) = rand(1, D) < probability;
end
end

function [r1, r2, r3] = distinct_donor_rows(NP, combined_count)
targets = (1:NP)';
r1 = randi(NP, NP, 1);
r2 = randi(combined_count, NP, 1);
r3 = randi(NP, NP, 1);
bad = r1 == targets;
while any(bad)
    r1(bad) = randi(NP, nnz(bad), 1);
    bad = r1 == targets;
end
bad = r2 == targets | r2 == r1;
while any(bad)
    r2(bad) = randi(combined_count, nnz(bad), 1);
    bad = r2 == targets | r2 == r1;
end
bad = r3 == targets | r3 == r1 | r3 == r2;
while any(bad)
    r3(bad) = randi(NP, nnz(bad), 1);
    bad = r3 == targets | r3 == r1 | r3 == r2;
end
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

function values = positive_cauchy(centers, scale)
values = centers + scale .* tan(pi * (rand(size(centers)) - 0.5));
bad = values <= 0;
while any(bad)
    values(bad) = centers(bad) + scale .* tan(pi * (rand(nnz(bad), 1) - 0.5));
    bad = values <= 0;
end
values = min(values, 1);
end

function value = weighted_lehmer(samples, weights, fallback)
denominator = sum(weights .* samples);
if denominator <= eps
    value = fallback;
else
    value = sum(weights .* samples .^ 2) / denominator;
end
end

function normalized = reflect_unit(normalized)
normalized = mod(normalized, 2);
over = normalized > 1;
normalized(over) = 2 - normalized(over);
normalized = min(1, max(0, normalized));
end

function [basis, eigenvalues] = stable_eigendecomposition(covariance)
D = size(covariance, 1);
covariance = (covariance + covariance') / 2;
[basis, eigenvalues] = eig(covariance, 'vector');
if any(~isfinite(eigenvalues)) || any(~isfinite(basis), 'all')
    basis = eye(D);
    eigenvalues = ones(D, 1);
    return;
end
largest = max(eigenvalues);
floor_value = max(1e-12, largest * 1e-12);
eigenvalues = max(eigenvalues, floor_value);
end

function [population, fitness, scout_norm, scout_fitness, xmean, covariance, sigma] = ...
        exchange_islands(population, fitness, scout_norm, scout_fitness, ...
        xmean, covariance, sigma, lb, span, weights)
[fitness, order] = sort(fitness);
population = population(order, :);
[scout_fitness, scout_order] = sort(scout_fitness);
scout_norm = scout_norm(scout_order, :);
NP = size(population, 1);
lambda = size(scout_norm, 1);
de_diversity = mean(sqrt(sum(((population - population(1, :)) ./ span) .^ 2, 2)));
scout_diversity = mean(sqrt(sum((scout_norm - scout_norm(1, :)) .^ 2, 2)));
de_quality = 1 / max(eps, abs(fitness(1)));
scout_quality = 1 / max(eps, abs(scout_fitness(1)));
de_score = de_quality / (de_quality + scout_quality) + ...
    de_diversity / max(eps, de_diversity + scout_diversity);
scout_score = 2 - de_score;

if de_score >= scout_score
    elite_count = max(2, min(NP, round(0.15 * NP)));
    transfer_count = max(1, min([lambda, elite_count, round(0.35 * lambda)]));
    selected = randperm(elite_count, transfer_count);
    scout_norm(end - transfer_count + 1:end, :) = ...
        (population(selected, :) - lb) ./ span;
    scout_fitness(end - transfer_count + 1:end) = fitness(selected);
else
    transfer_count = max(1, min(lambda, round(0.08 * NP)));
    population(end - transfer_count + 1:end, :) = ...
        lb + scout_norm(1:transfer_count, :) .* span;
    fitness(end - transfer_count + 1:end) = scout_fitness(1:transfer_count);
end

[scout_fitness, scout_order] = sort(scout_fitness);
scout_norm = scout_norm(scout_order, :);
mu = numel(weights);
old_mean = xmean;
xmean = scout_norm(1:mu, :)' * weights;
centered = scout_norm(1:mu, :)' - xmean;
sample_covariance = centered * diag(weights) * centered';
covariance = 0.7 * covariance + 0.3 * sample_covariance / max(eps, sigma ^ 2);
covariance = (covariance + covariance') / 2;
sigma = min(0.5, max(0.02, 0.85 * sigma + 0.15 * norm(xmean - old_mean) / sqrt(size(population, 2))));
end

function [scout_norm, scout_fitness, xmean, covariance, sigma, eval_count] = ...
        restart_scout(problem, population, fitness, scout_norm, scout_fitness, ...
        lb, span, lambda, weights, remaining)
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = max(3, min(size(population, 1), round(0.12 * size(population, 1))));
center_row = randi(elite_count);
center = (population(center_row, :) - lb) ./ span;
restart_count = min(lambda - 1, remaining);
if restart_count > 0
    radius = 0.16 * (0.35 + rand());
    candidates = center + radius * randn(restart_count, size(population, 2));
    candidates = reflect_unit(candidates);
    candidate_population = lb + candidates .* span;
    candidate_fitness = SOP_cec_evaluate(candidate_population, problem);
else
    candidates = zeros(0, size(population, 2));
    candidate_fitness = zeros(0, 1);
end
scout_norm = [scout_norm(1, :); candidates];
scout_fitness = [scout_fitness(1); candidate_fitness];
[scout_fitness, order] = sort(scout_fitness);
scout_norm = scout_norm(order, :);
mu = min(numel(weights), size(scout_norm, 1));
local_weights = weights(1:mu);
local_weights = local_weights / sum(local_weights);
xmean = scout_norm(1:mu, :)' * local_weights;
centered = scout_norm(1:mu, :)' - xmean;
covariance = centered * diag(local_weights) * centered' + 1e-4 * eye(size(population, 2));
sigma = 0.18;
eval_count = restart_count;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
