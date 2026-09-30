function result = SOP_agent_gsk_cov_refine(problem, seed, options)
% GSK followed by elite covariance local sampling.
%
% The first stage uses Gaining-Sharing Knowledge to locate a basin. The
% second stage keeps the final GSK population and samples around the best
% member along elite covariance directions with adaptive scale control.
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
population_num = get_option(options, 'population_num', 240);
base_fraction = get_option(options, 'base_fraction', 0.78);

gsk_options = options;
gsk_options.max_runtime_sec = base_fraction * max_runtime_sec;
gsk_options.max_iter = max(1, floor(base_fraction * get_option(options, 'max_iter', floor(max_fes / population_num))));
gsk_options.verbose = false;
gsk_result = SOP_agent_gsk(problem, double(seed), gsk_options);

D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = gsk_result.best_position;
best_raw = gsk_result.best_value;
eval_count = gsk_result.evaluation_count;
curve = gsk_result.convergence_curve(:);
if isfield(gsk_result, 'final_population') && ~isempty(gsk_result.final_population)
    population = gsk_result.final_population;
    fitness = gsk_result.final_fitness(:);
else
    population = repmat(best_x, population_num, 1) + randn(population_num, D) .* repmat(0.01 .* span, population_num, 1);
    population = min(max(population, lb), ub);
    fitness = SOP_cec_evaluate(population, problem);
    eval_count = eval_count + numel(fitness);
end

samples = get_option(options, 'refine_samples', max(80, round(0.55 * population_num)));
elite_rate = get_option(options, 'elite_rate', 0.20);
cov_scale = get_option(options, 'cov_scale', 0.18);
iso_scale = get_option(options, 'iso_scale', 0.0025);
min_cov_scale = get_option(options, 'min_cov_scale', 0.012);
min_iso_scale = get_option(options, 'min_iso_scale', 2e-5);
stall = 0;
refine_iter = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    count = min(samples, max_fes - eval_count);
    if count <= 0
        break;
    end
    [population, order] = sort_population(population, fitness);
    fitness = fitness(order);
    [R, center] = elite_covariance(population, span, elite_rate);
    candidates = make_candidates(best_x, center, population, R, lb, ub, span, count, cov_scale, iso_scale);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, trial_idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(trial_idx, :);
        cov_scale = max(0.92 * cov_scale, min_cov_scale);
        iso_scale = max(0.90 * iso_scale, min_iso_scale);
        stall = 0;
    else
        cov_scale = max(0.84 * cov_scale, min_cov_scale);
        iso_scale = max(0.82 * iso_scale, min_iso_scale);
        stall = stall + 1;
    end
    [~, worst_order] = sort(fitness, 'descend');
    replace_count = min(numel(worst_order), numel(values));
    for j = 1:replace_count
        if values(j) < fitness(worst_order(j))
            population(worst_order(j), :) = candidates(j, :);
            fitness(worst_order(j)) = values(j);
        end
    end
    if stall >= 25
        cov_scale = max(cov_scale, get_option(options, 'reset_cov_scale', 0.06) * (0.75 + 0.50 * rand()));
        iso_scale = max(iso_scale, get_option(options, 'reset_iso_scale', 0.0012) * (0.75 + 0.50 * rand()));
        stall = 0;
    end
    if mod(refine_iter, 10) == 0
        curve(end + 1, 1) = SOP_cec_record_value(best_raw, problem); %#ok<AGROW>
    end
end

result = gsk_result;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.convergence_curve = curve;
result.runtime = toc(t_start);
result.iteration = gsk_result.iteration + refine_iter;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK)\nElite covariance local stochastic sampling');
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function [population_sorted, order] = sort_population(population, fitness)
[~, order] = sort(fitness);
population_sorted = population(order, :);
end

function [R, center] = elite_covariance(population, span, elite_rate)
NP = size(population, 1);
elite_count = max(4, min(NP, round(elite_rate * NP)));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.0015 * span) .^ 2 + 1e-14);
cov_matrix = (cov_matrix + cov_matrix') / 2;
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), eps)));
end
end

function candidates = make_candidates(best_x, center, population, R, lb, ub, span, count, cov_scale, iso_scale)
D = numel(best_x);
cov_count = max(1, round(0.55 * count));
iso_count = max(1, round(0.25 * count));
diff_count = count - cov_count - iso_count;
candidates = zeros(count, D);
candidates(1:cov_count, :) = repmat(best_x, cov_count, 1) + cov_scale .* (randn(cov_count, D) * R);
rows = cov_count + (1:iso_count);
mask = rand(iso_count, D) < min(0.25, max(0.04, 10 / D));
empty = ~any(mask, 2);
if any(empty)
    cols = randi(D, sum(empty), 1);
    empty_rows = find(empty);
    for k = 1:numel(empty_rows)
        mask(empty_rows(k), cols(k)) = true;
    end
end
candidates(rows, :) = repmat(best_x, iso_count, 1) + mask .* randn(iso_count, D) .* repmat(iso_scale .* span, iso_count, 1);
if diff_count > 0
    rows = cov_count + iso_count + (1:diff_count);
    NP = size(population, 1);
    a = population(randi(NP, diff_count, 1), :);
    b = population(randi(NP, diff_count, 1), :);
    candidates(rows, :) = repmat(0.65 * best_x + 0.35 * center, diff_count, 1) + 0.20 * randn(diff_count, D) .* (a - b);
end
candidates = min(max(candidates, lb), ub);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
