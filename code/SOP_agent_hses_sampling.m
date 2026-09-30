function result = SOP_agent_hses_sampling(problem, seed, options)
% Hybrid Sampling Evolution Strategy (HS-ES/HSES) style sampler.
%
% The sampler combines multivariate covariance Gaussian sampling with
% univariate per-dimension Gaussian sampling, then updates both distributions
% from selected elites. It uses only objective rankings and bound repair.
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
lambda = max(12, get_option(options, 'population_num', max(28, round(0.30 * D))));
mu = max(3, min(lambda, round(get_option(options, 'elite_rate', 0.36) * lambda)));
weights = log(mu + 0.5) - log(1:mu);
weights = weights(:) ./ sum(weights);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

population = build_initial_population(problem, lambda, options);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
sigma_scale = get_option(options, 'sigma0', 0.0010);
min_sigma_scale = get_option(options, 'min_sigma', 1e-8);
max_sigma_scale = get_option(options, 'max_sigma', 0.030);
covariance_weight = get_option(options, 'covariance_sample_rate', 0.46);
univariate_weight = get_option(options, 'univariate_sample_rate', 0.38);
hybrid_weight = max(0, 1 - covariance_weight - univariate_weight);
curve = zeros(max(1, ceil(max_fes / lambda)), 1);
iter = 0;
stall = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness(:));
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_position = population(1, :);
        stall = 0;
    else
        stall = stall + 1;
    end
    usable_mu = min(mu, size(population, 1));
    w = weights(1:usable_mu);
    w = w ./ sum(w);
    elites = population(1:usable_mu, :);
    mean_x = w' * elites;
    mean_x = (1 - get_option(options, 'best_blend', 0.22)) .* mean_x + ...
        get_option(options, 'best_blend', 0.22) .* best_position;
    centered = elites - mean_x;
    diag_sigma = sqrt(max(w' * (centered .^ 2), 0)) + sigma_scale .* span;
    diag_sigma = min(max(diag_sigma, min_sigma_scale .* max(1, span)), max_sigma_scale .* span);
    cov_matrix = centered' * (centered .* w) + diag((get_option(options, 'cov_ridge', 0.30) .* diag_sigma) .^ 2 + 1e-18);
    cov_matrix = (cov_matrix + cov_matrix') ./ 2;
    [R, flag] = chol(cov_matrix, 'upper');
    if flag ~= 0
        R = diag(max(diag_sigma, 1e-12));
    end

    remaining = max_fes - eval_count;
    count = min(lambda, remaining);
    candidates = zeros(count, D);
    for i = 1:count
        mode_draw = rand();
        cov_sample = mean_x + get_option(options, 'cov_scale', 0.80) .* randn(1, D) * R;
        uni_sample = mean_x + get_option(options, 'uni_scale', 0.95) .* randn(1, D) .* diag_sigma;
        if mode_draw < covariance_weight
            child = cov_sample;
        elseif mode_draw < covariance_weight + univariate_weight
            child = uni_sample;
        else
            mask = rand(1, D) < get_option(options, 'hybrid_mask_rate', 0.42);
            if ~any(mask)
                mask(randi(D)) = true;
            end
            child = cov_sample;
            child(mask) = uni_sample(mask);
        end
        if rand() < get_option(options, 'elite_recomb_rate', 0.24)
            donor = elites(randi(usable_mu), :);
            mask = rand(1, D) < get_option(options, 'elite_recomb_mask_rate', 0.18);
            child(mask) = donor(mask);
        end
        if rand() < get_option(options, 'best_pull_rate', 0.24)
            pull = get_option(options, 'best_pull', 0.18) .* rand();
            child = child + pull .* (best_position - child);
        end
        candidates(i, :) = min(max(child, lb), ub);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    pool = [population; candidates];
    pool_fit = [fitness(:); values(:)];
    [pool_fit, order] = sort(pool_fit);
    pool = pool(order, :);
    keep = min(lambda, size(pool, 1));
    population = pool(1:keep, :);
    fitness = pool_fit(1:keep);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_position = population(1, :);
        sigma_scale = max(min_sigma_scale, get_option(options, 'success_sigma_decay', 0.985) * sigma_scale);
        stall = 0;
    else
        sigma_scale = max(min_sigma_scale, get_option(options, 'stall_sigma_decay', 0.94) * sigma_scale);
        stall = stall + 1;
    end
    if stall >= get_option(options, 'reset_stall', 18)
        sigma_scale = min(max_sigma_scale, max(sigma_scale, get_option(options, 'reset_sigma', 0.0030)));
        reset_count = min([lambda - 1, max_fes - eval_count, max(2, round(get_option(options, 'reset_rate', 0.22) * lambda))]);
        if reset_count <= 0
            curve(iter) = best_raw;
            continue;
        end
        rows = lambda - reset_count + 1:lambda;
        population(rows, :) = repmat(best_position, reset_count, 1) + ...
            randn(reset_count, D) .* repmat(sigma_scale .* span, reset_count, 1);
        population(rows, :) = min(max(population(rows, :), lb), ub);
        fitness(rows) = SOP_cec_evaluate(population(rows, :), problem);
        eval_count = eval_count + reset_count;
        stall = 0;
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

runtime = toc(t_start);
[final_fitness, final_order] = sort(fitness(:));
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = iter;
result.population_num = lambda;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Hybrid Sampling Evolution Strategy (HS-ES/HSES)\nCMA-ES covariance sampling and univariate Gaussian sampling');
result.combination_number = 1;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('HSES finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function population = build_initial_population(problem, lambda, options)
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
population = lb + rand(lambda, D) .* span;
initial_population = get_option(options, 'initial_population', []);
rows = 0;
if ~isempty(initial_population)
    rows = min(lambda, size(initial_population, 1));
    population(1:rows, :) = min(max(initial_population(1:rows, :), lb), ub);
end
initial_point = get_option(options, 'initial_point', []);
if ~isempty(initial_point)
    center = min(max(initial_point(:)', lb), ub);
    radius = get_option(options, 'initial_radius', []);
    if ~isempty(radius)
        if isscalar(radius)
            radius = radius .* span;
        else
            radius = reshape(radius, 1, []);
            if numel(radius) ~= D
                radius = mean(radius(:)) .* ones(1, D);
            end
        end
        population = repmat(center, lambda, 1) + randn(lambda, D) .* repmat(radius, lambda, 1);
        population = min(max(population, lb), ub);
        if get_option(options, 'preserve_initial_population_after_radius', true) && rows > 0
            population(1:rows, :) = min(max(initial_population(1:rows, :), lb), ub);
        end
    end
    population(1, :) = center;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
