function result = SOP_agent_cma_es(problem, seed, options)
% Restartable CMA-ES style covariance-matrix evolutionary strategy.
%
% This is an evolutionary metaheuristic candidate used as a distribution
% estimation alternative to DE/GSK. It learns a Gaussian search distribution
% only from objective rankings and respects the same benchmark interface.
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
lambda = get_option(options, 'population_num', 4 + floor(3 * log(D)) + 32);
lambda = max(12, lambda);
mu = floor(lambda / 2);
weights = log(mu + 0.5) - log(1:mu);
weights = weights(:) / sum(weights);
mueff = 1 / sum(weights .^ 2);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
eig_interval = get_option(options, 'eig_interval', max(4, floor(D / 8)));
restart_limit = get_option(options, 'restart_limit', 2);
verbose = get_option(options, 'verbose', false);

cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D);
cs = (mueff + 2) / (D + mueff + 5);
c1 = 2 / ((D + 1.3) ^ 2 + mueff);
cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ^ 2 + mueff));
damps = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cs;
chiN = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ^ 2));

initial_point = get_option(options, 'initial_point', []);
if isempty(initial_point)
    init_pop = lb + rand(lambda, D) .* span;
    init_fit = SOP_cec_evaluate(init_pop, problem);
    eval_count = numel(init_fit);
    [best_raw, best_idx] = min(init_fit);
    mean_x = init_pop(best_idx, :);
    [last_fitness, last_order] = sort(init_fit(:));
    last_population = init_pop(last_order, :);
else
    mean_x = min(max(initial_point(:)', lb), ub);
    best_raw = SOP_cec_evaluate(mean_x, problem);
    eval_count = 1;
    last_population = mean_x;
    last_fitness = best_raw;
end
best_position = mean_x;
sigma = get_option(options, 'sigma0', 0.22) * mean(span);
min_sigma = get_option(options, 'min_sigma', 1e-9) * mean(span);
C = initial_covariance_matrix(options, mean_x, sigma, lb, ub, D);
[B, evals] = eig(C, 'vector');
evals = max(real(evals), 1e-20);
[evals, eig_order] = sort(evals, 'descend');
B = real(B(:, eig_order));
diagD = sqrt(evals(:));
pc = zeros(1, D);
ps = zeros(1, D);
curve = zeros(max(1, ceil(max_fes / lambda)), 1);
iter = 0;
stall = 0;
restart_count = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    arz = randn(lambda, D);
    ary = arz * (B * diag(diagD))';
    candidates = repmat(mean_x, lambda, 1) + sigma .* ary;
    candidates = min(max(candidates, lb), ub);
    remaining = max_fes - eval_count;
    if remaining < lambda
        candidates = candidates(1:remaining, :);
        ary = ary(1:remaining, :);
        arz = arz(1:remaining, :);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [values, order] = sort(values);
    candidates = candidates(order, :);
    ary = ary(order, :);
    arz = arz(order, :);
    last_population = candidates;
    last_fitness = values(:);
    if values(1) < best_raw
        best_raw = values(1);
        best_position = candidates(1, :);
        stall = 0;
    else
        stall = stall + 1;
    end

    usable_mu = min(mu, size(candidates, 1));
    w = weights(1:usable_mu);
    w = w / sum(w);
    old_mean = mean_x;
    mean_x = w' * candidates(1:usable_mu, :);
    y_w = (mean_x - old_mean) / max(sigma, eps);
    z_w = w' * arz(1:usable_mu, :);
    ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) .* (z_w * B');
    norm_ps = norm(ps);
    hsig = norm_ps / sqrt(1 - (1 - cs) ^ (2 * iter)) / chiN < (1.4 + 2 / (D + 1));
    pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) .* y_w;
    artmp = ary(1:usable_mu, :);
    C = (1 - c1 - cmu) * C + c1 * (pc' * pc + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp' * diag(w) * artmp);
    C = (C + C') / 2;
    sigma = sigma * exp((cs / damps) * (norm_ps / chiN - 1));
    if mod(iter, eig_interval) == 0
        [B, evals] = eig(C, 'vector');
        evals = max(real(evals), 1e-20);
        [evals, eig_order] = sort(evals, 'descend');
        B = real(B(:, eig_order));
        diagD = sqrt(evals(:));
    end
    curve(iter) = best_raw;

    if (stall > 80 || sigma < min_sigma) && restart_count < restart_limit
        restart_count = restart_count + 1;
        mean_x = best_position + randn(1, D) .* (0.01 / restart_count) .* span;
        mean_x = min(max(mean_x, lb), ub);
        sigma = get_option(options, 'restart_sigma', 0.08) * mean(span) / restart_count;
        if get_option(options, 'preserve_covariance_on_restart', false)
            C = initial_covariance_matrix(options, mean_x, sigma, lb, ub, D);
            [B, evals] = eig(C, 'vector');
            evals = max(real(evals), 1e-20);
            [evals, eig_order] = sort(evals, 'descend');
            B = real(B(:, eig_order));
            diagD = sqrt(evals(:));
        else
            C = eye(D);
            B = eye(D);
            diagD = ones(D, 1);
        end
        pc(:) = 0;
        ps(:) = 0;
        stall = 0;
    end
end
curve = curve(1:iter);

runtime = toc(t_start);
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
result.algorithm_combination = sprintf('Covariance Matrix Adaptation Evolution Strategy (CMA-ES)\nRestarted rank-based Gaussian distribution adaptation');
result.combination_number = 1;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = last_population;
result.final_fitness = last_fitness;

if verbose
    fprintf('CMA-ES finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function C = initial_covariance_matrix(options, mean_x, sigma, lb, ub, D)
C = eye(D);
if ~get_option(options, 'seed_covariance', false)
    return;
end
population = get_option(options, 'initial_population', []);
if isempty(population)
    return;
end
population = min(max(population, lb), ub);
rows = min(size(population, 1), get_option(options, 'covariance_seed_count', min(size(population, 1), max(12, 2 * D))));
if rows < 3
    return;
end
population = population(1:rows, :);
weights = log(rows + 0.5) - log(1:rows);
weights = weights(:) / sum(weights);
center = get_option(options, 'covariance_center', mean_x);
center = min(max(center(:)', lb), ub);
scaled = (population - center) ./ max(sigma, eps);
covariance = scaled' * (scaled .* weights);
covariance = (covariance + covariance') / 2;
blend = get_option(options, 'seed_covariance_blend', 0.65);
ridge = get_option(options, 'seed_covariance_ridge', 0.08);
C = (1 - blend) .* eye(D) + blend .* covariance + ridge .* eye(D);
[basis, values] = eig(C, 'vector');
values = max(real(values), get_option(options, 'seed_covariance_min_eig', 0.04));
values = min(values, get_option(options, 'seed_covariance_max_eig', 18.0));
C = real(basis) * diag(values) * real(basis)';
C = (C + C') / 2;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
