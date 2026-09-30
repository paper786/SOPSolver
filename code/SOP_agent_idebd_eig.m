function result = SOP_agent_idebd_eig(problem, seed, options)
% Individual-dependent DE with alternating binomial/eigen crossover.
%
% The implementation follows the IDEbd structure used by EA4Eig: ranked
% superior/inferior sampling, individual-dependent F and CR, two search
% stages, linear population reduction, and probabilistic eigen crossover.
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
NP_init = get_option(options, 'population_num', 100);
NP_min = min(NP_init, get_option(options, 'min_population_num', max(10, round(0.19 * NP_init))));
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
sigma_normal = get_option(options, 'sigma_normal', 0.26);
ps_min = get_option(options, 'ps_min', 0.08);
ps_shape = get_option(options, 'ps_shape', 4.14);
stage_divisor = get_option(options, 'stage_divisor', 2.31);
eig_rate = get_option(options, 'eig_rate', 0.18);
eig_elite_rate = get_option(options, 'eig_elite_rate', 0.29);
eig_interval = max(1, round(get_option(options, 'eig_interval', 20)));
failure_divisor = max(1, get_option(options, 'failure_divisor', 20));
boundary_mode = lower(string(get_option(options, 'boundary_mode', 'half')));
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP_init, D) .* span;
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);

t = 0;
t_max = max(1, floor(max_fes / NP_init));
stage_threshold = max(1, floor(t_max / stage_divisor));
failure_threshold = max(1, floor(t_max / failure_divisor));
failure_count = 0;
eigen_basis = eye(D);
eigen_cr = sample_eigen_cr(NP_init, 0.19);
convergence_raw = zeros(max(1, ceil((max_fes - NP_init) / NP_min)), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    t = t + 1;
    t_max = max(t_max, t);
    [fitness, order] = sort(fitness(:));
    population = population(order, :);
    NP = size(population, 1);

    progress = min(1, eval_count / max_fes);
    stage_threshold = max(1, stage_threshold);
    ps = ps_min + (1 - ps_min) * 10 ^ (ps_shape * (min(1, t / t_max) - 1));
    superior_count = min(NP - 3, max(2, round(ps * NP)));

    if t == 1 || mod(t - 1, eig_interval) == 0
        eigen_basis = population_eigen_basis(population, eig_elite_rate);
    end

    trial = population;
    F_values = zeros(NP, 1);
    CR_values = zeros(NP, 1);
    use_eigen = rand(NP, 1) < eig_rate;
    for i = 1:NP
        indexes = sample_distinct_indexes(NP, i, 4);
        b = indexes(1);
        r1 = indexes(2);
        r2 = indexes(3);
        r3 = indexes(4);
        if t <= stage_threshold
            b = i;
        end
        if b > superior_count && r1 > superior_count
            allowed = setdiff(1:superior_count, [i, b, r2, r3]);
            if ~isempty(allowed)
                r1 = allowed(randi(numel(allowed)));
            end
        end

        F = b / NP + sigma_normal * randn();
        while F <= 0
            F = b / NP + sigma_normal * randn();
        end
        F = min(F, 1);
        CR = min(1, max(0, i / NP + sigma_normal * randn()));
        F_values(i) = F;
        CR_values(i) = CR;

        mutant = population(b, :) ...
            + F .* (population(r1, :) - population(b, :)) ...
            + F .* (population(r2, :) - population(r3, :));
        if use_eigen(i)
            cr = eigen_cr(min(i, numel(eigen_cr)));
            target_rot = population(i, :) * eigen_basis;
            mutant_rot = mutant * eigen_basis;
            mask = rand(1, D) <= cr;
            mask(randi(D)) = true;
            child = target_rot;
            child(mask) = mutant_rot(mask);
            child = child * eigen_basis';
        else
            mask = rand(1, D) <= CR;
            mask(randi(D)) = true;
            child = population(i, :);
            child(mask) = mutant(mask);
        end
        trial(i, :) = repair_bounds(child, population(i, :), lb, ub, boundary_mode);
    end

    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
        parent_rows = (1:remaining)';
    else
        parent_rows = (1:NP)';
    end
    if isempty(parent_rows)
        break;
    end
    trial_fitness = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(trial_fitness);
    improved = trial_fitness <= fitness(parent_rows);
    if any(improved)
        rows = parent_rows(improved);
        population(rows, :) = trial(improved, :);
        fitness(rows) = trial_fitness(improved);
        failure_count = 0;
    elseif t < stage_threshold
        failure_count = failure_count + 1;
    end
    if t < stage_threshold && failure_count >= failure_threshold
        stage_threshold = t;
    end

    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    convergence_raw(t) = best_raw;

    target_NP = max(NP_min, round(NP_init - (NP_init - NP_min) * progress));
    if target_NP < size(population, 1)
        [fitness, order] = sort(fitness(:));
        population = population(order(1:target_NP), :);
        fitness = fitness(1:target_NP);
    end
end

convergence_raw = convergence_raw(1:t);
runtime = toc(t_start);
[final_fitness, order] = sort(fitness(:));
final_population = population(order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(convergence_raw, problem);
result.raw_convergence_curve = convergence_raw;
result.runtime = runtime;
result.iteration = t;
result.population_num = NP_init;
result.evaluation_count = eval_count;
result.final_population = final_population;
result.final_fitness = final_fitness;
result.algorithm_combination = sprintf(['Individual-dependent differential evolution (IDEbd)\n' ...
    'Success-ranked sampling with probabilistic eigen/binomial crossover']);
result.combination_number = 2;
result.agent_id = 'Agent1';
result.problem = problem;

if verbose
    fprintf('IDEbd-Eigen finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function basis = population_eigen_basis(population, elite_rate)
NP = size(population, 1);
D = size(population, 2);
elite_count = min(NP, max(3, round(elite_rate * NP)));
centered = population(1:elite_count, :) - mean(population(1:elite_count, :), 1);
covariance = (centered' * centered) / max(1, elite_count - 1);
covariance = 0.5 * (covariance + covariance') + 1e-12 * eye(D);
[basis, values] = eig(covariance, 'vector');
[~, order] = sort(values, 'descend');
basis = basis(:, order);
end

function values = sample_eigen_cr(NP, sigma)
locations = 0.1 + 0.85 .* (rand(NP, 1) >= 0.5);
values = locations + sigma .* tan(pi .* (rand(NP, 1) - 0.5));
values = min(1, max(0, values));
end

function indexes = sample_distinct_indexes(NP, excluded, count)
pool = 1:NP;
pool(excluded) = [];
indexes = pool(randperm(numel(pool), count));
end

function child = repair_bounds(child, parent, lb, ub, mode)
if mode == "clamp"
    child = min(max(child, lb), ub);
    return;
end
low = child < lb;
high = child > ub;
child(low) = 0.5 .* (parent(low) + lb(low));
child(high) = 0.5 .* (parent(high) + ub(high));
child = min(max(child, lb), ub);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
