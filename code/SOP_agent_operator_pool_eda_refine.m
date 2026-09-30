function result = SOP_agent_operator_pool_eda_refine(problem, seed, options)
% Operator-pool search followed by elite EDA refinement.
%
% The base stage is the existing DE/GSK/RIME/covariance operator pool. The
% second stage estimates an elite covariance model from the final population
% and samples around the best point with shrinking radii and occasional
% heavy-tailed coordinate blocks. It is derivative-free and objective-only.
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
NP = get_option(options, 'population_num', 180);

base_options = options;
base_options.profile = get_option(options, 'profile', 'de_gsk_cma');
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'pre_runtime_sec', 215));
base_options.max_fes = min(max_fes, get_option(options, 'pre_max_fes', floor(0.86 * max_fes)));
base_options.max_iter = max(1, floor((base_options.max_fes - NP) / NP));
base_options.verbose = false;
base = SOP_agent_operator_pool(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
[best_x, best_raw, curve, extra_eval, extra_iter] = eda_refine(problem, base, remaining_fes, remaining_time, options);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = [base.raw_convergence_curve(:); curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + extra_iter;
result.evaluation_count = base.evaluation_count + extra_eval;
result.algorithm_combination = sprintf('Adaptive operator-pool metaheuristic\nDE/GSK/RIME/elite covariance competition\nElite EDA covariance and heavy-tailed block refinement');
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function [best_x, best_raw, curve, eval_count, iter] = eda_refine(problem, base, max_fes, max_runtime_sec, options)
t_start = tic;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
D = problem.dimension;
best_x = base.best_position;
best_raw = base.best_value;
population = base.final_population;
fitness = base.final_fitness;
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = min(size(population, 1), get_option(options, 'elite_count', max(12, round(0.16 * size(population, 1)))));
batch = get_option(options, 'eda_batch', max(96, round(0.45 * size(population, 1))));
sigma_floor = get_option(options, 'sigma_floor', 1e-7) .* span;
iso_scale = get_option(options, 'eda_iso_scale', 0.0016) .* span;
cov_scale = get_option(options, 'eda_cov_scale', 0.10);
block_rate = get_option(options, 'eda_block_rate', max(0.035, 5 / D));
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);
eval_count = 0;
iter = 0;
stall = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = sample_eda(population, fitness, best_x, lb, ub, span, elite_count, count, cov_scale, iso_scale, block_rate);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        population = [best_x; population(1:end-1, :)];
        fitness = [best_raw; fitness(1:end-1)];
        cov_scale = max(0.72 * cov_scale, 0.002);
        iso_scale = max(0.88 .* iso_scale, sigma_floor);
        stall = 0;
    else
        cov_scale = max(0.82 * cov_scale, 0.001);
        iso_scale = max(0.76 .* iso_scale, sigma_floor);
        stall = stall + 1;
    end
    if stall >= 10
        cov_scale = max(cov_scale, get_option(options, 'eda_reset_cov_scale', 0.035));
        iso_scale = max(iso_scale, get_option(options, 'eda_reset_iso_scale', 0.0006) .* span);
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
end

function candidates = sample_eda(population, fitness, best_x, lb, ub, span, elite_count, count, cov_scale, iso_scale, block_rate)
D = numel(best_x);
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
center = 0.65 * best_x + 0.35 * center;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((iso_scale .^ 2) + 1e-16);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), 1e-16)));
end
candidates = repmat(best_x, count, 1);
for i = 1:count
    if rand() < 0.70
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
    if rank_hint(fitness, i) > 0.8 && rand() < 0.20
        x = x + randn(1, D) .* (0.00035 .* span);
    end
    candidates(i, :) = min(max(x, lb), ub);
end
end

function hint = rank_hint(fitness, i)
hint = min(1, i / max(1, numel(fitness)));
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
