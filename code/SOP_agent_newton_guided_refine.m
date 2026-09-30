function result = SOP_agent_newton_guided_refine(problem, seed, options)
% Short black-box Newton-inspired guidance refiner.
%
% mode="nrbo" uses an NRBO-style Newton-Raphson Search Rule (NRSR).
% mode="ndo" uses an NDO-style Newton-downhill point with SSO/HGO updates.
% The implementation is derivative-free: it uses only population positions
% and objective values already available from SOP_cec_evaluate.
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
NP = get_option(options, 'population_num', 80);
max_fes = get_option(options, 'max_fes', 120000);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
mode = lower(string(get_option(options, 'mode', 'nrbo')));
radius = make_radius(get_option(options, 'initial_radius', 0.003), span, D);

population = initial_population(problem, options, NP, lb, ub, span, radius);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
curve = zeros(max(1, ceil(max_fes / max(1, NP))), 1);
actual_iter = 0;
stall = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    actual_iter = actual_iter + 1;
    progress = min(1, eval_count / max(1, max_fes));
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_position = population(1, :);
    end
    worst_position = population(end, :);
    trial = population;
    for i = 1:NP
        if mode == "ndo"
            trial(i, :) = ndo_guided_trial(population, fitness, i, best_position, best_raw, lb, ub, span, progress, options);
        else
            trial(i, :) = nrbo_guided_trial(population, fitness, i, best_position, worst_position, lb, ub, span, actual_iter, progress, options);
        end
    end
    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
    end
    trial_fitness = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    rows = numel(trial_fitness);
    improved = trial_fitness <= fitness(1:rows);
    if any(improved)
        population(improved, :) = trial(improved, :);
        fitness(improved) = trial_fitness(improved);
        [current_best, current_idx] = min(fitness);
        if current_best < best_raw
            best_raw = current_best;
            best_position = population(current_idx, :);
            stall = 0;
        else
            stall = stall + 1;
        end
    else
        stall = stall + 1;
    end
    if stall >= get_option(options, 'restart_after_stall', 10)
        [population, fitness, eval_delta] = restart_around_best(problem, population, fitness, best_position, lb, ub, radius, max_fes - eval_count, options);
        eval_count = eval_count + eval_delta;
        radius = max(0.55 .* radius, get_option(options, 'min_radius_scale', 2e-5) .* span);
        stall = 0;
    end
    if actual_iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(actual_iter) = best_raw;
end

curve = curve(1:actual_iter);
[final_fitness, final_order] = sort(fitness);
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = toc(t_start);
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = label_for(mode);
result.combination_number = 2;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;
end

function population = initial_population(problem, options, NP, lb, ub, span, radius)
D = problem.dimension;
population = [];
if isfield(options, 'initial_population') && ~isempty(options.initial_population)
    population = options.initial_population;
    if size(population, 2) ~= D
        population = [];
    end
end
if isempty(population)
    initial_point = get_option(options, 'initial_point', []);
    if ~isempty(initial_point)
        center = min(max(initial_point(:)', lb), ub);
        population = repmat(center, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
        if get_option(options, 'initial_cauchy', true)
            cauchy = tan(pi * (rand(NP, D) - 0.5));
            cauchy = min(max(cauchy, -8), 8);
            mask = rand(NP, D) < get_option(options, 'cauchy_rate', 0.22);
            center_matrix = repmat(center, NP, 1);
            radius_matrix = repmat(radius, NP, 1);
            population(mask) = center_matrix(mask) + cauchy(mask) .* radius_matrix(mask);
        end
        population(1, :) = center;
    else
        population = lb + rand(NP, D) .* span;
    end
end
if size(population, 1) < NP
    extra = lb + rand(NP - size(population, 1), D) .* span;
    population = [population; extra]; %#ok<AGROW>
elseif size(population, 1) > NP
    population = population(1:NP, :);
end
population = min(max(population, lb), ub);
end

function candidate = nrbo_guided_trial(population, fitness, i, best, worst, lb, ub, span, iter, progress, options)
[NP, D] = size(population);
x = population(i, :);
r1 = random_index_except(NP, i);
r2 = random_index_except(NP, r1);
delta_x = rand(1, D) .* abs(best - x) ./ max(1, iter);
den = 2 .* (worst + best - 2 .* x);
den_floor = 1e-12 .* max(1, abs(span));
small_den = abs(den) < den_floor;
den(small_den) = sign_nonzero(den(small_den)) .* den_floor(small_den);
nrsr = randn(1, D) .* ((worst - best) .* delta_x) ./ den;
nrsr = cap_step(nrsr, get_option(options, 'nrsr_step_cap', 0.030) .* span);
rho = (1 - 2 * progress) ^ 5;
pull = rand() .* (best - x) + rand() .* (population(r1, :) - population(r2, :));
pull = cap_step(pull, get_option(options, 'pull_step_cap', 0.018) .* span);
candidate = x - rho .* nrsr + get_option(options, 'pull_scale', 0.22) .* pull;
if rand() < get_option(options, 'tao_rate', 0.18)
    mean_pop = mean(population, 1);
    tao = rand(1, D) .* (mean_pop - x) + rand(1, D) .* (best - population(r1, :));
    candidate = candidate + get_option(options, 'tao_scale', 0.20) .* cap_step(tao, get_option(options, 'tao_step_cap', 0.020) .* span);
end
candidate = repair(candidate, x, lb, ub);
end

function candidate = ndo_guided_trial(population, fitness, i, best, best_raw, lb, ub, span, progress, options)
[NP, D] = size(population);
x = population(i, :);
fi = fitness(i);
r1 = random_index_except(NP, i);
r2 = random_index_except(NP, r1);
downhill_gain = abs(fi) / (abs(fi - best_raw) + eps);
downhill_gain = min(get_option(options, 'downhill_gain_cap', 2.8), max(0.08, downhill_gain));
lambda = min(get_option(options, 'lambda_cap', 1.35), rand() / max(rand(), 0.20));
x_star = x + lambda .* downhill_gain .* (best - x);
x_star = repair(x_star, x, lb, ub);
if rand() <= 0.5
    w1 = rand(1, D) <= 0.5;
    w2 = rand(1, D);
    ratio = abs(fitness(r2)) / (abs(fitness(r1)) + eps);
    w3 = exp(-progress * min(12, ratio));
    candidate = x;
    candidate(~w1) = best(~w1) + w2(~w1) .* (population(r1, ~w1) - population(r2, ~w1));
    candidate = candidate + w3 .* get_option(options, 'sso_downhill_scale', 0.26) .* (x_star - x);
else
    if rand() <= 0.5
        w4 = rand(1, D);
        mask = rand(1, D) <= 0.5;
        candidate = w4 .* x + (1 - w4) .* x_star;
        candidate(mask) = candidate(mask) + get_option(options, 'hgo_random_scale', 0.18) .* (population(r1, mask) - x(mask));
    else
        w5 = rand(1, D);
        w6 = rand(1, D);
        w7 = rand(1, D);
        candidate = w5 .* best + w6 .* (w7 .* (x_star - best) + (best - x));
    end
end
candidate = repair(candidate, x, lb, ub);
end

function [population, fitness, eval_count] = restart_around_best(problem, population, fitness, best, lb, ub, radius, remaining_fes, options)
NP = size(population, 1);
count = min(max(0, remaining_fes), max(4, round(get_option(options, 'restart_rate', 0.18) * NP)));
eval_count = 0;
if count <= 0
    return;
end
[~, order] = sort(fitness, 'descend');
rows = order(1:count);
candidates = repmat(best, count, 1) + randn(count, numel(best)) .* repmat(radius, count, 1);
candidates = min(max(candidates, lb), ub);
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
improved = values <= fitness(rows);
population(rows(improved), :) = candidates(improved, :);
fitness(rows(improved)) = values(improved);
end

function step = cap_step(step, cap)
step = min(max(step, -cap), cap);
end

function candidate = repair(candidate, parent, lb, ub)
low = candidate < lb;
high = candidate > ub;
candidate(low) = 0.5 .* (parent(low) + lb(low));
candidate(high) = 0.5 .* (parent(high) + ub(high));
candidate = min(max(candidate, lb), ub);
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function s = sign_nonzero(x)
s = sign(x);
s(s == 0) = 1;
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
else
    radius = reshape(radius, 1, []);
    if numel(radius) ~= D
        radius = median(radius(:)) .* ones(1, D);
    end
end
radius = max(radius, eps);
end

function label = label_for(mode)
if mode == "ndo"
    label = sprintf('Newton Downhill Optimizer guidance\nDHM with SSO/HGO-style population updates');
else
    label = sprintf('Newton-Raphson-Based Optimizer guidance\nNRSR with trap-avoidance perturbation');
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
