function result = SOP_agent_operator_pool_lshade_restart(problem, seed, options)
% Operator-pool basin finder crossed with L-SHADE-family local restart.
%
% This is a component crossover between the best F20 operator-pool search
% and the near-line L-SHADE-CMA restart/pattern refiner. It remains fully
% derivative-free and only uses objective evaluations.
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
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 210));
base_options.max_fes = min(max_fes, get_option(options, 'base_max_fes', floor(0.82 * max_fes)));
base_options.max_iter = max(1, floor((base_options.max_fes - NP) / NP));
base_options.verbose = false;
base = SOP_agent_operator_pool(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(1000, max_fes - base.evaluation_count);
pattern_reserve_fes = get_option(options, 'pattern_reserve_fes', 0);
pattern_reserve_fes = max(0, min(pattern_reserve_fes, max(0, remaining_fes - 1000)));
local_options = options;
local_options.population_num = get_option(options, 'local_population_num', max(70, round(0.45 * NP)));
local_options.max_runtime_sec = remaining_time;
local_options.max_fes = remaining_fes - pattern_reserve_fes;
local_options.initial_point = base.best_position;
local_options.initial_radius = get_option(options, 'local_radius', 0.004);
local_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
local_options.initial_population = build_elite_initial_population(base.best_position, base.final_population, ...
    local_options.population_num, local_options.initial_radius, problem.lb, problem.ub);
local_options.cma_rate = get_option(options, 'cma_rate', 0.16);
local_options.elite_rate = get_option(options, 'elite_rate', 0.22);
local_options.cma_interval = get_option(options, 'cma_interval', 12);
local_options.verbose = false;
local_algorithm = lower(string(get_option(options, 'local_algorithm', 'lshade_cma')));
switch local_algorithm
    case "lshade"
        local = SOP_agent_lshade(problem, double(seed) + 65537, local_options);
        local_label = 'L-SHADE local restart';
    case "jso"
        local = SOP_agent_lshade_jso(problem, double(seed) + 65537, local_options);
        local_label = 'jSO/L-SHADE local restart';
    otherwise
        local = SOP_agent_lshade_cma(problem, double(seed) + 65537, local_options);
        local_label = 'L-SHADE-CMA local restart';
end

if local.record_value < base.record_value
    result = local;
else
    result = base;
end
extra_eval = 0;
extra_iter = 0;
extra_curve = [];
extra_raw_curve = [];
if get_option(options, 'pattern_after', true) && toc(t_start) < max_runtime_sec
    pattern_fes = max(0, max_fes - base.evaluation_count - local.evaluation_count);
    if pattern_fes > 0
        pattern_time = max(1, max_runtime_sec - toc(t_start));
        pattern = pattern_refine(problem, result.best_position, result.best_value, pattern_fes, pattern_time, options);
        if pattern.record_value < result.record_value
            result.best_value = pattern.best_value;
            result.record_value = pattern.record_value;
            result.best_position = pattern.best_position;
        end
        extra_eval = pattern.evaluation_count;
        extra_iter = pattern.iteration;
        extra_curve = pattern.convergence_curve(:);
        extra_raw_curve = pattern.raw_convergence_curve(:);
    end
end

result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + local.evaluation_count + extra_eval;
result.iteration = base.iteration + local.iteration + extra_iter;
result.convergence_curve = [base.convergence_curve(:); local.convergence_curve(:); extra_curve(:)];
result.raw_convergence_curve = [base.raw_convergence_curve(:); local.raw_convergence_curve(:); extra_raw_curve(:)];
result.algorithm_combination = sprintf('Adaptive operator-pool metaheuristic\nDE/GSK/RIME/elite covariance competition\n%s\nStochastic block pattern refinement', local_label);
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function initial_population = build_elite_initial_population(best_x, elite_population, NP, radius_scale, lb, ub)
D = numel(best_x);
span = ub - lb;
radius = radius_scale .* span;
if isempty(elite_population)
    initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
else
    elite_count = min(size(elite_population, 1), max(4, floor(0.45 * NP)));
    initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
    initial_population(1, :) = best_x;
    initial_population(2:elite_count + 1, :) = elite_population(1:elite_count, :);
    remaining = NP - elite_count - 1;
    if remaining > 0
        donor = elite_population(randi(elite_count, remaining, 1), :);
        rows = elite_count + 2:NP;
        initial_population(rows, :) = donor + randn(remaining, D) .* repmat(0.35 .* radius, remaining, 1);
    end
end
initial_population = min(max(initial_population, lb), ub);
end

function pattern = pattern_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
batch = get_option(options, 'pattern_samples', 96);
sigma = get_option(options, 'pattern_sigma', 0.0008) .* span;
min_sigma = get_option(options, 'pattern_min_sigma', 1e-9) .* max(1, span);
block_rate = get_option(options, 'pattern_block_rate', min(0.10, max(0.03, 5 / D)));
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = repmat(best_x, count, 1);
    for i = 1:count
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        if rand() < 0.35
            noise = tan(pi * (rand(1, D) - 0.5));
            noise = min(max(noise, -8), 8);
        else
            noise = randn(1, D);
        end
        candidates(i, mask) = candidates(i, mask) + noise(mask) .* sigma(mask);
    end
    candidates = min(max(candidates, lb), ub);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        sigma = max(0.96 .* sigma, min_sigma);
    else
        sigma = max(0.78 .* sigma, min_sigma);
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
pattern = struct();
pattern.best_value = best_raw;
pattern.record_value = SOP_cec_record_value(best_raw, problem);
pattern.best_position = best_x;
pattern.raw_convergence_curve = curve;
pattern.convergence_curve = SOP_cec_record_value(curve, problem);
pattern.evaluation_count = eval_count;
pattern.iteration = iter;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
