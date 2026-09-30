function result = SOP_agent_lshade_tlbo_refine(problem, seed, options)
% L-SHADE-CMA basin search followed by compact TLBO learner refinement.
%
% TLBO here is used only as a local population learning layer around the
% best DE solution, giving a different exploitation dynamic from DE/CMA.
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
base_fes = min(max_fes, get_option(options, 'base_max_fes', 1000000));

base_options = options;
base_options.max_fes = base_fes;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 0.55 * max_runtime_sec));
base_options.verbose = false;
base = SOP_agent_lshade_cma(problem, double(seed), base_options);

D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'local_population_num', max(60, round(0.45 * get_option(options, 'population_num', 180))));
radius = get_option(options, 'local_radius', 0.006) .* span;
population = repmat(base.best_position, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    elite_count = min(size(base.final_population, 1), max(4, floor(0.35 * NP)));
    population(1:elite_count, :) = base.final_population(1:elite_count, :);
end
population(1, :) = base.best_position;
population = min(max(population, lb), ub);
fitness = SOP_cec_evaluate(population, problem);
eval_count = base.evaluation_count + numel(fitness);
[best_raw, best_idx] = min([base.best_value; fitness(:)]);
if best_idx == 1
    best_x = base.best_position;
else
    best_x = population(best_idx - 1, :);
end
curve = base.raw_convergence_curve(:);
iter = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    teacher = population(1, :);
    mean_x = mean(population, 1);
    trial = population;
    for i = 1:NP
        partner = random_index_except(NP, i);
        TF = randi(2);
        candidate = population(i, :) + rand(1, D) .* (teacher - TF .* mean_x);
        if fitness(i) < fitness(partner)
            candidate = candidate + rand(1, D) .* (population(i, :) - population(partner, :));
        else
            candidate = candidate + rand(1, D) .* (population(partner, :) - population(i, :));
        end
        if rand() < 0.20
            mask = rand(1, D) < min(0.15, max(0.04, 8 / D));
            if ~any(mask), mask(randi(D)) = true; end
            candidate(mask) = best_x(mask) + randn(1, sum(mask)) .* radius(mask);
        end
        trial(i, :) = min(max(candidate, lb), ub);
    end
    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
    end
    values = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    rows = numel(values);
    improved = values <= fitness(1:rows);
    population(improved, :) = trial(improved, :);
    fitness(improved) = values(improved);
    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_x = population(current_idx, :);
        radius = max(0.985 .* radius, 1e-9 .* max(1, span));
    else
        radius = max(0.90 .* radius, 1e-9 .* max(1, span));
    end
    if mod(iter, 10) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + iter;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE with elite covariance sampling\nLocal Teaching-Learning-Based Optimization refinement');
result.combination_number = 3;
result.agent_id = 'Agent2';
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
