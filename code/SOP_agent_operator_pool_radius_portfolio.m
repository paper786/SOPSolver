function result = SOP_agent_operator_pool_radius_portfolio(problem, seed, options)
% Operator-pool basin search with a radius-adaptive DE portfolio refiner.
%
% The first phase reuses the successful DE/GSK/RIME/covariance operator
% pool. The second phase is a short portfolio of literature-grounded DE
% refiners with shrinking and re-expanding neighborhoods around the current
% best point. No derivative or deterministic numerical optimizer is used.
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
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 198));
base_options.max_fes = min(max_fes, get_option(options, 'base_max_fes', floor(0.76 * max_fes)));
base_options.max_iter = max(1, floor((base_options.max_fes - NP) / NP));
base_options.verbose = false;
base = SOP_agent_operator_pool(problem, double(seed), base_options);

result = base;
total_eval = base.evaluation_count;
total_iter = base.iteration;
curve_parts = {base.convergence_curve(:)};
raw_curve_parts = {base.raw_convergence_curve(:)};
stage_labels = {};

elite_population = base.final_population;
stage_radii = get_option(options, 'stage_radii', [0.0060, 0.0035, 0.0016]);
stage_algorithms = get_option(options, 'stage_algorithms', {'lshade_cma', 'jso', 'lshade_cma'});
stage_fes_rates = get_option(options, 'stage_fes_rates', [0.42, 0.34, 1.00]);
stage_pop_rates = get_option(options, 'stage_pop_rates', [0.44, 0.36, 0.30]);

for stage = 1:numel(stage_radii)
    if toc(t_start) >= max_runtime_sec || total_eval >= max_fes
        break;
    end
    remaining_time = max(1, max_runtime_sec - toc(t_start));
    remaining_fes = max_fes - total_eval;
    local_np = max(48, round(stage_pop_rates(min(stage, numel(stage_pop_rates))) * NP));
    if remaining_fes < max(4 * local_np, 1200)
        break;
    end
    fes_rate = stage_fes_rates(min(stage, numel(stage_fes_rates)));
    stage_fes = max(4 * local_np, floor(fes_rate * remaining_fes));
    stage_fes = min(stage_fes, remaining_fes);

    local_options = options;
    local_options.population_num = local_np;
    local_options.max_runtime_sec = remaining_time;
    local_options.max_fes = stage_fes;
    local_options.initial_point = result.best_position;
    local_options.initial_radius = stage_radii(stage);
    local_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
    local_options.initial_population = build_elite_initial_population(result.best_position, elite_population, ...
        local_np, stage_radii(stage), problem.lb, problem.ub);
    local_options.cma_rate = get_option(options, 'cma_rate', 0.16 + 0.02 * (stage == 1));
    local_options.elite_rate = get_option(options, 'elite_rate', 0.22);
    local_options.cma_interval = get_option(options, 'cma_interval', 12);
    local_options.ranked_r1 = get_option(options, 'ranked_r1', stage == 2);
    local_options.rank_pressure = get_option(options, 'rank_pressure', 1.8);
    local_options.p_rate = max(0.045, get_option(options, 'p_rate', 0.08) - 0.010 * (stage - 1));
    local_options.verbose = false;

    algorithm = lower(string(stage_algorithms{min(stage, numel(stage_algorithms))}));
    switch algorithm
        case "jso"
            local = SOP_agent_lshade_jso(problem, double(seed) + 811 * stage + 65537, local_options);
            stage_labels{end + 1} = sprintf('jSO/RSP radius %.4g', stage_radii(stage)); %#ok<AGROW>
        case "lshade"
            local = SOP_agent_lshade(problem, double(seed) + 811 * stage + 65537, local_options);
            stage_labels{end + 1} = sprintf('L-SHADE radius %.4g', stage_radii(stage)); %#ok<AGROW>
        otherwise
            local = SOP_agent_lshade_cma(problem, double(seed) + 811 * stage + 65537, local_options);
            stage_labels{end + 1} = sprintf('L-SHADE-CMA radius %.4g', stage_radii(stage)); %#ok<AGROW>
    end

    total_eval = total_eval + local.evaluation_count;
    total_iter = total_iter + local.iteration;
    curve_parts{end + 1} = local.convergence_curve(:); %#ok<AGROW>
    raw_curve_parts{end + 1} = local.raw_convergence_curve(:); %#ok<AGROW>
    if isfield(local, 'final_population') && ~isempty(local.final_population)
        elite_population = local.final_population;
    else
        elite_population = build_elite_initial_population(local.best_position, elite_population, local_np, ...
            max(0.50 * stage_radii(stage), 1e-5), problem.lb, problem.ub);
    end
    if local.record_value < result.record_value
        result = local;
    end
end

extra_eval = 0;
extra_iter = 0;
if get_option(options, 'pattern_after', true) && toc(t_start) < max_runtime_sec && total_eval < max_fes
    pattern_fes = max_fes - total_eval;
    pattern_time = max(1, max_runtime_sec - toc(t_start));
    pattern = pattern_refine(problem, result.best_position, result.best_value, pattern_fes, pattern_time, options);
    if pattern.record_value < result.record_value
        result.best_value = pattern.best_value;
        result.record_value = pattern.record_value;
        result.best_position = pattern.best_position;
    end
    extra_eval = pattern.evaluation_count;
    extra_iter = pattern.iteration;
    curve_parts{end + 1} = pattern.convergence_curve(:);
    raw_curve_parts{end + 1} = pattern.raw_convergence_curve(:);
end

result.runtime = toc(t_start);
result.evaluation_count = total_eval + extra_eval;
result.iteration = total_iter + extra_iter;
result.convergence_curve = vertcat(curve_parts{:});
result.raw_convergence_curve = vertcat(raw_curve_parts{:});
result.algorithm_combination = sprintf('Adaptive operator-pool metaheuristic\nDE/GSK/RIME/elite covariance basin search\n%s\nStochastic block pattern refinement', ...
    strjoin(stage_labels, newline));
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function initial_population = build_elite_initial_population(best_x, elite_population, NP, radius_scale, lb, ub)
D = numel(best_x);
span = ub - lb;
radius = radius_scale .* span;
initial_population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
initial_population(1, :) = best_x;
if ~isempty(elite_population)
    elite_count = min(size(elite_population, 1), max(4, floor(0.42 * NP)));
    initial_population(2:elite_count + 1, :) = elite_population(1:elite_count, :);
    remaining = NP - elite_count - 1;
    if remaining > 0
        donor = elite_population(randi(elite_count, remaining, 1), :);
        rows = elite_count + 2:NP;
        local_radius = max(0.25 .* radius, 1e-12 .* max(1, span));
        initial_population(rows, :) = donor + randn(remaining, D) .* repmat(local_radius, remaining, 1);
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
batch = get_option(options, 'pattern_samples', 160);
sigma = get_option(options, 'pattern_sigma', 0.00055) .* span;
min_sigma = get_option(options, 'pattern_min_sigma', 1e-9) .* max(1, span);
block_rate = get_option(options, 'pattern_block_rate', min(0.08, max(0.025, 4 / D)));
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
        if rand() < 0.30
            noise = tan(pi * (rand(1, D) - 0.5));
            noise = min(max(noise, -6), 6);
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
        sigma = max(0.92 .* sigma, min_sigma);
    else
        sigma = max(0.74 .* sigma, min_sigma);
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
