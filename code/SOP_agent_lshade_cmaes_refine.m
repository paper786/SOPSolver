function result = SOP_agent_lshade_cmaes_refine(problem, seed, options)
% L-SHADE family followed by local CMA-ES refinement.
%
% This staged hybrid keeps the reliable DE search as the basin finder and
% uses a rank-based covariance evolution strategy only around the best point
% found by the base stage.
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
base_fraction = get_option(options, 'base_fraction', 0.72);
base_algorithm = lower(string(get_option(options, 'base_algorithm', 'lshade')));

base_options = options;
base_options.max_runtime_sec = base_fraction * max_runtime_sec;
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.verbose = false;
switch base_algorithm
    case "lshade_cma"
        base = SOP_agent_lshade_cma(problem, double(seed), base_options);
        base_label = 'L-SHADE with elite covariance sampling';
    case "lshade_jso"
        base = SOP_agent_lshade_jso(problem, double(seed), base_options);
        base_label = 'jSO/L-SHADE success-history adaptation';
    case "sade"
        base_options.max_iter = max(1, floor((base_options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
        base = SOP_agent1_adaptive_de(problem, double(seed), base_options);
        base_label = 'Self-Adaptive DE/current-to-pbest with archive';
    otherwise
        base = SOP_agent_lshade(problem, double(seed), base_options);
        base_label = 'L-SHADE success-history adaptation';
end

remaining_time = max_runtime_sec - toc(t_start);
pattern_reserve_fes = get_option(options, 'pattern_reserve_fes', 0);
pattern_reserve_fes = max(0, min(pattern_reserve_fes, max(0, max_fes - base.evaluation_count - 1000)));
cma_options = struct();
cma_options.population_num = get_option(options, 'cma_population_num', max(28, round(0.35 * get_option(options, 'population_num', 180))));
tail_reserve_runtime_sec = get_option(options, 'tail_reserve_runtime_sec', 0);
if get_option(options, 'eda_after', false)
    tail_reserve_runtime_sec = max(tail_reserve_runtime_sec, get_option(options, 'eda_reserve_runtime_sec', 0));
end
cma_options.max_runtime_sec = max(1, remaining_time - tail_reserve_runtime_sec);
cma_options.max_fes = max(1000, max_fes - base.evaluation_count - pattern_reserve_fes);
cma_options.initial_point = base.best_position;
cma_options.sigma0 = get_option(options, 'local_sigma', 0.006);
cma_options.restart_sigma = get_option(options, 'restart_sigma', 0.003);
cma_options.restart_limit = get_option(options, 'restart_limit', 2);
cma_options.eig_interval = get_option(options, 'eig_interval', 8);
cma_options.verbose = false;
if get_option(options, 'cma_island_count', 1) > 1
    refine = run_cma_islands(problem, double(seed), base, cma_options, options);
else
    refine = SOP_agent_cma_es(problem, double(seed) + 104729, cma_options);
end

if refine.record_value < base.record_value
    result = refine;
else
    result = base;
end
extra_eval = 0;
extra_iter = 0;
extra_curve = [];
if get_option(options, 'lshade_cma_after', false) && toc(t_start) < max_runtime_sec
    relay_fes = max(0, max_fes - base.evaluation_count - refine.evaluation_count);
    if relay_fes > 0
        relay_options = options;
        relay_options.max_fes = relay_fes;
        relay_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
        relay_options.population_num = get_option(options, 'relay_population_num', max(44, round(0.20 * get_option(options, 'population_num', 180))));
        relay_options.initial_point = result.best_position;
        relay_options.initial_radius = get_option(options, 'relay_radius', 0.0018);
        relay_options.initial_cauchy = get_option(options, 'relay_cauchy', true);
        if get_option(options, 'relay_archive_exchange', false)
            relay_options.initial_population = block_archive_exchange_population(base, refine, result.best_position, ...
                relay_options.population_num, problem.lb, problem.ub, get_option(options, 'relay_exchange_block_rate', 0.08), ...
                get_option(options, 'relay_exchange_de_weight', 0.16));
            relay_options.preserve_initial_population_after_radius = true;
        end
        relay_options.cma_rate = get_option(options, 'relay_cma_rate', 0.14);
        relay_options.elite_rate = get_option(options, 'relay_elite_rate', 0.20);
        relay_options.cma_interval = get_option(options, 'relay_cma_interval', 10);
        relay_options.verbose = false;
        relay_algorithm = lower(string(get_option(options, 'relay_algorithm', 'lshade_cma')));
        if relay_algorithm == "jso"
            relay_options.ranked_r1 = true;
            relay_options.rank_pressure = get_option(options, 'relay_rank_pressure', 1.7);
            relay_options.mu_F_init = get_option(options, 'relay_mu_F_init', 0.34);
            relay_options.mu_CR_init = get_option(options, 'relay_mu_CR_init', 0.86);
            relay_options.p_rate_start = get_option(options, 'relay_p_rate_start', 0.20);
            relay_options.p_rate_end = get_option(options, 'relay_p_rate_end', 0.040);
            relay_options.archive_factor_start = get_option(options, 'relay_archive_factor_start', 1.8);
            relay_options.archive_factor_end = get_option(options, 'relay_archive_factor_end', 3.2);
            relay = SOP_agent_lshade_jso(problem, double(seed) + 65537, relay_options);
        else
            relay = SOP_agent_lshade_cma(problem, double(seed) + 65537, relay_options);
        end
        if relay.record_value < result.record_value
            result = relay;
        end
        extra_eval = extra_eval + relay.evaluation_count;
        extra_iter = extra_iter + relay.iteration;
        extra_curve = [extra_curve(:); relay.convergence_curve(:)];
    end
end
if get_option(options, 'pattern_after', false) && toc(t_start) < max_runtime_sec
    pattern_fes = max(0, max_fes - base.evaluation_count - refine.evaluation_count);
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
    end
end
if get_option(options, 'eda_after', false) && toc(t_start) < max_runtime_sec
    eda_fes = max(0, max_fes - base.evaluation_count - refine.evaluation_count - extra_eval);
    if eda_fes > 0
        eda_time = max(1, max_runtime_sec - toc(t_start));
        eda = eda_refine(problem, result.best_position, result.best_value, base, eda_fes, eda_time, options);
        if eda.record_value < result.record_value
            result.best_value = eda.best_value;
            result.record_value = eda.record_value;
            result.best_position = eda.best_position;
        end
        extra_eval = extra_eval + eda.evaluation_count;
        extra_iter = extra_iter + eda.iteration;
        extra_curve = [extra_curve(:); eda.convergence_curve(:)];
    end
end
result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + refine.evaluation_count + extra_eval;
result.iteration = base.iteration + refine.iteration + extra_iter;
result.convergence_curve = [base.convergence_curve(:); refine.convergence_curve(:); extra_curve(:)];
tail_label = '';
if get_option(options, 'lshade_cma_after', false)
    if lower(string(get_option(options, 'relay_algorithm', 'lshade_cma'))) == "jso"
        tail_label = sprintf('%s\njSO/RSP relay back from CMA-ES best point', tail_label);
    else
        tail_label = sprintf('%s\nL-SHADE-CMA relay back from CMA-ES best point', tail_label);
    end
end
if get_option(options, 'eda_after', false)
    tail_label = sprintf('%s\nElite EDA covariance sampling tail', tail_label);
end
result.algorithm_combination = sprintf('Differential Evolution (DE)\n%s\nLocal CMA-ES rank-based covariance refinement%s', base_label, tail_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function best_refine = run_cma_islands(problem, seed, base, cma_options, options)
island_count = max(2, round(get_option(options, 'cma_island_count', 2)));
total_fes = cma_options.max_fes;
total_time = cma_options.max_runtime_sec;
elites = result_elites(base);
if isempty(elites)
    elites = base.best_position;
end
sigma_scales = get_option(options, 'cma_island_sigma_scales', linspace(0.65, 1.35, island_count));
if numel(sigma_scales) < island_count
    sigma_scales = repmat(sigma_scales(:)', 1, ceil(island_count / numel(sigma_scales)));
end

best_refine = [];
all_curve = [];
total_eval = 0;
total_iter = 0;
total_runtime = 0;
for island = 1:island_count
    remaining_fes = total_fes - total_eval;
    remaining_time = total_time - total_runtime;
    if remaining_fes < 1000 || remaining_time <= 1
        break;
    end
    island_options = cma_options;
    island_options.max_fes = max(1000, floor(remaining_fes / max(1, island_count - island + 1)));
    island_options.max_runtime_sec = max(1, remaining_time / max(1, island_count - island + 1));
    elite_idx = 1 + mod(island - 1, size(elites, 1));
    island_options.initial_point = elites(elite_idx, :);
    if island > 1
        island_options.initial_point = 0.72 .* island_options.initial_point + 0.28 .* base.best_position;
    end
    island_options.sigma0 = cma_options.sigma0 .* sigma_scales(island);
    island_options.restart_sigma = cma_options.restart_sigma .* min(1.25, max(0.60, sigma_scales(island)));
    item = SOP_agent_cma_es(problem, seed + 104729 + 7919 * island, island_options);
    if isempty(best_refine) || item.record_value < best_refine.record_value
        best_refine = item;
    end
    total_eval = total_eval + item.evaluation_count;
    total_iter = total_iter + item.iteration;
    total_runtime = total_runtime + item.runtime;
    all_curve = [all_curve(:); raw_curve_for(item)]; %#ok<AGROW>
end
if isempty(best_refine)
    best_refine = SOP_agent_cma_es(problem, seed + 104729, cma_options);
    return;
end
best_refine.runtime = total_runtime;
best_refine.evaluation_count = total_eval;
best_refine.iteration = total_iter;
best_refine.raw_convergence_curve = all_curve;
best_refine.convergence_curve = SOP_cec_record_value(all_curve, problem);
best_refine.algorithm_combination = sprintf('%s\nMulti-island micro-CMA-ES covariance refinement', ...
    best_refine.algorithm_combination);
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
elseif isfield(result, 'convergence_curve') && ~isempty(result.convergence_curve)
    curve = result.convergence_curve(:);
else
    curve = result.record_value;
end
end

function pattern = pattern_refine(problem, best_x, best_raw, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
batch = get_option(options, 'pattern_samples', 96);
sigma = get_option(options, 'pattern_sigma', 0.00045) .* span;
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
        sigma = max(0.985 .* sigma, min_sigma);
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
pattern.convergence_curve = SOP_cec_record_value(curve, problem);
pattern.evaluation_count = eval_count;
pattern.iteration = iter;
end

function eda = eda_refine(problem, best_x, best_raw, base, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
if isfield(base, 'final_population') && ~isempty(base.final_population)
    population = base.final_population;
else
    population = repmat(best_x, max(20, get_option(options, 'eda_population_num', 80)), 1);
end
if isfield(base, 'final_fitness') && ~isempty(base.final_fitness) && numel(base.final_fitness) == size(population, 1)
    fitness = base.final_fitness(:);
else
    fitness = SOP_cec_evaluate(population, problem);
end
[fitness, order] = sort(fitness);
population = population(order, :);
elite_count = min(size(population, 1), get_option(options, 'eda_elite_count', max(12, round(0.18 * size(population, 1)))));
batch = get_option(options, 'eda_batch', max(96, round(0.50 * size(population, 1))));
cov_scale = get_option(options, 'eda_cov_scale', 0.040);
iso_scale = get_option(options, 'eda_iso_scale', 0.00025) .* span;
min_iso = get_option(options, 'eda_min_iso_scale', 1e-8) .* max(1, span);
reset_cov_scale = get_option(options, 'eda_reset_cov_scale', 0.012);
reset_iso = get_option(options, 'eda_reset_iso_scale', 0.00008) .* span;
block_rate = get_option(options, 'eda_block_rate', max(0.028, 4 / D));
eval_count = 0;
iter = 0;
stall = 0;
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = sample_eda_candidates(population, best_x, lb, ub, elite_count, count, cov_scale, iso_scale, block_rate);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        population = [best_x; population(1:end-1, :)]; %#ok<AGROW>
        fitness = [best_raw; fitness(1:end-1)]; %#ok<AGROW>
        cov_scale = max(0.72 * cov_scale, 0.002);
        iso_scale = max(0.86 .* iso_scale, min_iso);
        stall = 0;
    else
        cov_scale = max(0.80 * cov_scale, 0.0012);
        iso_scale = max(0.74 .* iso_scale, min_iso);
        stall = stall + 1;
    end
    if stall >= 10
        cov_scale = max(cov_scale, reset_cov_scale);
        iso_scale = max(iso_scale, reset_iso);
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
eda = struct();
eda.best_value = best_raw;
eda.record_value = SOP_cec_record_value(best_raw, problem);
eda.best_position = best_x;
eda.convergence_curve = SOP_cec_record_value(curve, problem);
eda.evaluation_count = eval_count;
eda.iteration = iter;
end

function candidates = sample_eda_candidates(population, best_x, lb, ub, elite_count, count, cov_scale, iso_scale, block_rate)
D = numel(best_x);
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
center = 0.70 * best_x + 0.30 * center;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((iso_scale .^ 2) + 1e-16);
[R, flag] = chol(cov_matrix, 'upper');
if flag ~= 0
    R = diag(sqrt(max(diag(cov_matrix), 1e-16)));
end
candidates = repmat(best_x, count, 1);
for i = 1:count
    if rand() < 0.62
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
    candidates(i, :) = min(max(x, lb), ub);
end
end

function population = block_archive_exchange_population(base, refine, best_x, NP, lb, ub, block_rate, de_weight)
D = numel(best_x);
span = ub - lb;
base_elites = result_elites(base);
refine_elites = result_elites(refine);
if isempty(base_elites)
    base_elites = best_x;
end
if isempty(refine_elites)
    refine_elites = best_x;
end
population = repmat(best_x, NP, 1);
population(1, :) = best_x;
if NP >= 2
    population(2, :) = base_elites(1, :);
end
if NP >= 3
    population(3, :) = refine_elites(1, :);
end
for i = 4:NP
    a = base_elites(randi(size(base_elites, 1)), :);
    b = refine_elites(randi(size(refine_elites, 1)), :);
    child = best_x;
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randperm(D, max(1, round(block_rate * D)))) = true;
    end
    if rand() < 0.55
        child(mask) = a(mask);
    else
        child(mask) = b(mask);
    end
    if rand() < 0.50
        c = base_elites(randi(size(base_elites, 1)), :);
        d = refine_elites(randi(size(refine_elites, 1)), :);
        child(mask) = child(mask) + de_weight .* (c(mask) - d(mask));
    end
    if rand() < 0.18
        child(mask) = child(mask) + randn(1, sum(mask)) .* (0.10 .* span(mask));
    end
    population(i, :) = min(max(child, lb), ub);
end
population = min(max(population, lb), ub);
end

function elites = result_elites(result)
if isstruct(result) && isfield(result, 'final_population') && ~isempty(result.final_population)
    elites = result.final_population;
elseif isstruct(result) && isfield(result, 'best_position') && ~isempty(result.best_position)
    elites = result.best_position;
else
    elites = [];
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
