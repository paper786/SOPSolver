function result = SOP_agent_family_then_swarm_relay(problem, seed, options)
% DE-family basin search followed by a short literature-swarm relay.
%
% This is a metaheuristic-only relay: the first phase uses a selected
% L-SHADE-family optimizer to find a basin, and the second phase restarts a
% different literature metaheuristic near that best point.
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
base_options.max_runtime_sec = max(1, base_fraction * max_runtime_sec);
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.verbose = false;
[base, base_label] = run_base(problem, double(seed), base_options, base_algorithm);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
relay = [];
relay_label = 'No relay budget remaining';
if remaining_fes > 0 && remaining_time > 1
    relay_method = lower(string(get_option(options, 'relay_method', 'rime')));
    relay_options = options;
    relay_options.population_num = get_option(options, 'relay_population_num', max(40, round(0.35 * get_option(options, 'population_num', 180))));
    relay_options.max_runtime_sec = remaining_time;
    relay_options.max_fes = remaining_fes;
    relay_options.initial_point = base.best_position;
    relay_options.initial_radius = get_option(options, 'relay_radius', 0.006);
    relay_options.initial_cauchy = get_option(options, 'relay_cauchy', true);
    relay_options.verbose = false;
    switch relay_method
        case "gsk"
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay_options.knowledge_rate = get_option(options, 'relay_knowledge_rate', 0.70);
            relay_options.knowledge_factor = get_option(options, 'relay_knowledge_factor', 0.34);
            relay = SOP_agent_gsk(problem, double(seed) + 65537, relay_options);
            relay_label = 'Gaining-Sharing Knowledge local relay';
        case "lshade"
            relay = SOP_agent_lshade(problem, double(seed) + 65537, relay_options);
            relay_label = 'L-SHADE local relay';
        case "lshade_cma"
            relay = SOP_agent_lshade_cma(problem, double(seed) + 65537, relay_options);
            relay_label = 'L-SHADE-CMA local relay';
        case "jso"
            relay = SOP_agent_lshade_jso(problem, double(seed) + 65537, relay_options);
            relay_label = 'jSO/L-SHADE local relay';
        case "island"
            if isfield(base, 'final_population') && ~isempty(base.final_population)
                relay_options.initial_population = base.final_population;
                relay_options.preserve_initial_population_after_radius = true;
            end
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay_options.island_count = get_option(options, 'relay_island_count', 4);
            relay_options.block_rate_start = get_option(options, 'relay_block_rate_start', 0.14);
            relay_options.block_rate_end = get_option(options, 'relay_block_rate_end', 0.032);
            relay_options.block_de_weight = get_option(options, 'relay_block_de_weight', 0.10);
            relay_options.wide_block_probability = get_option(options, 'relay_wide_block_probability', 0.08);
            relay_options.migration_interval = get_option(options, 'relay_migration_interval', 12);
            relay_options.migration_sigma = get_option(options, 'relay_migration_sigma', 0.0012);
            relay_options.migration_block_rate = get_option(options, 'relay_migration_block_rate', 0.08);
            relay = SOP_agent_block_island_pool(problem, double(seed) + 65537, relay_options);
            relay_label = 'Cooperative GSK/RIME/TLBO/MPA-WSO block-island local relay';
        case "ccde"
            relay = ccde_tail_refine(problem, base, double(seed) + 65537, relay_options);
            relay_label = 'Cooperative coevolution random-subspace DE local relay';
        case "nrbo"
            relay_options.mode = 'nrbo';
            if isfield(base, 'final_population') && ~isempty(base.final_population)
                relay_options.initial_population = base.final_population;
            end
            relay = SOP_agent_newton_guided_refine(problem, double(seed) + 65537, relay_options);
            relay_label = 'Newton-Raphson Search Rule local guidance relay';
        case "ndo"
            relay_options.mode = 'ndo';
            if isfield(base, 'final_population') && ~isempty(base.final_population)
                relay_options.initial_population = base.final_population;
            end
            relay = SOP_agent_newton_guided_refine(problem, double(seed) + 65537, relay_options);
            relay_label = 'Newton-downhill SSO/HGO local guidance relay';
        case {"bfgs", "simplex", "powell"}
            relay_options.mode = char(relay_method);
            if isfield(base, 'final_population') && ~isempty(base.final_population)
                relay_options.initial_population = base.final_population;
            end
            relay = SOP_agent_numeric_guided_refine(problem, double(seed) + 65537, relay_options);
            relay_label = sprintf('%s local numerical-optimization guidance relay', upper(char(relay_method)));
        otherwise
            relay_options.method = char(relay_method);
            relay_options.max_iter = max(1, floor(remaining_fes / relay_options.population_num));
            relay = SOP_agent_literature_swarm(problem, double(seed) + 65537, relay_options);
            relay_label = sprintf('%s local literature-swarm relay', upper(char(relay_method)));
    end
end

result = base;
if ~isempty(relay) && relay.record_value < result.record_value
    result = relay;
end
result.runtime = toc(t_start);
if isempty(relay)
    result.evaluation_count = base.evaluation_count;
    result.iteration = base.iteration;
    result.convergence_curve = base.convergence_curve(:);
    result.raw_convergence_curve = raw_curve_for(base);
else
    result.evaluation_count = base.evaluation_count + relay.evaluation_count;
    result.iteration = base.iteration + relay.iteration;
    result.convergence_curve = [base.convergence_curve(:); relay.convergence_curve(:)];
    result.raw_convergence_curve = [raw_curve_for(base); raw_curve_for(relay)];
end
result.algorithm_combination = sprintf('%s\n%s', base_label, relay_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function result = ccde_tail_refine(problem, base, seed, options)
if ~isempty(seed)
    rng(double(seed), 'twister');
end
t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 1000);
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', 72);
best_x = base.best_position;
best_raw = base.best_value;
radius = get_option(options, 'ccde_radius', get_option(options, 'initial_radius', 0.0030)) .* span;
min_radius = get_option(options, 'ccde_min_radius', 1e-8) .* max(1, span);
reset_radius = get_option(options, 'ccde_reset_radius', 0.0045) .* span;
block_rate = get_option(options, 'ccde_block_rate', min(0.18, max(0.035, 6 / D)));
block_min = get_option(options, 'ccde_block_min', 4);
F0 = get_option(options, 'ccde_F', 0.48);
CR0 = get_option(options, 'ccde_CR', 0.34);

population = repmat(best_x, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    rows = min(NP, size(base.final_population, 1));
    population(1:rows, :) = base.final_population(1:rows, :);
end
population(1, :) = best_x;
population = min(max(population, lb), ub);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[trial_raw, idx] = min(fitness);
if trial_raw < best_raw
    best_raw = trial_raw;
    best_x = population(idx, :);
end
curve = zeros(max(1, ceil(max_fes / max(1, NP))), 1);
iter = 0;
stall = 0;
archive = zeros(0, D);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_x = population(1, :);
    end
    combined = [population; archive];
    trial = population;
    for i = 1:NP
        mask = rand(1, D) < block_rate;
        if sum(mask) < block_min
            cols = randperm(D, min(D, block_min));
            mask(cols) = true;
        end
        p_count = max(2, round((0.08 + 0.08 * rand()) * NP));
        pbest = randi(p_count);
        r1 = random_index_except(NP, i);
        r2 = randi(size(combined, 1));
        F = min(0.88, max(0.12, F0 + 0.16 * tan(pi * (rand() - 0.5))));
        CR = min(1, max(0.02, CR0 + 0.14 * randn()));
        mutant = population(i, :);
        mutant(mask) = population(i, mask) + ...
            F .* (population(pbest, mask) - population(i, mask)) + ...
            F .* (population(r1, mask) - combined(r2, mask));
        cross = (rand(1, D) < CR) & mask;
        if ~any(cross)
            mask_idx = find(mask);
            cross(mask_idx(randi(numel(mask_idx)))) = true;
        end
        candidate = population(i, :);
        candidate(cross) = mutant(cross);
        if rand() < get_option(options, 'ccde_noise_rate', 0.16)
            candidate(mask) = candidate(mask) + randn(1, sum(mask)) .* radius(mask);
        end
        low = candidate < lb;
        high = candidate > ub;
        candidate(low) = 0.5 * (population(i, low) + lb(low));
        candidate(high) = 0.5 * (population(i, high) + ub(high));
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
    improved = values(:) <= fitness(1:rows);
    if any(improved)
        archive = [archive; population(improved, :)]; %#ok<AGROW>
        if size(archive, 1) > 2 * NP
            archive = archive(randperm(size(archive, 1), 2 * NP), :);
        end
        population(improved, :) = trial(improved, :);
        fitness(improved) = values(improved);
    end
    [iter_best, iter_idx] = min(fitness);
    if iter_best < best_raw
        best_raw = iter_best;
        best_x = population(iter_idx, :);
        radius = max(0.985 .* radius, min_radius);
        stall = 0;
    else
        radius = max(0.94 .* radius, min_radius);
        stall = stall + 1;
    end
    if stall >= 28
        radius = max(radius, reset_radius .* (0.70 + 0.60 * rand(1, D)));
        stall = 0;
    end
    if iter > numel(curve)
        curve(end + 128, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = iter;
result.evaluation_count = eval_count;
result.population_num = NP;
result.final_population = population;
result.final_fitness = fitness;
result.algorithm_combination = 'Cooperative coevolution random-subspace DE refinement';
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function [result, label] = run_base(problem, seed, options, algorithm)
switch algorithm
    case "gsk_exploit"
        options.max_iter = max(1, floor((options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
        options.knowledge_rate = get_option(options, 'knowledge_rate', 0.72);
        options.knowledge_factor = get_option(options, 'knowledge_factor', 0.38);
        result = SOP_agent_gsk(problem, seed, options);
        label = 'GSK exploitation basin search';
    case "gsk_explore"
        options.max_iter = max(1, floor((options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
        options.knowledge_rate = get_option(options, 'knowledge_rate', 0.96);
        options.knowledge_factor = get_option(options, 'knowledge_factor', 0.72);
        result = SOP_agent_gsk(problem, seed, options);
        label = 'GSK exploration basin search';
    case "gsk"
        options.max_iter = max(1, floor((options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
        result = SOP_agent_gsk(problem, seed, options);
        label = 'GSK basin search';
    case "lshade_cma"
        result = SOP_agent_lshade_cma(problem, seed, options);
        label = 'L-SHADE-CMA basin search';
    case "lshade_jso"
        result = SOP_agent_lshade_jso(problem, seed, options);
        label = 'jSO/L-SHADE basin search';
    otherwise
        result = SOP_agent_lshade(problem, seed, options);
        label = 'L-SHADE basin search';
end
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
