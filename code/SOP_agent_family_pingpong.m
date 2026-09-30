function result = SOP_agent_family_pingpong(problem, seed, options)
% Multi-stage metaheuristic ping-pong between DE-family optimizers.
%
% This is a staged island-like schedule: each stage runs a real optimizer
% with its own evaluation budget, then passes elite population material to
% the next stage. It avoids gradient or deterministic numerical guidance.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
NP = get_option(options, 'population_num', 180);
stage_algorithms = get_option(options, 'stage_algorithms', {'jso_rsp', 'lshade_cma', 'jso_rsp', 'lshade_cma'});
stage_fes_rates = get_option(options, 'stage_fes_rates', ones(1, numel(stage_algorithms)) ./ numel(stage_algorithms));
stage_time_rates = get_option(options, 'stage_time_rates', stage_fes_rates);
stage_pop_rates = get_option(options, 'stage_pop_rates', ones(1, numel(stage_algorithms)));
stage_radius = get_option(options, 'stage_radius', 0.004);

eval_total = 0;
iter_total = 0;
curve = [];
raw_curve = [];
best_raw = inf;
best_record = inf;
best_position = [];
previous = [];
best_stage = [];
last_stage = [];

for s = 1:numel(stage_algorithms)
    remaining_fes = max_fes - eval_total;
    remaining_time = max_runtime_sec - toc(t_start);
    if remaining_fes <= 0 || remaining_time <= 0
        break;
    end

    if s < numel(stage_algorithms)
        stage_fes = max(1000, floor(stage_fes_rates(s) * max_fes));
    else
        stage_fes = remaining_fes;
    end
    stage_options = options;
    stage_options.max_fes = min(remaining_fes, stage_fes);
    stage_options.max_runtime_sec = max(1, min(remaining_time, stage_time_rates(min(s, numel(stage_time_rates))) * max_runtime_sec));
    stage_options.population_num = max(30, round(stage_pop_rates(min(s, numel(stage_pop_rates))) * NP));
    stage_options.verbose = false;
    if ~isempty(best_position)
        stage_options.initial_point = best_position;
        stage_options.initial_radius = pick_stage_radius(stage_radius, s);
        stage_options.initial_cauchy = true;
    end
    if ~isempty(previous)
        stage_options.initial_population = migration_population(previous, best_position, stage_options.population_num, ...
            problem.lb, problem.ub, get_option(options, 'migration_radius', 0.004), ...
            get_option(options, 'migration_block_rate', 0.10), get_option(options, 'migration_de_weight', 0.16));
        stage_options.preserve_initial_population_after_radius = true;
    end

    stage_key = lower(string(stage_algorithms{s}));
    stage = run_stage(problem, double(seed) + 1009 * s, stage_key, stage_options);
    eval_total = eval_total + stage.evaluation_count;
    iter_total = iter_total + stage.iteration;
    curve = [curve; stage.convergence_curve(:)]; %#ok<AGROW>
    raw_curve = [raw_curve; raw_curve_for(stage)]; %#ok<AGROW>
    last_stage = stage;
    previous = stage;
    if stage.record_value < best_record
        best_record = stage.record_value;
        best_raw = stage.best_value;
        best_position = stage.best_position;
        best_stage = stage;
    end
end

if isempty(best_stage)
    error('SOP_agent_family_pingpong:NoStage', 'No ping-pong stage was executed.');
end
runtime = toc(t_start);
if isempty(last_stage)
    last_stage = best_stage;
end
result = best_stage;
result.best_value = best_raw;
result.record_value = best_record;
result.best_position = best_position;
result.convergence_curve = curve;
result.raw_convergence_curve = raw_curve;
result.runtime = runtime;
result.evaluation_count = eval_total;
result.iteration = iter_total;
result.population_num = NP;
result.final_population = get_result_population(last_stage);
result.final_fitness = get_result_fitness(last_stage);
result.algorithm_combination = sprintf('Staged metaheuristic ping-pong\n%s', stage_list_label(stage_algorithms));
result.combination_number = 5;
result.agent_id = 'Agent2';
result.problem = problem;
end

function stage = run_stage(problem, seed, stage_key, options)
switch stage_key
    case "jso_rsp"
        options.include_center = false;
        options.ranked_r1 = true;
        options.rank_pressure = get_option(options, 'rank_pressure', 1.7);
        options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
        options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
        options.p_rate_start = get_option(options, 'p_rate_start', 0.20);
        options.p_rate_end = get_option(options, 'p_rate_end', 0.040);
        options.weight_start = get_option(options, 'weight_start', 0.62);
        options.weight_end = get_option(options, 'weight_end', 1.36);
        options.archive_factor_start = get_option(options, 'archive_factor_start', 1.8);
        options.archive_factor_end = get_option(options, 'archive_factor_end', 3.2);
        stage = SOP_agent_lshade_jso(problem, seed, options);
    case "lshade_cma"
        options.cma_rate = get_option(options, 'cma_rate', 0.16);
        options.elite_rate = get_option(options, 'elite_rate', 0.22);
        options.cma_interval = get_option(options, 'cma_interval', 12);
        stage = SOP_agent_lshade_cma(problem, seed, options);
    case "lshade"
        stage = SOP_agent_lshade(problem, seed, options);
    otherwise
        error('SOP_agent_family_pingpong:BadStage', 'Unknown stage %s.', stage_key);
end
end

function population = migration_population(previous, best_position, NP, lb, ub, radius_scale, block_rate, de_weight)
D = numel(lb);
span = ub - lb;
elites = get_result_population(previous);
if isempty(best_position)
    best_position = previous.best_position;
end
if isempty(elites)
    elites = best_position;
end
population = repmat(best_position, NP, 1);
keep = min(size(elites, 1), max(2, round(0.35 * NP)));
population(1:keep, :) = elites(1:keep, :);
radius = radius_scale .* span;
for i = keep + 1:NP
    donor = elites(randi(size(elites, 1)), :);
    child = best_position;
    mask = rand(1, D) < block_rate;
    if ~any(mask)
        mask(randperm(D, max(1, round(block_rate * D)))) = true;
    end
    child(mask) = donor(mask);
    if rand() < 0.55 && size(elites, 1) >= 2
        a = elites(randi(size(elites, 1)), :);
        b = elites(randi(size(elites, 1)), :);
        child(mask) = child(mask) + de_weight .* (a(mask) - b(mask));
    end
    if rand() < 0.25
        child(mask) = child(mask) + randn(1, nnz(mask)) .* radius(mask);
    end
    population(i, :) = min(max(child, lb), ub);
end
population = min(max(population, lb), ub);
end

function radius = pick_stage_radius(stage_radius, stage_index)
if numel(stage_radius) >= stage_index
    radius = stage_radius(stage_index);
else
    radius = stage_radius(end);
end
end

function population = get_result_population(result)
if isfield(result, 'final_population') && ~isempty(result.final_population)
    population = result.final_population;
elseif isfield(result, 'best_position') && ~isempty(result.best_position)
    population = result.best_position;
else
    population = [];
end
end

function fitness = get_result_fitness(result)
if isfield(result, 'final_fitness') && ~isempty(result.final_fitness)
    fitness = result.final_fitness;
else
    fitness = [];
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

function label = stage_list_label(stage_algorithms)
label = '';
for i = 1:numel(stage_algorithms)
    if i > 1
        label = sprintf('%s -> ', label);
    end
    label = sprintf('%s%s', label, char(stage_algorithms{i}));
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
