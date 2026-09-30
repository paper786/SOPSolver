function result = SOP_agent_swarm_selector_refine(problem, seed, options)
% Literature swarm scouts followed by one DE/CMA-family exploitation run.
%
% The scouts are run independently under short budgets. Only the best basin
% is passed as the initial point to the main metaheuristic refiner.
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
scout_a = lower(string(get_option(options, 'scout_a', 'hho')));
scout_b = lower(string(get_option(options, 'scout_b', 'gwo')));
scout_methods = get_option(options, 'scout_methods', {});
if isempty(scout_methods)
    scout_methods = {char(scout_a), char(scout_b)};
elseif isstring(scout_methods)
    scout_methods = cellstr(scout_methods);
elseif ischar(scout_methods)
    scout_methods = cellstr(split(string(scout_methods), ','));
end
for i = 1:numel(scout_methods)
    scout_methods{i} = char(lower(string(strtrim(scout_methods{i}))));
end
scout_fraction = get_option(options, 'scout_fraction', 0.12);

scout_results = cell(1, numel(scout_methods));
seed_offsets = [0, 3571, 6151, 7919, 104729, 131071, 524287, 8191];
for s = 1:numel(scout_methods)
    scout_options = options;
    scout_options.method = scout_methods{s};
    scout_options.max_fes = max(1000, floor(scout_fraction * max_fes));
    scout_options.max_runtime_sec = max(1, min(max_runtime_sec - toc(t_start), scout_fraction * max_runtime_sec));
    scout_options.max_iter = max(1, floor((scout_options.max_fes - NP) / NP));
    scout_options.verbose = false;
    scout_results{s} = SOP_agent_literature_swarm(problem, double(seed) + seed_offsets(1 + mod(s - 1, numel(seed_offsets))), scout_options);
end

best_idx = 1;
for s = 2:numel(scout_results)
    if scout_results{s}.record_value < scout_results{best_idx}.record_value
        best_idx = s;
    end
end
selected = scout_results{best_idx};
selected_label = method_label(scout_methods{best_idx});

refine_options = options;
refine_options.initial_point = selected.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.006);
refine_options.initial_cauchy = get_option(options, 'refine_cauchy', false);
refine_options.max_fes = max(1000, max_fes - sum_cell_eval(scout_results));
refine_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
refine_options.cma_rate = get_option(options, 'cma_rate', 0.18);
refine_options.elite_rate = get_option(options, 'elite_rate', 0.24);
refine_options.cma_interval = get_option(options, 'cma_interval', 12);
refine_options.verbose = false;
refiner = lower(string(get_option(options, 'refiner', 'lshade_cma')));
switch refiner
    case "jso"
        refine_options.ranked_r1 = true;
        refine_options.rank_pressure = get_option(options, 'rank_pressure', 1.7);
        refine_options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
        refine_options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
        refine_options.p_rate_start = get_option(options, 'p_rate_start', 0.20);
        refine_options.p_rate_end = get_option(options, 'p_rate_end', 0.040);
        refine_options.archive_factor_start = get_option(options, 'archive_factor_start', 1.8);
        refine_options.archive_factor_end = get_option(options, 'archive_factor_end', 3.2);
        refine = SOP_agent_lshade_jso(problem, double(seed) + 7919, refine_options);
        refiner_label = 'jSO/L-SHADE ranked-r1 exploitation';
    otherwise
        refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE-CMA exploitation';
end

result = selected;
if refine.record_value < result.record_value
    result = refine;
end
result.runtime = toc(t_start);
result.evaluation_count = sum_cell_eval(scout_results) + refine.evaluation_count;
result.iteration = sum_cell_iter(scout_results) + refine.iteration;
result.convergence_curve = [cell_curve_concat(scout_results, false); refine.convergence_curve(:)];
result.raw_convergence_curve = [cell_curve_concat(scout_results, true); raw_curve_for(refine)];
scout_text = join_labels(scout_methods);
result.algorithm_combination = sprintf('%s short scouts\n%s selected basin\n%s', ...
    scout_text, selected_label, refiner_label);
result.combination_number = 3;
result.agent_id = 'Agent2';
end

function label = method_label(method)
switch method
    case "hho"
        label = 'Harris Hawks Optimization (HHO)';
    case "gwo"
        label = 'Grey Wolf Optimizer (GWO)';
    case "wso"
        label = 'White Shark Optimizer (WSO)';
    case "mpa"
        label = 'Marine Predators Algorithm (MPA)';
    case "mvo"
        label = 'Multi-Verse Optimizer (MVO)';
    case "avoa"
        label = 'African Vultures Optimization Algorithm (AVOA)';
    case "rime"
        label = 'RIME Optimization Algorithm';
    case "ao"
        label = 'Aquila Optimizer (AO)';
    case "woa"
        label = 'Whale Optimization Algorithm (WOA)';
    case "eo"
        label = 'Equilibrium Optimizer (EO)';
    case "sma"
        label = 'Slime Mould Algorithm (SMA)';
    case "hgs"
        label = 'Hunger Games Search (HGS)';
    case "tlbo"
        label = 'Teaching-Learning-Based Optimization (TLBO)';
    otherwise
        label = upper(char(method));
end
end

function text = join_labels(methods)
labels = cell(1, numel(methods));
for i = 1:numel(methods)
    labels{i} = method_label(methods{i});
end
text = strjoin(labels, ', ');
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

function count = sum_cell_eval(results)
count = 0;
for i = 1:numel(results)
    count = count + results{i}.evaluation_count;
end
end

function count = sum_cell_iter(results)
count = 0;
for i = 1:numel(results)
    count = count + results{i}.iteration;
end
end

function curve = cell_curve_concat(results, use_raw)
curve = [];
for i = 1:numel(results)
    if use_raw
        part = raw_curve_for(results{i});
    elseif isfield(results{i}, 'convergence_curve') && ~isempty(results{i}.convergence_curve)
        part = results{i}.convergence_curve(:);
    else
        part = results{i}.record_value;
    end
    curve = [curve; part(:)]; %#ok<AGROW>
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
