function result = SOP_agent_swarm_chain_refine(problem, seed, options)
% Literature-swarm chain followed by L-SHADE-CMA exploitation.
%
% Stage 1 uses one literature metaheuristic as a basin explorer. Stage 2
% restarts a second literature metaheuristic near the best basin to perturb
% or teach around that point. Stage 3 starts L-SHADE-CMA from the best point.
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
explorer = lower(string(get_option(options, 'explorer', 'hgs')));
puncture = lower(string(get_option(options, 'puncture', 'rime')));
explore_fraction = get_option(options, 'explore_fraction', 0.20);
puncture_fraction = get_option(options, 'puncture_fraction', 0.16);

explore_options = options;
explore_options.method = char(explorer);
explore_options.max_fes = max(1000, floor(explore_fraction * max_fes));
explore_options.max_runtime_sec = max(1, explore_fraction * max_runtime_sec);
explore_options.max_iter = max(1, floor((explore_options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
explore_options.verbose = false;
explore = SOP_agent_literature_swarm(problem, double(seed), explore_options);

puncture_options = options;
puncture_options.method = char(puncture);
puncture_options.initial_point = explore.best_position;
puncture_options.initial_radius = get_option(options, 'puncture_radius', 0.025);
puncture_options.initial_cauchy = get_option(options, 'puncture_cauchy', true);
puncture_options.max_fes = max(1000, floor(puncture_fraction * max_fes));
puncture_options.max_runtime_sec = max(1, min(max_runtime_sec - toc(t_start), puncture_fraction * max_runtime_sec));
puncture_options.max_iter = max(1, floor((puncture_options.max_fes - get_option(options, 'population_num', 180)) / get_option(options, 'population_num', 180)));
puncture_options.verbose = false;
if puncture == "quantum_snap"
    puncture_options.profile = get_option(options, 'quantum_profile', 'snap_focus');
    puncture_options.include_center = true;
    puncture_result = SOP_agent_quantum_snap_de(problem, double(seed) + 3571, puncture_options);
else
    puncture_result = SOP_agent_literature_swarm(problem, double(seed) + 3571, puncture_options);
end

if puncture_result.record_value < explore.record_value
    best_phase = puncture_result;
else
    best_phase = explore;
end

refine_options = options;
refine_options.initial_point = best_phase.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.006);
refine_options.initial_cauchy = get_option(options, 'refine_cauchy', false);
refine_options.max_fes = max(1000, max_fes - explore.evaluation_count - puncture_result.evaluation_count);
refine_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
refine_options.cma_rate = get_option(options, 'cma_rate', 0.18);
refine_options.elite_rate = get_option(options, 'elite_rate', 0.24);
refine_options.cma_interval = get_option(options, 'cma_interval', 12);
refine_options.verbose = false;
refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);

result = best_phase;
if refine.record_value < result.record_value
    result = refine;
end
result.runtime = toc(t_start);
result.evaluation_count = explore.evaluation_count + puncture_result.evaluation_count + refine.evaluation_count;
result.iteration = explore.iteration + puncture_result.iteration + refine.iteration;
result.convergence_curve = [explore.convergence_curve(:); puncture_result.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [explore.raw_convergence_curve(:); puncture_result.raw_convergence_curve(:); refine.raw_convergence_curve(:)];
result.algorithm_combination = sprintf('%s explorer\n%s basin puncture/teaching restart\nL-SHADE-CMA objective-only exploitation', ...
    method_label(explorer), method_label(puncture));
result.combination_number = 3;
result.agent_id = 'Agent2';
end

function label = method_label(method)
switch method
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
    case "sma"
        label = 'Slime Mould Algorithm (SMA)';
    case "hgs"
        label = 'Hunger Games Search (HGS)';
    case "tlbo"
        label = 'Teaching-Learning-Based Optimization (TLBO)';
    case "gsk"
        label = 'Gaining-Sharing Knowledge (GSK)';
    case "quantum_snap"
        label = 'Quantum/opposition snap Differential Evolution';
    otherwise
        label = upper(char(method));
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
