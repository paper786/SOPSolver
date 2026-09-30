function result = SOP_agent_clpso_pingpong_refine(problem, seed, options)
% CLPSO scout followed by L-SHADE-CMA/CMA-ES/L-SHADE-CMA ping-pong.
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
scout_fraction = get_option(options, 'scout_fraction', 0.14);

scout_options = options;
scout_options.method = 'clpso';
scout_options.max_fes = max(1000, floor(scout_fraction * max_fes));
scout_options.max_runtime_sec = max(1, scout_fraction * max_runtime_sec);
scout_options.verbose = false;
scout = SOP_agent_literature_swarm(problem, double(seed), scout_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
refine_options = options;
refine_options.max_fes = max(1000, max_fes - scout.evaluation_count);
refine_options.max_runtime_sec = remaining_time;
refine_options.base_algorithm = 'lshade_cma';
refine_options.base_fraction = min(0.86, 1000000 / max(1000000, refine_options.max_fes));
refine_options.initial_point = scout.best_position;
if isfield(scout, 'final_population') && ~isempty(scout.final_population)
    refine_options.initial_population = scout.final_population;
end
refine_options.local_sigma = get_option(options, 'local_sigma', 0.00025);
refine_options.restart_sigma = get_option(options, 'restart_sigma', 0.00012);
refine_options.restart_limit = get_option(options, 'restart_limit', 3);
refine_options.cma_population_num = get_option(options, 'cma_population_num', max(28, round(0.18 * get_option(options, 'population_num', 180))));
refine_options.lshade_cma_after = true;
refine_options.relay_population_num = get_option(options, 'relay_population_num', max(44, round(0.20 * get_option(options, 'population_num', 180))));
refine_options.relay_radius = get_option(options, 'relay_radius', 0.0018);
refine_options.relay_cauchy = true;
refine_options.relay_cma_rate = get_option(options, 'relay_cma_rate', 0.14);
refine_options.relay_elite_rate = get_option(options, 'relay_elite_rate', 0.20);
refine_options.relay_cma_interval = get_option(options, 'relay_cma_interval', 10);
refine = SOP_agent_lshade_cmaes_refine(problem, double(seed) + 7919, refine_options);

if refine.record_value < scout.record_value
    result = refine;
else
    result = scout;
end
result.runtime = toc(t_start);
result.evaluation_count = scout.evaluation_count + refine.evaluation_count;
result.iteration = scout.iteration + refine.iteration;
result.convergence_curve = [scout.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [raw_curve_for(scout); raw_curve_for(refine)];
result.algorithm_combination = sprintf('%s\n%s', scout.algorithm_combination, refine.algorithm_combination);
result.combination_number = 4;
result.agent_id = 'Agent2';
end

function curve = raw_curve_for(item)
if isfield(item, 'raw_convergence_curve') && ~isempty(item.raw_convergence_curve)
    curve = item.raw_convergence_curve(:);
elseif isfield(item, 'convergence_curve') && ~isempty(item.convergence_curve)
    curve = item.convergence_curve(:);
else
    curve = item.record_value;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
