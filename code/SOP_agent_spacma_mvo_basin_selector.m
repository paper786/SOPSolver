function result = SOP_agent_spacma_mvo_basin_selector(problem, seed, options)
% Independent whole-basin SPACMA and MVO selector.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 20000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
spacma_fes = min(10000 * problem.dimension, max_fes - 1000);

spacma_options = struct();
spacma_options.population_num = get_option(options, 'spacma_population_num', 18 * problem.dimension);
spacma_options.min_population_num = 4;
spacma_options.max_fes = spacma_fes;
spacma_options.max_runtime_sec = max(1, min(0.56 * max_runtime_sec, max_runtime_sec - 2));
spacma_options.memory_size = 5;
spacma_options.p_rate = 0.11;
spacma_options.archive_rate = 1.4;
spacma_options.class_learning_rate = 0.8;
spacma_options.sigma_init = 0.5;
spacma_options.verbose = false;
spacma = SOP_agent_lshade_spacma_faithful(problem, double(seed), spacma_options);

remaining_fes = max(1000, max_fes - spacma.evaluation_count);
mvo_options = struct();
mvo_options.method = 'mvo';
mvo_options.population_num = get_option(options, 'mvo_population_num', 260);
mvo_options.max_fes = remaining_fes;
mvo_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
mvo_options.max_iter = max(1, floor((remaining_fes - mvo_options.population_num) / mvo_options.population_num));
mvo_options.verbose = false;
mvo = SOP_agent_literature_swarm(problem, double(seed) + 65537, mvo_options);

if spacma.record_value <= mvo.record_value
    result = spacma;
    selected_label = 'L-SHADE-SPACMA';
else
    result = mvo;
    selected_label = 'Multi-Verse Optimizer';
end
result.runtime = toc(t_start);
result.evaluation_count = spacma.evaluation_count + mvo.evaluation_count;
result.iteration = spacma.iteration + mvo.iteration;
result.convergence_curve = [spacma.convergence_curve(:); mvo.convergence_curve(:)];
result.raw_convergence_curve = [raw_curve_for(spacma); raw_curve_for(mvo)];
result.population_num = spacma_options.population_num + mvo_options.population_num;
result.algorithm_combination = sprintf(['Independent L-SHADE-SPACMA whole-basin search\n' ...
    'Independent Multi-Verse Optimizer whole-basin search\n' ...
    'Objective-based selector retained %s'], selected_label);
result.combination_number = 3;
result.agent_id = 'EvaluateAgent';
result.problem = problem;
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
else
    curve = result.best_value;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
