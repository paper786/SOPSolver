function result = SOP_agent_gsk_de_hybrid(problem, seed, options)
% GSK basin finder followed by L-SHADE/L-SHADE-CMA refinement.
%
% This combines the Gaining-Sharing Knowledge population learning mechanism
% with a Differential Evolution refiner. It is used when GSK reaches a good
% basin but needs a different local search dynamic.
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
gsk_fraction = get_option(options, 'gsk_fraction', 0.72);
refiner = lower(string(get_option(options, 'refiner', 'lshade_cma')));

gsk_options = options;
gsk_options.max_runtime_sec = gsk_fraction * max_runtime_sec;
gsk_options.max_iter = max(1, floor(gsk_fraction * get_option(options, 'max_iter', floor(max_fes / get_option(options, 'population_num', 240)))));
gsk_options.verbose = false;
gsk_result = SOP_agent_gsk(problem, double(seed), gsk_options);

remaining_time = max_runtime_sec - toc(t_start);
refine_options = options;
refine_options.max_runtime_sec = max(1, remaining_time);
refine_options.max_fes = max(1000, max_fes - gsk_result.evaluation_count);
refine_options.initial_point = gsk_result.best_position;
refine_radius = get_option(options, 'refine_radius', []);
if ~isempty(refine_radius)
    refine_options.initial_radius = refine_radius;
    refine_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
end
refine_options.verbose = false;
switch refiner
    case "lshade"
        refine_result = SOP_agent_lshade(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE success-history adaptation';
    case "lshade_eig"
        refine_result = SOP_agent_lshade_eig(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE with eigen-coordinate crossover';
    otherwise
        refine_result = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE with elite covariance sampling';
end

if refine_result.record_value < gsk_result.record_value
    result = refine_result;
else
    result = gsk_result;
end
result.runtime = toc(t_start);
result.evaluation_count = gsk_result.evaluation_count + refine_result.evaluation_count;
result.iteration = gsk_result.iteration + refine_result.iteration;
result.convergence_curve = [gsk_result.convergence_curve(:); refine_result.convergence_curve(:)];
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK)\n%s', refiner_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
