function result = SOP_agent_cma_de_refine(problem, seed, options)
% CMA-ES basin finder followed by DE-family exploitation.
%
% This reverses the usual DE-then-CMA order: CMA-ES first builds a local
% covariance basin from the center, then L-SHADE/L-SHADE-CMA restarts inside
% that basin for objective-only exploitation.
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
base_fraction = get_option(options, 'base_fraction', 0.32);
refiner = lower(string(get_option(options, 'refiner', 'lshade_cma')));

cma_options = struct();
cma_options.population_num = get_option(options, 'cma_population_num', max(60, round(0.22 * get_option(options, 'population_num', 240))));
cma_options.max_fes = max(1000, floor(base_fraction * max_fes));
cma_options.max_runtime_sec = max(1, base_fraction * max_runtime_sec);
cma_options.initial_point = 0.5 * (problem.lb + problem.ub);
cma_options.sigma0 = get_option(options, 'sigma0', 0.003);
cma_options.restart_sigma = get_option(options, 'restart_sigma', 0.0012);
cma_options.restart_limit = get_option(options, 'restart_limit', 3);
cma_options.eig_interval = get_option(options, 'eig_interval', 6);
cma_options.verbose = false;
cma = SOP_agent_cma_es(problem, double(seed), cma_options);

refine_options = options;
refine_options.max_fes = max(1000, max_fes - cma.evaluation_count);
refine_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
refine_options.initial_point = cma.best_position;
refine_options.initial_radius = get_option(options, 'initial_radius', 0.010);
refine_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
refine_options.verbose = false;
switch refiner
    case "lshade"
        refine = SOP_agent_lshade(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE success-history adaptation';
    case "jso"
        refine_options.include_center = false;
        refine = SOP_agent_lshade_jso(problem, double(seed) + 7919, refine_options);
        refiner_label = 'jSO/L-SHADE staged parameter adaptation';
    otherwise
        refine_options.cma_rate = get_option(options, 'cma_rate', 0.12);
        refine_options.elite_rate = get_option(options, 'elite_rate', 0.20);
        refine_options.cma_interval = get_option(options, 'cma_interval', 16);
        refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE with elite covariance sampling';
end

if refine.record_value < cma.record_value
    result = refine;
else
    result = cma;
end
result.runtime = toc(t_start);
result.evaluation_count = cma.evaluation_count + refine.evaluation_count;
result.iteration = cma.iteration + refine.iteration;
result.convergence_curve = [cma.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [cma.raw_convergence_curve(:); refine.raw_convergence_curve(:)];
result.algorithm_combination = sprintf('Center-start CMA-ES basin finder\n%s basin exploitation', refiner_label);
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
