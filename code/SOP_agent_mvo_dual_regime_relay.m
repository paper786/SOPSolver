function result = SOP_agent_mvo_dual_regime_relay(problem, seed, options)
% MVO two-regime best-point relay followed by L-SHADE/L-SHADE-CMA.
%
% The first MVO run searches globally. The second MVO run restarts around
% the first best point with a smaller Cauchy/Gaussian cloud. Only the better
% best point is handed to the DE-family refiner; the noisy MVO population is
% deliberately not preserved.
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
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
NP = get_option(options, 'population_num', 180);
refiner = lower(string(get_option(options, 'refiner', 'lshade_cma')));
global_fraction = get_option(options, 'global_mvo_fraction', 0.15);
local_fraction = get_option(options, 'local_mvo_fraction', 0.09);

global_options = options;
global_options.method = 'mvo';
global_options.population_num = NP;
global_options.max_fes = max(1000, floor(global_fraction * max_fes));
global_options.max_runtime_sec = max(1, global_fraction * max_runtime_sec);
global_options.max_iter = max(1, floor((global_options.max_fes - NP) / NP));
global_options.verbose = false;
global_mvo = SOP_agent_literature_swarm(problem, double(seed), global_options);

local_options = options;
local_options.method = 'mvo';
local_options.population_num = max(48, round(get_option(options, 'local_mvo_pop_rate', 0.62) * NP));
local_options.initial_point = global_mvo.best_position;
local_options.initial_radius = get_option(options, 'local_mvo_radius', 0.030);
local_options.initial_cauchy = get_option(options, 'local_mvo_cauchy', true);
local_options.max_fes = max(1000, floor(local_fraction * max_fes));
local_options.max_runtime_sec = max(1, min(max_runtime_sec - toc(t_start), local_fraction * max_runtime_sec));
local_options.max_iter = max(1, floor((local_options.max_fes - local_options.population_num) / local_options.population_num));
local_options.verbose = false;
local_mvo = SOP_agent_literature_swarm(problem, double(seed) + 3571, local_options);

if local_mvo.record_value < global_mvo.record_value
    handoff = local_mvo;
else
    handoff = global_mvo;
end

refine_options = options;
refine_options.initial_point = handoff.best_position;
refine_options.max_fes = max(1000, max_fes - global_mvo.evaluation_count - local_mvo.evaluation_count);
refine_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
refine_options.verbose = false;
switch refiner
    case "lshade"
        refine = SOP_agent_lshade(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE success-history adaptation';
    otherwise
        refine_options.cma_rate = get_option(options, 'cma_rate', 0.16);
        refine_options.elite_rate = get_option(options, 'elite_rate', 0.22);
        refine_options.cma_interval = get_option(options, 'cma_interval', 12);
        refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE with elite covariance sampling';
end

result = handoff;
if refine.record_value < result.record_value
    result = refine;
end
result.runtime = toc(t_start);
result.evaluation_count = global_mvo.evaluation_count + local_mvo.evaluation_count + refine.evaluation_count;
result.iteration = global_mvo.iteration + local_mvo.iteration + refine.iteration;
result.convergence_curve = [global_mvo.convergence_curve(:); local_mvo.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [global_mvo.raw_convergence_curve(:); local_mvo.raw_convergence_curve(:); raw_curve_for(refine)];
result.algorithm_combination = sprintf('MVO global basin discovery\nMVO local exploit best-point check\n%s', refiner_label);
result.combination_number = 3;
result.agent_id = 'Agent2';
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

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
