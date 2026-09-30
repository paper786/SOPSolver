function result = SOP_agent_lshade_hses_refine(problem, seed, options)
% L-SHADE-family search followed by a short HSES distribution sampler.
%
% This keeps the proven L-SHADE/CC basin finder intact, then uses HS-ES
% covariance/univariate sampling only as a bounded metaheuristic relay.
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
hses_reserve_fes = get_option(options, 'hses_reserve_fes', max(1000, floor(0.10 * max_fes)));
hses_reserve_fes = max(1000, min(hses_reserve_fes, max(1000, max_fes - 1000)));

base_options = options;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 0.80 * max_runtime_sec));
base_options.max_fes = max(1000, max_fes - hses_reserve_fes);
base_options.verbose = false;
base = SOP_agent_lshade_cc_refine(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
if remaining_time <= 1 || remaining_fes < 1000
    result = base;
    result.algorithm_combination = sprintf('%s\nShort HSES relay skipped by budget/time limit', base.algorithm_combination);
    return;
end

hses_options = struct();
hses_options.population_num = get_option(options, 'hses_population_num', max(28, round(0.18 * get_option(options, 'population_num', 180))));
hses_options.max_runtime_sec = min(remaining_time, get_option(options, 'hses_runtime_sec', remaining_time));
hses_options.max_fes = min(remaining_fes, hses_reserve_fes);
hses_options.initial_point = base.best_position;
if isfield(base, 'final_population') && ~isempty(base.final_population)
    hses_options.initial_population = base.final_population;
end
hses_options.initial_radius = get_option(options, 'hses_initial_radius', 0.0018);
hses_options.sigma0 = get_option(options, 'hses_sigma0', 0.00085);
hses_options.reset_sigma = get_option(options, 'hses_reset_sigma', 0.0028);
hses_options.covariance_sample_rate = get_option(options, 'hses_covariance_sample_rate', 0.44);
hses_options.univariate_sample_rate = get_option(options, 'hses_univariate_sample_rate', 0.40);
hses_options.hybrid_mask_rate = get_option(options, 'hses_hybrid_mask_rate', 0.38);
hses_options.cov_scale = get_option(options, 'hses_cov_scale', 0.74);
hses_options.uni_scale = get_option(options, 'hses_uni_scale', 0.96);
hses_options.best_blend = get_option(options, 'hses_best_blend', 0.26);
hses_options.elite_recomb_rate = get_option(options, 'hses_elite_recomb_rate', 0.18);
hses_options.best_pull_rate = get_option(options, 'hses_best_pull_rate', 0.22);
hses_options.verbose = false;
hses = SOP_agent_hses_sampling(problem, double(seed) + 104729, hses_options);

result = base;
if hses.record_value < result.record_value
    result.best_value = hses.best_value;
    result.record_value = hses.record_value;
    result.best_position = hses.best_position;
end
result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + hses.evaluation_count;
result.iteration = base.iteration + hses.iteration;
result.raw_convergence_curve = [raw_curve_for(base); raw_curve_for(hses)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.algorithm_combination = sprintf('%s\nHybrid Sampling Evolution Strategy (HS-ES/HSES) covariance/univariate relay', ...
    base.algorithm_combination);
result.combination_number = 5;
result.agent_id = 'Agent2';
end

function curve = raw_curve_for(item)
if isfield(item, 'raw_convergence_curve') && ~isempty(item.raw_convergence_curve)
    curve = item.raw_convergence_curve(:);
elseif isfield(item, 'convergence_curve') && ~isempty(item.convergence_curve)
    curve = item.convergence_curve(:);
else
    curve = item.best_value;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
