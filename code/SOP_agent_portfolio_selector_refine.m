function result = SOP_agent_portfolio_selector_refine(problem, seed, options)
% Portfolio selector: short scouts from distinct metaheuristics, then refine.
%
% The selector compares jSO/RSP, L-SHADE-CMA, and GSK scouts under short
% budgets. The best scout's basin is passed to the matching metaheuristic
% for the main exploitation phase.
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
scout_fraction = get_option(options, 'scout_fraction', 0.12);
scout_fes = max(1000, floor(scout_fraction * max_fes));
scout_time = max(1, scout_fraction * max_runtime_sec);

jso_options = jso_rsp_options(options);
jso_options.max_fes = scout_fes;
jso_options.max_runtime_sec = scout_time;
jso_options.verbose = false;
jso = SOP_agent_lshade_jso(problem, double(seed), jso_options);

cma_options = options;
cma_options.max_fes = scout_fes;
cma_options.max_runtime_sec = max(1, min(scout_time, max_runtime_sec - toc(t_start)));
cma_options.verbose = false;
lcma = SOP_agent_lshade_cma(problem, double(seed) + 3571, cma_options);

gsk_options = options;
gsk_options.max_runtime_sec = max(1, min(scout_time, max_runtime_sec - toc(t_start)));
gsk_options.max_iter = max(1, floor((scout_fes - NP) / NP));
gsk_options.knowledge_rate = get_option(options, 'gsk_knowledge_rate', 0.72);
gsk_options.knowledge_factor = get_option(options, 'gsk_knowledge_factor', 0.38);
gsk_options.verbose = false;
gsk = SOP_agent_gsk(problem, double(seed) + 7919, gsk_options);

scouts = {jso, lcma, gsk};
labels = {'jSO/RSP', 'L-SHADE-CMA', 'GSK'};
best_idx = 1;
for i = 2:numel(scouts)
    if scouts{i}.record_value < scouts{best_idx}.record_value
        best_idx = i;
    end
end
selected = scouts{best_idx};

remaining_fes = max(1000, max_fes - jso.evaluation_count - lcma.evaluation_count - gsk.evaluation_count);
remaining_time = max(1, max_runtime_sec - toc(t_start));
refine_options = options;
refine_options.max_fes = remaining_fes;
refine_options.max_runtime_sec = remaining_time;
refine_options.initial_point = selected.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.006);
refine_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
refine_options.verbose = false;
switch best_idx
    case 1
        refine_options = copy_jso_options(refine_options, jso_rsp_options(options));
        refine = SOP_agent_lshade_jso(problem, double(seed) + 104729, refine_options);
    case 2
        refine = SOP_agent_lshade_cma(problem, double(seed) + 104729, refine_options);
    otherwise
        refine_options.max_iter = max(1, floor((remaining_fes - NP) / NP));
        refine_options.knowledge_rate = get_option(options, 'gsk_knowledge_rate', 0.72);
        refine_options.knowledge_factor = get_option(options, 'gsk_knowledge_factor', 0.38);
        refine = SOP_agent_gsk(problem, double(seed) + 104729, refine_options);
end

result = selected;
if refine.record_value < result.record_value
    result = refine;
end
result.runtime = toc(t_start);
result.evaluation_count = jso.evaluation_count + lcma.evaluation_count + gsk.evaluation_count + refine.evaluation_count;
result.iteration = jso.iteration + lcma.iteration + gsk.iteration + refine.iteration;
result.convergence_curve = [jso.convergence_curve(:); lcma.convergence_curve(:); gsk.convergence_curve(:); refine.convergence_curve(:)];
result.algorithm_combination = sprintf('Portfolio selector scouts: jSO/RSP, L-SHADE-CMA, GSK\nSelected basin: %s\nMatching metaheuristic relay refinement', labels{best_idx});
result.combination_number = 4;
result.agent_id = 'Agent1';
end

function options = jso_rsp_options(options)
options.include_center = true;
options.ranked_r1 = true;
options.rank_pressure = 1.7;
options.mu_F_init = 0.34;
options.mu_CR_init = 0.86;
options.p_rate_start = 0.20;
options.p_rate_end = 0.040;
options.archive_factor_start = 1.8;
options.archive_factor_end = 3.2;
end

function target = copy_jso_options(target, source)
names = {'include_center', 'ranked_r1', 'rank_pressure', 'mu_F_init', 'mu_CR_init', ...
    'p_rate_start', 'p_rate_end', 'archive_factor_start', 'archive_factor_end'};
for i = 1:numel(names)
    target.(names{i}) = source.(names{i});
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
