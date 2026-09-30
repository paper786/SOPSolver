function result = SOP_agent_gsk_relay(problem, seed, options)
% Two-stage pure GSK relay: broad knowledge sharing then exploitative GSK.
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
explore_fraction = get_option(options, 'explore_fraction', 0.34);

explore_options = options;
explore_options.max_fes = max(1000, floor(explore_fraction * max_fes));
explore_options.max_runtime_sec = max(1, explore_fraction * max_runtime_sec);
explore_options.max_iter = max(1, floor((explore_options.max_fes - NP) / NP));
explore_options.knowledge_rate = get_option(options, 'explore_knowledge_rate', 0.96);
explore_options.knowledge_factor = get_option(options, 'explore_knowledge_factor', 0.72);
explore_options.verbose = false;
explore = SOP_agent_gsk(problem, double(seed), explore_options);

exploit_options = options;
exploit_options.max_fes = max(1000, max_fes - explore.evaluation_count);
exploit_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
exploit_options.max_iter = max(1, floor((exploit_options.max_fes - NP) / NP));
exploit_options.knowledge_rate = get_option(options, 'exploit_knowledge_rate', 0.72);
exploit_options.knowledge_factor = get_option(options, 'exploit_knowledge_factor', 0.38);
exploit_options.initial_point = explore.best_position;
exploit_options.initial_radius = get_option(options, 'initial_radius', 0.018);
exploit_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
exploit_options.verbose = false;
exploit = SOP_agent_gsk(problem, double(seed) + 7919, exploit_options);

if exploit.record_value < explore.record_value
    result = exploit;
else
    result = explore;
end
result.runtime = toc(t_start);
result.evaluation_count = explore.evaluation_count + exploit.evaluation_count;
result.iteration = explore.iteration + exploit.iteration;
result.convergence_curve = [explore.convergence_curve(:); exploit.convergence_curve(:)];
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK) explore phase\nGaining-Sharing Knowledge (GSK) exploit relay restart');
result.combination_number = 2;
result.agent_id = 'Agent2';
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
