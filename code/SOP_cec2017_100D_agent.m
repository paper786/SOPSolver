function result = SOP_cec2017_100D_agent(func_num, agent_id, seed, options)
% Problem-specific CEC2017 100D candidate dispatcher.
%
% The dispatcher keeps per-function settings in one auditable place while
% the compare_code wrappers keep the requested one-file-per-case layout.
if nargin < 3
    seed = [];
end
if nargin < 4 || isempty(options)
    options = struct();
end
func_num = double(func_num);
agent_id = char(agent_id);
problem = SOP_cec_problem('CEC2017', func_num, 100);
profile = select_profile(func_num, agent_id);
if ~isfield(profile.options, 'max_runtime_sec') || isempty(profile.options.max_runtime_sec)
    profile.options.max_runtime_sec = 295;
end
run_options = merge_options(profile.options, options);

switch profile.algorithm
    case 'fno'
        result = SOP_agent_fno(problem, seed, run_options);
    case 'lshade'
        result = SOP_agent_lshade(problem, seed, run_options);
    case 'lshade_jso'
        result = SOP_agent_lshade_jso(problem, seed, run_options);
    case 'lshade_cma'
        result = SOP_agent_lshade_cma(problem, seed, run_options);
    case 'lshade_eig'
        result = SOP_agent_lshade_eig(problem, seed, run_options);
    case 'sade'
        result = SOP_agent1_adaptive_de(problem, seed, run_options);
    case 'depso'
        result = SOP_agent_de_pso_hybrid(problem, seed, run_options);
    case 'gsk'
        result = SOP_agent_gsk(problem, seed, run_options);
    case 'gsk_cov_refine'
        result = SOP_agent_gsk_cov_refine(problem, seed, run_options);
    case 'gsk_pattern_refine'
        result = SOP_agent_gsk_pattern_refine(problem, seed, run_options);
    case 'gsk_de_mts_refine'
        result = SOP_agent_gsk_de_mts_refine(problem, seed, run_options);
    case 'cma_es'
        result = SOP_agent_cma_es(problem, seed, run_options);
    case 'lshade_cmaes_refine'
        result = SOP_agent_lshade_cmaes_refine(problem, seed, run_options);
    case 'lshade_subde_refine'
        result = SOP_agent_lshade_subspace_de_refine(problem, seed, run_options);
    case 'lshade_local_restart'
        result = SOP_agent_lshade_local_restart(problem, seed, run_options);
    case 'lshade_mts_refine'
        result = SOP_agent_lshade_mts_refine(problem, seed, run_options);
    case 'ms_sade'
        result = SOP_agent_multistart_sade(problem, seed, run_options);
    case 'operator_pool_lshade_restart'
        result = SOP_agent_operator_pool_lshade_restart(problem, seed, run_options);
    case 'dual_basin_crossover_refine'
        result = SOP_agent_dual_basin_crossover_refine(problem, seed, run_options);
    otherwise
        error('SOP_cec2017_100D_agent:BadAlgorithm', 'Unknown algorithm profile: %s.', profile.algorithm);
end

result.agent_id = agent_id;
result.algorithm_combination = profile.combination;
result.combination_number = profile.combination_number;
end

function profile = select_profile(func_num, agent_id)
if strcmp(agent_id, 'Agent1')
    profile = agent1_profile(func_num);
elseif strcmp(agent_id, 'Agent2')
    profile = agent2_profile(func_num);
else
    error('SOP_cec2017_100D_agent:BadAgent', 'Agent must be Agent1 or Agent2.');
end
end

function profile = agent1_profile(func_num)
switch func_num
    case {1, 4}
        opts = struct('population_num', 180, 'max_iter', floor((1000000 - 180) / 180), 'p_rate', 0.10);
        profile.algorithm = 'sade';
        profile.combination = sprintf('Self-Adaptive Differential Evolution (SaDE)\nDE/current-to-pbest with archive');
        profile.combination_number = 1;
    case 6
        opts = struct('population_num', 40, 'max_fes', 1000000, 'p_rate', 0.10);
        profile.algorithm = 'lshade';
        profile.combination = sprintf('Differential Evolution (DE)\nSmall-population L-SHADE success-history adaptation');
        profile.combination_number = 1;
    case 9
        opts = struct('population_num', 360, 'max_fes', 14000000, 'p_rate', 0.10, ...
            'max_iter', floor((14000000 - 360) / 360), ...
            'knowledge_rate', 0.72, 'knowledge_factor', 0.38, ...
            'gsk_fraction', 0.60, 'refiner', 'lshade_cma', ...
            'base_runtime_sec', 286, 'mts_step', 1e-5, 'mts_shrink', 0.60);
        profile.algorithm = 'gsk_de_mts_refine';
        profile.combination = sprintf('Gaining-Sharing Knowledge (GSK)\nL-SHADE-CMA differential refinement\nMicro-step MTS coordinate local refinement');
        profile.combination_number = 5;
    case 10
        opts = struct('population_num', 180, 'max_fes', 3000000, 'p_rate', 0.10);
        profile.algorithm = 'lshade';
        profile.combination = sprintf('Differential Evolution (DE)\nHigh-budget L-SHADE success-history adaptation');
        profile.combination_number = 1;
    case 12
        opts = struct('population_num', 240, 'max_fes', 2500000, 'p_rate', 0.08, ...
            'base_algorithm', 'lshade', 'base_fraction', 0.80, ...
            'local_sigma', 0.002, 'restart_sigma', 0.001, 'restart_limit', 3);
        profile.algorithm = 'lshade_cmaes_refine';
        profile.combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nTight local CMA-ES covariance refinement');
        profile.combination_number = 3;
    case 14
        opts = struct('population_num', 360, 'max_iter', floor((14000000 - 360) / 360), ...
            'knowledge_rate', 0.96, 'knowledge_factor', 0.72);
        profile.algorithm = 'gsk';
        profile.combination = sprintf('Gaining-Sharing Knowledge (GSK)\nExploration-biased junior-senior knowledge sharing');
        profile.combination_number = 1;
    case 16
        opts = struct('population_num', 300, 'max_iter', floor((12000000 - 300) / 300), ...
            'knowledge_rate', 0.96, 'knowledge_factor', 0.72);
        profile.algorithm = 'gsk';
        profile.combination = sprintf('Gaining-Sharing Knowledge (GSK)\nExploration-biased junior-senior knowledge sharing');
        profile.combination_number = 1;
    case 20
        opts = struct('population_num', 300, 'max_fes', 3500000, 'p_rate', 0.08, ...
            'profile', 'de_gsk_cma', 'base_runtime_sec', 210, ...
            'base_max_fes', 3000000, 'local_algorithm', 'lshade_cma', ...
            'local_radius', 0.004, 'local_population_num', 126, ...
            'pattern_sigma', 0.0008, 'pattern_block_rate', 0.035, ...
            'pattern_samples', 128);
        profile.algorithm = 'operator_pool_lshade_restart';
        profile.combination = sprintf('Adaptive operator-pool metaheuristic\nDE/GSK/RIME/elite covariance competition\nL-SHADE-CMA local restart\nStochastic block pattern refinement');
        profile.combination_number = 5;
    case 23
        opts = struct('population_num', 180, 'max_fes', 1000000, 'p_rate', 0.10, ...
            'base_algorithm', 'lshade', 'base_fraction', 0.80, ...
            'local_sigma', 0.002, 'restart_sigma', 0.001, 'restart_limit', 3);
        profile.algorithm = 'lshade_cmaes_refine';
        profile.combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nLocal CMA-ES rank-based covariance refinement');
        profile.combination_number = 3;
    case 25
        opts = struct('population_num', 240, 'max_fes', 1000000, 'p_rate', 0.10);
        profile.algorithm = 'lshade';
        profile.combination = sprintf('Differential Evolution (DE)\nLarge-population L-SHADE success-history adaptation');
        profile.combination_number = 1;
    case 28
        opts = struct('population_num', 180, 'max_fes', 1000000, 'p_rate', 0.10);
        profile.algorithm = 'lshade_cma';
        profile.combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nElite covariance evolutionary sampling');
        profile.combination_number = 5;
    case {29, 30}
        opts = struct('population_num', 180, 'max_fes', 1000000, 'p_rate', 0.10);
        profile.algorithm = 'lshade_cma';
        profile.combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nElite covariance evolutionary sampling');
        profile.combination_number = 5;
    otherwise
        opts = struct('population_num', 180, 'max_fes', 1000000, 'p_rate', 0.10);
        if any(func_num == [3 9])
            opts.include_center = true;
        end
        profile.algorithm = 'lshade';
        profile.combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation');
        profile.combination_number = 1;
end
profile.options = opts;
end

function profile = agent2_profile(func_num)
if func_num == 20
    opts = struct('population_num', 300, 'max_fes', 3500000, 'p_rate', 0.08, ...
        'scout_a', 'operator_pool', 'scout_b', 'operator_pool_gsk', ...
        'scout_a_fraction', 0.30, 'scout_b_fraction', 0.24, ...
        'scout_b_pop_rate', 0.90, 'refiner', 'lshade_cma', ...
        'local_population_num', 138, 'refine_radius', 0.004, ...
        'cross_radius', 0.0035, 'cma_rate', 0.16, ...
        'elite_rate', 0.22, 'cma_interval', 12);
    profile.algorithm = 'dual_basin_crossover_refine';
    profile.combination = sprintf('DE/GSK/RIME/CMA operator-pool scout\nGSK/RIME-biased operator-pool scout\nElite coordinate crossover between both basins\nL-SHADE-CMA exploitation');
    profile.combination_number = 5;
elseif any(func_num == [1 3 4 6 9 11 12 13 14 15 18 19 21 30])
    opts = struct('population_num', 130, 'max_iter', 1400);
    profile.algorithm = 'gsk';
    profile.combination = sprintf('Gaining-Sharing Knowledge (GSK)\nJunior-senior knowledge sharing');
    profile.combination_number = 1;
elseif any(func_num == [20 22])
    opts = struct('population_num', 90, 'max_iter', 800, 'num_starts', 2, 'p_rate', 0.18);
    profile.algorithm = 'ms_sade';
    profile.combination = sprintf('Self-Adaptive Differential Evolution (SaDE)\nMulti-start DE/current-to-pbest with archive');
    profile.combination_number = 4;
else
    opts = struct('population_num', 90, 'max_iter', 750);
    profile.algorithm = 'depso';
    profile.combination = sprintf('Differential Evolution (DE)\nParticle Swarm Optimization (PSO)');
    profile.combination_number = 3;
end
profile.options = opts;
end

function out = merge_options(defaults, overrides)
out = defaults;
if ~isstruct(overrides)
    return;
end
names = fieldnames(overrides);
for i = 1:numel(names)
    if ~isempty(overrides.(names{i}))
        out.(names{i}) = overrides.(names{i});
    end
end
end
