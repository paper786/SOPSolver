function result = SOP_cec2017_100D_retained_algorithm(func_num, seed, overrides)
% Fixed retained configurations for the user-accepted CEC2017 100D cases.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(overrides)
    overrides = struct();
end

problem = SOP_cec_problem('CEC2017', func_num, 100);

switch double(func_num)
    case 5
        options = struct('population_num', 420, 'max_fes', 5000000, ...
            'p_rate', 0.10, 'max_runtime_sec', 295);
        options.scout_a = 'jso_rsp';
        options.scout_b = 'lshade_cma';
        options.scout_a_fraction = 0.34;
        options.scout_b_fraction = 0.20;
        options.refiner = 'lshade_cma_cmaes';
        options.local_population_num = max(76, round(0.48 * options.population_num));
        options.refine_radius = 0.005;
        options.cross_radius = 0.004;
        options.crossover_mode = 'bbo_tlbo_bridge';
        options.fusion_tail_fraction = 0.40;
        options.bbo_bridge_elite_count = max(24, round(0.24 * options.population_num));
        options.bbo_tlbo_block_min = 5;
        options.bbo_tlbo_block_max = 13;
        options.bbo_tlbo_block_passes = 2;
        options.bbo_immigration_base = 0.22;
        options.bbo_immigration_span = 0.56;
        options.bbo_mutation_base = 0.07;
        options.bbo_mutation_span = 0.17;
        options.bbo_tlbo_teacher_scale = 0.44;
        options.bbo_tlbo_learner_scale = 0.20;
        options.bbo_tlbo_mutation_scale = 0.18;
        options.tlbo_bridge_rate = 0.72;
        options.tlbo_learner_rate = 0.42;
        options.bbo_bridge_center_blend_rate = 0.34;
        options.bbo_bridge_center_blend = 0.22;
        options.bbo_bridge_best_pull_rate = 0.28;
        options.bbo_bridge_best_pull = 0.18;
        options.cma_rate = 0.16;
        options.elite_rate = 0.22;
        options.cma_interval = 12;
        options.cmaes_reserve_fes = 420000;
        options.local_sigma = 0.00035;
        options.restart_sigma = 0.00015;
        options.restart_limit = 2;
        options.cma_population_num = max(20, round(0.10 * options.population_num));
        options = merge_options(options, overrides);
        result = SOP_agent_dual_basin_crossover_refine(problem, seed, options);
        combination = sprintf(['jSO ranked-r1/RSP independent scout\n' ...
            'L-SHADE-CMA independent scout\n' ...
            'BBO migration and TLBO block-learning bridge\n' ...
            'L-SHADE-CMA exploitation with micro-CMA-ES refinement']);
        combination_number = 5;

    case 7
        options = struct('population_num', 260, 'max_fes', 2200000, ...
            'p_rate', 0.10, 'max_runtime_sec', 295);
        options.explorer = 'mvo';
        options.refiner = 'lshade_cma';
        options.explore_fraction = 0.22;
        options = merge_options(options, overrides);
        result = SOP_agent_explore_exploit_hybrid(problem, seed, options);
        combination = sprintf(['Multi-Verse Optimizer (MVO)\n' ...
            'L-SHADE-CMA differential and covariance refinement']);
        combination_number = 3;

    case 8
        options = struct('population_num', 420, 'max_fes', 5000000, ...
            'p_rate', 0.08, 'max_runtime_sec', 295);
        options.base_algorithm = 'lshade_jso';
        options.include_center = true;
        options.ranked_r1 = true;
        options.rank_pressure = 1.7;
        options.base_fraction = 0.70;
        options.mu_F_init = 0.34;
        options.mu_CR_init = 0.86;
        options.p_rate_start = 0.20;
        options.p_rate_end = 0.040;
        options.weight_start = 0.62;
        options.weight_end = 1.36;
        options.archive_factor_start = 1.8;
        options.archive_factor_end = 3.2;
        options.local_sigma = 0.00028;
        options.restart_sigma = 0.00012;
        options.restart_limit = 2;
        options.cma_population_num = max(24, round(0.10 * options.population_num));
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cmaes_refine(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'Center-seeded jSO ranked-r1/RSP search\n' ...
            'Micro-CMA-ES covariance refinement']);
        combination_number = 3;

    case 11
        options = struct('population_num', 1800, 'max_fes', 1000000, ...
            'p_rate', 0.11, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.archive_rate = 1.4;
        options.eigen_rate = 0.4;
        options.neighbor_rate = 0.5;
        options.learning_period = 20;
        options.initial_frequency = 0.5;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cnepsin(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE success-history adaptation and linear population reduction\n' ...
            'Ensemble sinusoidal scaling-factor schedule\n' ...
            'Euclidean-neighborhood covariance coordinate crossover']);
        combination_number = 4;

    case 13
        options = struct('population_num', 220, 'max_fes', 3000000, ...
            'p_rate', 0.08, 'max_runtime_sec', 295);
        options.base_algorithm = 'lshade_cma';
        options.base_fraction = 1 / 3;
        options.local_sigma = 0.00025;
        options.restart_sigma = 0.00012;
        options.restart_limit = 3;
        options.cma_population_num = max(28, round(0.18 * options.population_num));
        options.lshade_cma_after = true;
        options.relay_population_num = max(44, round(0.20 * options.population_num));
        options.relay_radius = 0.0018;
        options.relay_cauchy = true;
        options.relay_cma_rate = 0.14;
        options.relay_elite_rate = 0.20;
        options.relay_cma_interval = 10;
        options.eda_after = true;
        options.eda_reserve_runtime_sec = 24;
        options.eda_cov_scale = 0.026;
        options.eda_iso_scale = 0.00014;
        options.eda_reset_cov_scale = 0.008;
        options.eda_reset_iso_scale = 0.00006;
        options.eda_block_rate = 0.030;
        options.eda_batch = 128;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cmaes_refine(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE-CMA covariance-assisted search\n' ...
            'CMA-ES refinement and L-SHADE-CMA relay\n' ...
            'Elite EDA covariance sampling tail']);
        combination_number = 5;

    case 15
        options = struct('population_num', 1800, 'max_fes', 1000000, ...
            'p_rate', 0.11, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.archive_rate = 1.4;
        options.eigen_rate = 0.4;
        options.neighbor_rate = 0.5;
        options.learning_period = 20;
        options.initial_frequency = 0.5;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cnepsin_faithful(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE success-history adaptation and linear population reduction\n' ...
            'Ensemble sinusoidal scaling-factor schedule\n' ...
            'Deduplicated archive and conditioned neighborhood eigen crossover']);
        combination_number = 4;

    case 17
        options = struct('population_num', 1800, 'max_fes', 1000000, ...
            'p_rate', 0.11, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.archive_rate = 1.4;
        options.class_learning_rate = 0.8;
        options.sigma_init = 0.5;
        options.eigen_rate = 0.4;
        options.neighbor_rate = 0.5;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_spacma_faithful(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE success-history adaptation and linear population reduction\n' ...
            'Semi-adaptive SPACMA DE/CMA offspring allocation\n' ...
            'cnEpSin Euclidean-neighborhood eigen-coordinate crossover']);
        combination_number = 5;

    case 18
        options = struct('population_num', 240, 'max_fes', 4000000, ...
            'p_rate', 0.08, 'max_runtime_sec', 295);
        options.pre_runtime_sec = 210;
        options.pre_max_fes = 3000000;
        options.base_algorithm = 'lshade';
        options.local_algorithm = 'lshade';
        options.base_max_fes = 1000000;
        options.local_radius = 0.018;
        options.local_population_num = max(100, round(0.70 * options.population_num));
        options.use_base_elites = false;
        options.cc_group_size = 8;
        options.cc_batch = max(72, round(0.30 * options.population_num));
        options.cc_partitions = 4;
        options.cc_sigma = 0.0012;
        options.cc_reset_sigma = 0.0030;
        options.cc_stall_limit = 14;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cc_refine(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'Plain L-SHADE local restart\n' ...
            'Wide block cooperative DE/stochastic refinement']);
        combination_number = 5;

    case 19
        options = struct('population_num', 1800, 'max_fes', 1000000, ...
            'p_rate', 0.11, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.archive_rate = 1.4;
        options.eigen_rate = 0.4;
        options.neighbor_rate = 0.5;
        options.learning_period = 20;
        options.initial_frequency = 0.5;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cnepsin(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE success-history adaptation and linear population reduction\n' ...
            'Ensemble sinusoidal scaling-factor schedule\n' ...
            'Euclidean-neighborhood covariance coordinate crossover']);
        combination_number = 4;

    case 21
        options = struct('population_num', 320, 'max_fes', 3000000, ...
            'p_rate', 0.07, 'max_runtime_sec', 295);
        options.base_algorithm = 'lshade_jso';
        options.include_center = true;
        options.ranked_r1 = true;
        options.rank_pressure = 1.7;
        options.base_fraction = 0.86;
        options.mu_F_init = 0.34;
        options.mu_CR_init = 0.86;
        options.p_rate_start = 0.20;
        options.p_rate_end = 0.040;
        options.archive_factor_start = 1.8;
        options.archive_factor_end = 3.2;
        options.local_population_num = max(58, round(0.19 * options.population_num));
        options.local_radius = 0.0040;
        options.reset_radius = 0.0085;
        options.local_F = 0.54;
        options.local_CR = 0.42;
        options.block_rate = 0.05;
        options.block_min = 4;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_subspace_de_refine(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'Center-seeded jSO ranked-r1/RSP search\n' ...
            'Random-subspace DE refinement']);
        combination_number = 3;

    case 22
        options = struct('population_num', 224, 'max_fes', 1000000, ...
            'p_rate', 0.10, 'max_runtime_sec', 295);
        options = merge_options(options, overrides);
        result = SOP_agent_bipop_cmaes_selector(problem, seed, options);
        combination = sprintf(['BIPOP-style CMA-ES basin selection\n' ...
            'Small, medium, and large population/step-size regimes\n' ...
            'Whole-vector objective selection']);
        combination_number = 1;

    case 24
        options = struct('population_num', 300, 'max_fes', 4000000, ...
            'p_rate', 0.10, 'max_runtime_sec', 295);
        options.scout_a = 'jso_rsp';
        options.scout_b = 'lshade_cma';
        options.jso_rsp_include_center = true;
        options.scout_a_fraction = 0.34;
        options.scout_b_fraction = 0.20;
        options.refiner = 'lshade_cma_cmaes';
        options.local_population_num = max(76, round(0.48 * options.population_num));
        options.refine_radius = 0.005;
        options.cross_radius = 0.004;
        options.crossover_mode = 'bbo_tlbo_bridge';
        options.fusion_tail_fraction = 0.40;
        options.bbo_bridge_elite_count = max(24, round(0.24 * options.population_num));
        options.bbo_tlbo_block_min = 5;
        options.bbo_tlbo_block_max = 13;
        options.bbo_tlbo_block_passes = 2;
        options.bbo_immigration_base = 0.22;
        options.bbo_immigration_span = 0.56;
        options.bbo_mutation_base = 0.07;
        options.bbo_mutation_span = 0.17;
        options.bbo_tlbo_teacher_scale = 0.44;
        options.bbo_tlbo_learner_scale = 0.20;
        options.bbo_tlbo_mutation_scale = 0.18;
        options.tlbo_bridge_rate = 0.72;
        options.tlbo_learner_rate = 0.42;
        options.bbo_bridge_center_blend_rate = 0.34;
        options.bbo_bridge_center_blend = 0.22;
        options.bbo_bridge_best_pull_rate = 0.28;
        options.bbo_bridge_best_pull = 0.18;
        options.cma_rate = 0.16;
        options.elite_rate = 0.22;
        options.cma_interval = 12;
        options.cmaes_reserve_fes = 420000;
        options.local_sigma = 0.00035;
        options.restart_sigma = 0.00015;
        options.restart_limit = 2;
        options.cma_population_num = max(20, round(0.10 * options.population_num));
        options = merge_options(options, overrides);
        result = SOP_agent_dual_basin_crossover_refine(problem, seed, options);
        combination = sprintf(['Center-seeded jSO ranked-r1/RSP independent scout\n' ...
            'L-SHADE-CMA independent scout\n' ...
            'BBO migration and TLBO block-learning bridge\n' ...
            'L-SHADE-CMA exploitation with micro-CMA-ES refinement']);
        combination_number = 5;

    case 26
        options = struct('population_num', 1800, 'max_fes', 1000000, ...
            'p_rate', 0.11, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.archive_rate = 1.4;
        options.eigen_rate = 0.4;
        options.neighbor_rate = 0.5;
        options.learning_period = 20;
        options.initial_frequency = 0.5;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_cnepsin_faithful(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'L-SHADE success-history adaptation and linear population reduction\n' ...
            'Ensemble sinusoidal scaling-factor schedule\n' ...
            'Deduplicated archive and conditioned neighborhood eigen crossover']);
        combination_number = 4;

    case 27
        options = struct('population_num', 1151, 'max_fes', 2000000, ...
            'p_rate', 0.25, 'max_runtime_sec', 295);
        options.memory_size = 5;
        options.min_population_num = 4;
        options.mu_F_init = 0.30;
        options.mu_CR_init = 0.80;
        options.p_rate_start = 0.25;
        options.p_rate_end = 0.25;
        options.archive_factor_start = 1.0;
        options.archive_factor_end = 1.0;
        options.fixed_high_memory = true;
        options.fixed_high_memory_rate = 0.20;
        options.fixed_high_F = 0.90;
        options.fixed_high_CR = 0.90;
        options.official_F_cap = true;
        options.official_stage_weight = true;
        options = merge_options(options, overrides);
        result = SOP_agent_lshade_jso(problem, seed, options);
        combination = sprintf(['Differential Evolution (DE)\n' ...
            'Official jSO staged parameter adaptation\n' ...
            'Success-history archive and linear population reduction']);
        combination_number = 3;

    otherwise
        error('SOP_cec2017_100D_retained_algorithm:UnsupportedFunction', ...
            'Supported accepted functions are F5, F7, F8, F11, F13, F15, F17, F18, F19, F21, F22, F24, F26, and F27.');
end

result.population_num = options.population_num;
result.algorithm_combination = combination;
result.combination_number = combination_number;
result.agent_id = 'Agent1';
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
