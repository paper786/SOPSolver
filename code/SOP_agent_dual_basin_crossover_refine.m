function result = SOP_agent_dual_basin_crossover_refine(problem, seed, options)
% Dual-basin metaheuristic crossover followed by DE-family exploitation.
%
% Two distinct derivative-free metaheuristics first search independently.
% Their elite populations are then crossed coordinate-wise and used as the
% initial population for a final L-SHADE/L-SHADE-CMA/jSO exploitation stage.
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
scout_a_key = lower(string(get_option(options, 'scout_a', 'operator_pool')));
scout_b_key = lower(string(get_option(options, 'scout_b', 'jso_rsp')));
scout_c_key = lower(string(get_option(options, 'scout_c', '')));
scout_a_fraction = get_option(options, 'scout_a_fraction', 0.30);
scout_b_fraction = get_option(options, 'scout_b_fraction', 0.24);
scout_c_fraction = get_option(options, 'scout_c_fraction', 0.12);

scout_a_options = options;
scout_a_options.max_fes = max(1000, floor(scout_a_fraction * max_fes));
scout_a_options.max_runtime_sec = max(1, scout_a_fraction * max_runtime_sec);
scout_a_options.population_num = max(30, round(get_option(options, 'scout_a_pop_rate', 1.0) * NP));
scout_a_options.verbose = false;
scout_a = run_scout(problem, double(seed), scout_a_key, scout_a_options);

scout_b_options = options;
scout_b_options.max_fes = max(1000, floor(scout_b_fraction * max_fes));
scout_b_options.max_runtime_sec = max(1, min(max_runtime_sec - toc(t_start), scout_b_fraction * max_runtime_sec));
scout_b_options.population_num = max(30, round(get_option(options, 'scout_b_pop_rate', 1.0) * NP));
if isfield(options, 'scout_b_cma_rate')
    scout_b_options.cma_rate = options.scout_b_cma_rate;
end
if isfield(options, 'scout_b_elite_rate')
    scout_b_options.elite_rate = options.scout_b_elite_rate;
end
if isfield(options, 'scout_b_cma_interval')
    scout_b_options.cma_interval = options.scout_b_cma_interval;
end
scout_b_options.verbose = false;
scout_b = run_scout(problem, double(seed) + 3571, scout_b_key, scout_b_options);

scouts = {scout_a, scout_b};
scout_labels = {label_for(scout_a_key), label_for(scout_b_key)};
if strlength(scout_c_key) > 0
    scout_c_options = options;
    scout_c_options.max_fes = max(1000, floor(scout_c_fraction * max_fes));
    scout_c_options.max_runtime_sec = max(1, min(max_runtime_sec - toc(t_start), scout_c_fraction * max_runtime_sec));
    scout_c_options.population_num = max(30, round(get_option(options, 'scout_c_pop_rate', 1.0) * NP));
    scout_c_options.verbose = false;
    scout_c = run_scout(problem, double(seed) + 6151, scout_c_key, scout_c_options);
    scouts{end + 1} = scout_c;
    scout_labels{end + 1} = label_for(scout_c_key);
end
[~, scout_order] = sort(cellfun(@(item) item.record_value, scouts));
selected = scouts{scout_order(1)};
secondary = scouts{scout_order(2)};
selected_label = scout_labels{scout_order(1)};
secondary_label = scout_labels{scout_order(2)};
ordered_scouts = scouts(scout_order);
ordered_labels = scout_labels(scout_order);
fusion_scout_count = min(numel(ordered_scouts), max(2, round(get_option(options, 'fusion_scout_count', numel(ordered_scouts)))));
fusion_scouts = ordered_scouts(1:fusion_scout_count);

remaining_fes = max(1000, max_fes - sum_cell_eval(scouts));
remaining_time = max(1, max_runtime_sec - toc(t_start));
refine_options = options;
refine_options.max_fes = remaining_fes;
refine_options.max_runtime_sec = remaining_time;
refine_options.population_num = get_option(options, 'local_population_num', max(60, round(0.55 * NP)));
if isfield(options, 'refiner_p_rate')
    refine_options.p_rate = options.refiner_p_rate;
end
refine_options.initial_point = selected.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.004);
refine_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
if get_option(options, 'anchor_archive_refiner', false)
    refine_options.external_anchor_archive = build_external_anchor_archive(ordered_scouts, ...
        get_option(options, 'anchor_archive_elite_rate', 0.36), problem.lb, problem.ub);
    refine_options.anchor_archive_rate = get_option(options, 'anchor_archive_rate', 0.18);
    refine_options.anchor_archive_start_progress = get_option(options, 'anchor_archive_start_progress', 0.08);
    refine_options.anchor_archive_end_progress = get_option(options, 'anchor_archive_end_progress', 0.84);
end
fusion_eval_count = 0;
fusion_iter_count = 0;
fusion_curve = [];
cross_radius = get_option(options, 'cross_radius', refine_options.initial_radius);
if get_option(options, 'success_filtered_fusion', false)
    [refine_options.initial_population, fusion_curve, fusion_eval_count, fusion_iter_count] = ...
        success_filtered_fusion_population(problem, ordered_scouts, refine_options.population_num, ...
        cross_radius, options);
    refine_options.max_fes = max(1000, refine_options.max_fes - fusion_eval_count);
elseif numel(fusion_scouts) > 2
    refine_options.initial_population = crossover_population_multi(fusion_scouts, refine_options.population_num, ...
        cross_radius, problem.lb, problem.ub, options);
else
    refine_options.initial_population = crossover_population(fusion_scouts{1}, fusion_scouts{2}, refine_options.population_num, ...
        cross_radius, problem.lb, problem.ub, options);
end
if get_option(options, 'protected_clearing_probe', false)
    [refine_options.initial_population, probe_curve, probe_eval_count, probe_iter_count] = ...
        protected_clearing_probe_population(problem, ordered_scouts, refine_options.initial_population, ...
        cross_radius, options);
    fusion_curve = [fusion_curve(:); probe_curve(:)];
    fusion_eval_count = fusion_eval_count + probe_eval_count;
    fusion_iter_count = fusion_iter_count + probe_iter_count;
    refine_options.max_fes = max(1000, refine_options.max_fes - probe_eval_count);
end
refine_options.preserve_initial_population_after_radius = true;
refine_options.verbose = false;

if get_option(options, 'split_island_refine', false)
    [refine, refiner_label] = split_island_refine(problem, double(seed), fusion_scouts, refine_options, ...
        cross_radius, options);
else
refiner_key = lower(string(get_option(options, 'refiner', 'lshade_cma')));
switch refiner_key
    case "jso"
        refine_options = apply_jso_rsp(refine_options);
        refine = SOP_agent_lshade_jso(problem, double(seed) + 7919, refine_options);
        refiner_label = 'jSO/L-SHADE ranked-r1 exploitation';
    case "lshade"
        refine = SOP_agent_lshade(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE exploitation';
    case "lshade_cma_hses"
        hses_reserve_fes = get_option(options, 'hses_reserve_fes', max(1000, floor(0.10 * remaining_fes)));
        hses_reserve_fes = max(1000, min(hses_reserve_fes, max(1000, remaining_fes - 1000)));
        lshade_cma_options = refine_options;
        lshade_cma_options.max_fes = remaining_fes - hses_reserve_fes;
        lshade_cma_options.max_runtime_sec = max(1, get_option(options, 'lshade_cma_hses_fraction', 0.86) * remaining_time);
        lshade_cma_options.cma_rate = get_option(options, 'cma_rate', 0.16);
        lshade_cma_options.elite_rate = get_option(options, 'elite_rate', 0.22);
        lshade_cma_options.cma_interval = get_option(options, 'cma_interval', 12);
        lshade_cma = SOP_agent_lshade_cma(problem, double(seed) + 7919, lshade_cma_options);

        hses_options = struct();
        hses_options.population_num = get_option(options, 'hses_population_num', max(24, round(0.12 * NP)));
        hses_options.max_runtime_sec = max(1, remaining_time - lshade_cma.runtime);
        hses_options.max_fes = hses_reserve_fes;
        hses_options.initial_point = lshade_cma.best_position;
        hses_options.initial_population = lshade_cma.final_population;
        hses_options.initial_radius = get_option(options, 'hses_initial_radius', 0.0016);
        hses_options.sigma0 = get_option(options, 'hses_sigma0', 0.00075);
        hses_options.reset_sigma = get_option(options, 'hses_reset_sigma', 0.0025);
        hses_options.covariance_sample_rate = get_option(options, 'hses_covariance_sample_rate', 0.46);
        hses_options.univariate_sample_rate = get_option(options, 'hses_univariate_sample_rate', 0.40);
        hses_options.hybrid_mask_rate = get_option(options, 'hses_hybrid_mask_rate', 0.36);
        hses_options.cov_scale = get_option(options, 'hses_cov_scale', 0.72);
        hses_options.uni_scale = get_option(options, 'hses_uni_scale', 0.92);
        hses_options.best_blend = get_option(options, 'hses_best_blend', 0.26);
        hses_options.elite_recomb_rate = get_option(options, 'hses_elite_recomb_rate', 0.18);
        hses_options.best_pull_rate = get_option(options, 'hses_best_pull_rate', 0.22);
        hses_options.verbose = false;
        hses = SOP_agent_hses_sampling(problem, double(seed) + 104729, hses_options);

        if hses.record_value < lshade_cma.record_value
            refine = hses;
        else
            refine = lshade_cma;
        end
        refine.runtime = lshade_cma.runtime + hses.runtime;
        refine.evaluation_count = lshade_cma.evaluation_count + hses.evaluation_count;
        refine.iteration = lshade_cma.iteration + hses.iteration;
        refine.convergence_curve = [lshade_cma.convergence_curve(:); hses.convergence_curve(:)];
        refine.raw_convergence_curve = [lshade_cma.raw_convergence_curve(:); hses.raw_convergence_curve(:)];
        refiner_label = 'L-SHADE-CMA exploitation plus HSES hybrid covariance/univariate sampling refinement';
    case "lshade_cma_jso_cmaes"
        cmaes_reserve_fes = get_option(options, 'cmaes_reserve_fes', max(1000, floor(0.16 * remaining_fes)));
        jso_reserve_fes = get_option(options, 'jso_reserve_fes', max(1000, floor(0.18 * remaining_fes)));
        cmaes_reserve_fes = max(1000, min(cmaes_reserve_fes, max(1000, remaining_fes - 2000)));
        jso_reserve_fes = max(1000, min(jso_reserve_fes, max(1000, remaining_fes - cmaes_reserve_fes - 1000)));
        lshade_cma_fes = max(1000, remaining_fes - cmaes_reserve_fes - jso_reserve_fes);
        lshade_cma_options = refine_options;
        lshade_cma_options.max_fes = lshade_cma_fes;
        lshade_cma_options.max_runtime_sec = max(1, 0.56 * remaining_time);
        lshade_cma_options.cma_rate = get_option(options, 'cma_rate', 0.16);
        lshade_cma_options.elite_rate = get_option(options, 'elite_rate', 0.22);
        lshade_cma_options.cma_interval = get_option(options, 'cma_interval', 12);
        lshade_cma = SOP_agent_lshade_cma(problem, double(seed) + 7919, lshade_cma_options);

        jso_options = options;
        jso_options.max_fes = jso_reserve_fes;
        jso_options.max_runtime_sec = max(1, 0.26 * remaining_time);
        jso_options.population_num = get_option(options, 'jso_population_num', max(42, round(0.24 * NP)));
        jso_options.initial_point = lshade_cma.best_position;
        jso_options.initial_radius = get_option(options, 'jso_radius', 0.0024);
        jso_options.initial_cauchy = true;
        jso_options.verbose = false;
        jso_options = apply_jso_rsp(jso_options);
        jso_refine = SOP_agent_lshade_jso(problem, double(seed) + 65537, jso_options);

        if jso_refine.record_value < lshade_cma.record_value
            cma_start = jso_refine.best_position;
        else
            cma_start = lshade_cma.best_position;
        end
        cma_options = struct();
        cma_options.population_num = get_option(options, 'cma_population_num', max(18, round(0.08 * NP)));
        cma_options.max_runtime_sec = max(1, remaining_time - lshade_cma.runtime - jso_refine.runtime);
        cma_options.max_fes = cmaes_reserve_fes;
        cma_options.initial_point = cma_start;
        cma_options.sigma0 = get_option(options, 'local_sigma', 0.00028);
        cma_options.restart_sigma = get_option(options, 'restart_sigma', 0.00012);
        cma_options.restart_limit = get_option(options, 'restart_limit', 2);
        cma_options.eig_interval = get_option(options, 'eig_interval', 8);
        cma_options.verbose = false;
        cmaes = SOP_agent_cma_es(problem, double(seed) + 104729, cma_options);

        refine = lshade_cma;
        if jso_refine.record_value < refine.record_value
            refine = jso_refine;
        end
        if cmaes.record_value < refine.record_value
            refine = cmaes;
        end
        refine.runtime = lshade_cma.runtime + jso_refine.runtime + cmaes.runtime;
        refine.evaluation_count = lshade_cma.evaluation_count + jso_refine.evaluation_count + cmaes.evaluation_count;
        refine.iteration = lshade_cma.iteration + jso_refine.iteration + cmaes.iteration;
        refine.convergence_curve = [lshade_cma.convergence_curve(:); jso_refine.convergence_curve(:); cmaes.convergence_curve(:)];
        refine.raw_convergence_curve = [lshade_cma.raw_convergence_curve(:); raw_curve_for(jso_refine); raw_curve_for(cmaes)];
        refiner_label = 'L-SHADE-CMA exploitation plus jSO relay and micro-CMA-ES refinement';
    case "lshade_cma_cmaes"
        cmaes_reserve_fes = get_option(options, 'cmaes_reserve_fes', max(1000, floor(0.22 * remaining_fes)));
        cmaes_reserve_fes = max(1000, min(cmaes_reserve_fes, max(1000, remaining_fes - 1000)));
        lshade_cma_options = refine_options;
        lshade_cma_options.max_fes = remaining_fes - cmaes_reserve_fes;
        lshade_cma_options.max_runtime_sec = max(1, 0.72 * remaining_time);
        lshade_cma_options.cma_rate = get_option(options, 'cma_rate', 0.16);
        lshade_cma_options.elite_rate = get_option(options, 'elite_rate', 0.22);
        lshade_cma_options.cma_interval = get_option(options, 'cma_interval', 12);
        lshade_cma = SOP_agent_lshade_cma(problem, double(seed) + 7919, lshade_cma_options);
        cma_options = struct();
        cma_options.population_num = get_option(options, 'cma_population_num', max(20, round(0.10 * NP)));
        cma_options.max_runtime_sec = max(1, remaining_time - lshade_cma.runtime);
        cma_options.max_fes = cmaes_reserve_fes;
        cma_options.initial_point = lshade_cma.best_position;
        cma_options.sigma0 = get_option(options, 'local_sigma', 0.00035);
        cma_options.restart_sigma = get_option(options, 'restart_sigma', 0.00015);
        cma_options.restart_limit = get_option(options, 'restart_limit', 2);
        cma_options.eig_interval = get_option(options, 'eig_interval', 8);
        if get_option(options, 'seed_covariance', false) && isfield(lshade_cma, 'final_population')
            cma_options.initial_population = lshade_cma.final_population;
            cma_options.seed_covariance = true;
            cma_options.covariance_seed_count = get_option(options, 'covariance_seed_count', min(size(lshade_cma.final_population, 1), 2 * problem.dimension));
            cma_options.seed_covariance_blend = get_option(options, 'seed_covariance_blend', 0.55);
            cma_options.seed_covariance_ridge = get_option(options, 'seed_covariance_ridge', 0.08);
            cma_options.seed_covariance_min_eig = get_option(options, 'seed_covariance_min_eig', 0.05);
            cma_options.seed_covariance_max_eig = get_option(options, 'seed_covariance_max_eig', 12.0);
        end
        cma_options.verbose = false;
        cmaes = SOP_agent_cma_es(problem, double(seed) + 104729, cma_options);
        if cmaes.record_value < lshade_cma.record_value
            refine = cmaes;
        else
            refine = lshade_cma;
        end
        refine.runtime = lshade_cma.runtime + cmaes.runtime;
        refine.evaluation_count = lshade_cma.evaluation_count + cmaes.evaluation_count;
        refine.iteration = lshade_cma.iteration + cmaes.iteration;
        refine.convergence_curve = [lshade_cma.convergence_curve(:); cmaes.convergence_curve(:)];
        refine.raw_convergence_curve = [lshade_cma.raw_convergence_curve(:); cmaes.raw_convergence_curve(:)];
        refiner_label = 'L-SHADE-CMA exploitation plus micro-CMA-ES refinement';
    otherwise
        refine_options.cma_rate = get_option(options, 'cma_rate', 0.16);
        refine_options.elite_rate = get_option(options, 'elite_rate', 0.22);
        refine_options.cma_interval = get_option(options, 'cma_interval', 12);
        refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
        refiner_label = 'L-SHADE-CMA exploitation';
end
end

result = selected;
for k = 2:numel(ordered_scouts)
    if ordered_scouts{k}.record_value < result.record_value
        result = ordered_scouts{k};
    end
end
if refine.record_value < result.record_value
    result = refine;
end
result.runtime = toc(t_start);
result.evaluation_count = sum_cell_eval(scouts) + fusion_eval_count + refine.evaluation_count;
result.iteration = sum_cell_iter(scouts) + fusion_iter_count + refine.iteration;
result.convergence_curve = [cell_curve_concat(scouts, false); SOP_cec_record_value(fusion_curve(:), problem); refine.convergence_curve(:)];
result.raw_convergence_curve = [cell_curve_concat(scouts, true); fusion_curve(:); raw_curve_for(refine)];
scout_text = sprintf('%s independent scout\n', ordered_labels{:});
if get_option(options, 'success_filtered_fusion', false)
    fusion_label = 'Success-filtered elite-DE/BLX, soft opposition/snap, and coordinate crossover between retained basins';
elseif lower(string(get_option(options, 'crossover_mode', 'coordinate'))) == "hses_bridge"
    fusion_label = 'HSES covariance and univariate sampling bridge between retained basins';
elseif lower(string(get_option(options, 'crossover_mode', 'coordinate'))) == "bbo_tlbo_bridge"
    fusion_label = 'BBO migration and TLBO block-learning bridge between retained basins';
else
    fusion_label = 'Elite coordinate crossover between retained basins';
end
if get_option(options, 'anchor_archive_refiner', false)
    refiner_label = sprintf('%s with scout-anchor archive donors', refiner_label);
end
result.algorithm_combination = sprintf('%s%s\n%s', scout_text, fusion_label, refiner_label);
result.combination_number = 5;
result.agent_id = 'Agent2';
end

function anchors = build_external_anchor_archive(scouts, elite_rate, lb, ub)
anchors = [];
for s = 1:numel(scouts)
    scout = scouts{s};
    if isfield(scout, 'final_population') && ~isempty(scout.final_population)
        take = max(2, round(elite_rate * size(scout.final_population, 1)));
        take = min(size(scout.final_population, 1), take);
        anchors = [anchors; scout.final_population(1:take, :)]; %#ok<AGROW>
    elseif isfield(scout, 'best_position') && ~isempty(scout.best_position)
        anchors = [anchors; scout.best_position(:)']; %#ok<AGROW>
    end
end
if isempty(anchors)
    return;
end
anchors = min(max(anchors, lb), ub);
end

function [population, curve, eval_count, iter_count] = protected_clearing_probe_population(problem, scouts, population, radius_scale, options)
eval_count = 0;
iter_count = 0;
curve = [];
[NP, D] = size(population);
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
elite_sets = cellfun(@get_elites, scouts, 'UniformOutput', false);
all_elites = vertcat(elite_sets{:});
if size(all_elites, 1) < 4 || NP < 8
    return;
end
elite_count = min(size(all_elites, 1), get_option(options, 'probe_elite_count', max(24, round(0.38 * NP))));
pool = all_elites(1:elite_count, :);
niche_count = min(elite_count, get_option(options, 'probe_niche_count', max(6, round(0.08 * NP))));
representatives = select_clearing_representatives(pool, (1:elite_count)', niche_count, ...
    get_option(options, 'probe_clearing_radius', 0.010), span);
rep_count = size(representatives, 1);
if rep_count < 2
    return;
end
probe_count = min(max(4, round(get_option(options, 'probe_pool_rate', 0.46) * NP)), ...
    max(4, problem.dimension * get_option(options, 'probe_dim_eval_rate', 1.2)));
replace_count = min(NP - 3, max(2, round(get_option(options, 'probe_replace_rate', 0.07) * NP)));
probe_count = min(probe_count, max(0, get_option(options, 'probe_max_eval', probe_count)));
if probe_count <= 0 || replace_count <= 0
    return;
end
radius = radius_scale .* span;
best = scouts{1}.best_position;
candidates = zeros(probe_count, D);
block_min = max(2, round(get_option(options, 'probe_block_min_rate', 0.035) * D));
block_max = max(block_min, round(get_option(options, 'probe_block_max_rate', 0.11) * D));
for k = 1:probe_count
    rep = representatives(randi(rep_count), :);
    donor_a = representatives(randi(rep_count), :);
    donor_b = sample_elite(elite_sets);
    donor_c = sample_elite(elite_sets);
    child = rep;
    block_len = min(D, randi([block_min, block_max]));
    dims = randperm(D, block_len);
    if rand() < get_option(options, 'probe_blx_rate', 0.48)
        lo = min(rep(dims), donor_a(dims));
        hi = max(rep(dims), donor_a(dims));
        width = max(hi - lo, 1e-12 .* span(dims));
        alpha = get_option(options, 'probe_blx_alpha', 0.16);
        child(dims) = lo - alpha .* width + rand(1, block_len) .* ((1 + 2 * alpha) .* width);
    else
        F = get_option(options, 'probe_de_weight', 0.30) + 0.20 * rand();
        child(dims) = child(dims) + F .* (donor_b(dims) - donor_c(dims));
    end
    if rand() < get_option(options, 'probe_best_pull_rate', 0.34)
        pull = get_option(options, 'probe_best_pull', 0.14) * rand();
        child = child + pull .* (best - child);
    end
    if rand() < get_option(options, 'probe_noise_rate', 0.22)
        child(dims) = child(dims) + randn(1, block_len) .* ...
            (get_option(options, 'probe_noise_scale', 0.18) .* radius(dims));
    end
    candidates(k, :) = min(max(child, lb), ub);
end
values = SOP_cec_evaluate(candidates, problem);
eval_count = numel(values);
iter_count = 1;
[best_values, order] = sort(values);
take = min(replace_count, numel(order));
rows = NP - take + 1:NP;
population(rows, :) = candidates(order(1:take), :);
curve = cummin(best_values(:));
end

function [refine, label] = split_island_refine(problem, seed, scouts, refine_options, cross_radius, options)
remaining_fes = refine_options.max_fes;
remaining_time = refine_options.max_runtime_sec;
NP = refine_options.population_num;
rate_a = get_option(options, 'split_island_a_fes_rate', 0.52);
rate_b = get_option(options, 'split_island_b_fes_rate', 0.34);
fes_a = max(1000, floor(rate_a * remaining_fes));
fes_b = max(1000, floor(rate_b * remaining_fes));
fes_m = max(0, remaining_fes - fes_a - fes_b);
time_a = max(1, get_option(options, 'split_island_a_time_rate', 0.50) * remaining_time);
time_b = max(1, get_option(options, 'split_island_b_time_rate', 0.32) * remaining_time);
time_m = max(1, remaining_time - time_a - time_b);

mode_a_options = options;
mode_a_options.crossover_mode = 'elite_de_blx';
mode_b_options = options;
mode_b_options.crossover_mode = 'soft_obl_snap';
if numel(scouts) > 2
    population_a = crossover_population_multi(scouts, NP, cross_radius, problem.lb, problem.ub, mode_a_options);
    population_b = crossover_population_multi(scouts, NP, cross_radius, problem.lb, problem.ub, mode_b_options);
else
    population_a = crossover_population(scouts{1}, scouts{2}, NP, cross_radius, problem.lb, problem.ub, mode_a_options);
    population_b = crossover_population(scouts{1}, scouts{2}, NP, cross_radius, problem.lb, problem.ub, mode_b_options);
end

island_a_options = refine_options;
island_a_options.max_fes = fes_a;
island_a_options.max_runtime_sec = time_a;
island_a_options.initial_population = population_a;
island_a_options.initial_point = population_a(1, :);
island_a_options.preserve_initial_population_after_radius = true;
island_a_options.verbose = false;
island_a = SOP_agent_lshade_cma(problem, seed + 7919, island_a_options);

island_b_options = refine_options;
island_b_options.max_fes = fes_b;
island_b_options.max_runtime_sec = time_b;
island_b_options.initial_population = population_b;
island_b_options.initial_point = population_b(1, :);
island_b_options.preserve_initial_population_after_radius = true;
island_b_options.verbose = false;
island_b = SOP_agent_lshade_cma(problem, seed + 65537, island_b_options);

best = island_a;
if island_b.record_value < best.record_value
    best = island_b;
end
migration = [];
if fes_m >= 1000 && time_m > 1
    migration_options = refine_options;
    migration_options.max_fes = fes_m;
    migration_options.max_runtime_sec = time_m;
    migration_options.population_num = max(42, round(get_option(options, 'split_migration_pop_rate', 0.34) * NP));
    migration_options.initial_population = migration_population(island_a, island_b, migration_options.population_num, problem.lb, problem.ub, options);
    migration_options.initial_point = best.best_position;
    migration_options.initial_radius = get_option(options, 'split_migration_radius', 0.0022);
    migration_options.preserve_initial_population_after_radius = true;
    migration_options.verbose = false;
    migration = SOP_agent_lshade_cma(problem, seed + 104729, migration_options);
    if migration.record_value < best.record_value
        best = migration;
    end
end

refine = best;
refine.runtime = island_a.runtime + island_b.runtime;
refine.evaluation_count = island_a.evaluation_count + island_b.evaluation_count;
refine.iteration = island_a.iteration + island_b.iteration;
raw_curve = [raw_curve_for(island_a); raw_curve_for(island_b)];
curve = [island_a.convergence_curve(:); island_b.convergence_curve(:)];
if ~isempty(migration)
    refine.runtime = refine.runtime + migration.runtime;
    refine.evaluation_count = refine.evaluation_count + migration.evaluation_count;
    refine.iteration = refine.iteration + migration.iteration;
    raw_curve = [raw_curve; raw_curve_for(migration)];
    curve = [curve; migration.convergence_curve(:)];
end
refine.raw_convergence_curve = raw_curve;
refine.convergence_curve = curve;
label = 'Split-island L-SHADE-CMA refinement with elite-DE/BLX island, soft OBL/snap island, and elite migration';
end

function population = migration_population(island_a, island_b, NP, lb, ub, options)
D = numel(island_a.best_position);
span = ub - lb;
population = repmat(island_a.best_position, NP, 1);
if island_b.record_value < island_a.record_value
    population = repmat(island_b.best_position, NP, 1);
end
elites_a = final_elites(island_a, max(8, round(0.42 * NP)));
elites_b = final_elites(island_b, max(8, round(0.42 * NP)));
rows_a = 1:min(size(elites_a, 1), NP);
population(rows_a, :) = elites_a(rows_a, :);
start_b = numel(rows_a) + 1;
rows_b = start_b:min(NP, start_b + size(elites_b, 1) - 1);
if ~isempty(rows_b)
    population(rows_b, :) = elites_b(1:numel(rows_b), :);
end
radius = get_option(options, 'split_migration_radius', 0.0022) .* span;
for i = max(2, numel(rows_a) + numel(rows_b) + 1):NP
    donor_a = elites_a(randi(size(elites_a, 1)), :);
    donor_b = elites_b(randi(size(elites_b, 1)), :);
    alpha = -0.08 + 1.16 * rand(1, D);
    child = donor_a + alpha .* (donor_b - donor_a);
    if rand() < 0.35
        mask = rand(1, D) < get_option(options, 'split_migration_block_rate', 0.06);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        center = 0.5 * (donor_a + donor_b);
        child(mask) = center(mask) + randn(1, nnz(mask)) .* radius(mask);
    end
    population(i, :) = child;
end
population = min(max(population, lb), ub);
end

function elites = final_elites(result, count)
if isfield(result, 'final_population') && ~isempty(result.final_population)
    elites = result.final_population(1:min(count, size(result.final_population, 1)), :);
else
    elites = repmat(result.best_position, max(1, count), 1);
end
if size(elites, 1) < 2
    elites = [elites; result.best_position];
end
end

function result = run_scout(problem, seed, key, options)
switch key
    case "operator_pool"
        options.profile = get_option(options, 'operator_profile', 'de_gsk_cma');
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        result = SOP_agent_operator_pool(problem, seed, options);
    case "operator_pool_gsk"
        options.profile = 'gsk_rime_de';
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        result = SOP_agent_operator_pool(problem, seed, options);
    case "operator_pool_cma"
        options.profile = 'de_cma_exploit';
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        result = SOP_agent_operator_pool(problem, seed, options);
    case "jso_rsp"
        options = apply_jso_rsp(options);
        result = SOP_agent_lshade_jso(problem, seed, options);
    case "code_epsde"
        options.profile = get_option(options, 'code_profile', 'exploit');
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        result = SOP_agent_code_epsde_pool(problem, seed, options);
    case "quantum_snap"
        options.profile = get_option(options, 'quantum_profile', 'snap_focus');
        options.include_center = get_option(options, 'include_center', true);
        result = SOP_agent_quantum_snap_de(problem, seed, options);
    case "lshade_cma"
        result = SOP_agent_lshade_cma(problem, seed, options);
    case "lshade"
        result = SOP_agent_lshade(problem, seed, options);
    case "spacma"
        options.explore_fraction = get_option(options, 'spacma_explore_fraction', 0.60);
        options.cma_fraction = get_option(options, 'spacma_cma_fraction', 0.16);
        options.local_radius = get_option(options, 'spacma_local_radius', 0.0030);
        options.cma_sigma = get_option(options, 'spacma_cma_sigma', 0.0020);
        options.tail_cma_rate = get_option(options, 'spacma_tail_cma_rate', 0.14);
        options.tail_elite_rate = get_option(options, 'spacma_tail_elite_rate', 0.22);
        options.tail_cma_interval = get_option(options, 'spacma_tail_cma_interval', 10);
        result = SOP_agent_lshade_spacma_archive(problem, seed, options);
    case "cma_es"
        options.sigma0 = get_option(options, 'cma_scout_sigma0', 0.24);
        options.restart_sigma = get_option(options, 'cma_scout_restart_sigma', 0.075);
        options.restart_limit = get_option(options, 'cma_scout_restart_limit', 1);
        options.eig_interval = get_option(options, 'cma_scout_eig_interval', max(6, floor(problem.dimension / 10)));
        result = SOP_agent_cma_es(problem, seed, options);
    case "gsk"
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        options.knowledge_rate = get_option(options, 'gsk_knowledge_rate', 0.72);
        options.knowledge_factor = get_option(options, 'gsk_knowledge_factor', 0.38);
        result = SOP_agent_gsk(problem, seed, options);
    otherwise
        options.method = char(key);
        options.max_iter = max(1, floor((options.max_fes - options.population_num) / options.population_num));
        result = SOP_agent_literature_swarm(problem, seed, options);
end
end

function population = crossover_population(primary, secondary, NP, radius_scale, lb, ub, options)
if nargin < 7 || isempty(options)
    options = struct();
end
D = numel(primary.best_position);
span = ub - lb;
radius = radius_scale .* span;
population = repmat(primary.best_position, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
population(1, :) = primary.best_position;
population(2, :) = secondary.best_position;
elite_a = get_elites(primary);
elite_b = get_elites(secondary);
elite_count = min(max(4, floor(0.30 * NP)), size(elite_a, 1));
population(3:2 + elite_count, :) = elite_a(1:elite_count, :);
offset = 2 + elite_count;
elite_b_count = min(max(4, floor(0.20 * NP)), min(size(elite_b, 1), NP - offset));
if elite_b_count > 0
    population(offset + (1:elite_b_count), :) = elite_b(1:elite_b_count, :);
    offset = offset + elite_b_count;
end
for i = offset + 1:NP
    donor_a = elite_a(randi(size(elite_a, 1)), :);
    donor_b = elite_b(randi(size(elite_b, 1)), :);
    mask = rand(1, D) < 0.50;
    child = donor_a;
    child(mask) = donor_b(mask);
    if rand() < 0.50
        diff = primary.best_position - secondary.best_position;
        child = child + (0.10 + 0.30 * rand()) .* randn(1, D) .* diff;
    else
        child = child + randn(1, D) .* radius;
    end
    population(i, :) = child;
end
population = min(max(population, lb), ub);
population = apply_crossover_mode(population, {primary, secondary}, radius_scale, lb, ub, options);
end

function population = crossover_population_multi(scouts, NP, radius_scale, lb, ub, options)
if nargin < 6 || isempty(options)
    options = struct();
end
primary = scouts{1};
secondary = scouts{2};
base_options = options;
base_options.crossover_mode = 'coordinate';
population = crossover_population(primary, secondary, NP, radius_scale, lb, ub, base_options);
D = numel(primary.best_position);
span = ub - lb;
radius = radius_scale .* span;
offset = min(NP, max(3, floor(0.55 * NP)));
elite_sets = cellfun(@get_elites, scouts, 'UniformOutput', false);
for i = offset + 1:NP
    child = primary.best_position;
    for s = 1:numel(elite_sets)
        donors = elite_sets{s};
        donor = donors(randi(size(donors, 1)), :);
        mask_rate = 0.25 + 0.20 * rand();
        mask = rand(1, D) < mask_rate;
        if s == 1 && ~any(mask)
            mask(randi(D)) = true;
        end
        child(mask) = donor(mask);
    end
    if rand() < 0.55
        a = elite_sets{randi(numel(elite_sets))};
        b = elite_sets{randi(numel(elite_sets))};
        donor_a = a(randi(size(a, 1)), :);
        donor_b = b(randi(size(b, 1)), :);
        child = child + (0.10 + 0.25 * rand()) .* (donor_a - donor_b);
    else
        child = child + randn(1, D) .* radius;
    end
    population(i, :) = child;
end
population = min(max(population, lb), ub);
population = apply_crossover_mode(population, scouts, radius_scale, lb, ub, options);
end

function population = apply_crossover_mode(population, scouts, radius_scale, lb, ub, options)
mode = lower(string(get_option(options, 'crossover_mode', 'coordinate')));
if mode == "coordinate"
    return;
end

[NP, D] = size(population);
span = ub - lb;
radius = radius_scale .* span;
elite_sets = cellfun(@get_elites, scouts, 'UniformOutput', false);
if mode == "bestpoint_elite_de"
    elite_sets{1} = scouts{1}.best_position;
elseif mode == "secondary_bestpoint_elite_de" && numel(elite_sets) > 1
    elite_sets{2} = scouts{2}.best_position;
end
all_elites = vertcat(elite_sets{:});
primary = scouts{1};
best = primary.best_position;
center = mean(all_elites(1:min(size(all_elites, 1), max(4, round(0.12 * NP))), :), 1);
tail_fraction = get_option(options, 'fusion_tail_fraction', 0.38);
start_row = min(NP, max(4, floor((1 - tail_fraction) * NP)));
domain_center = 0.5 .* (lb + ub);
grid_scale = get_option(options, 'snap_grid_scale', 0.003);

if mode == "path_relink"
    path_noise = get_option(options, 'path_noise_scale', 0.16);
    path_mask_rate = get_option(options, 'path_mask_rate', 0.62);
    for row = start_row:NP
        donor_a = sample_elite(elite_sets);
        donor_b = sample_elite(elite_sets);
        if mod(row - start_row, 5) == 0 && numel(scouts) > 1
            donor_a = scouts{1}.best_position;
            donor_b = scouts{2}.best_position;
        end
        t = ((row - start_row + 1) / max(1, NP - start_row + 1));
        t = min(1, max(0, 0.08 + 0.84 * t + 0.08 * randn()));
        mask = rand(1, D) < path_mask_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        child = donor_a;
        child(mask) = donor_a(mask) + t .* (donor_b(mask) - donor_a(mask));
        if rand() < 0.35
            child = child + (0.10 + 0.30 * rand()) .* (best - donor_a);
        end
        child = child + randn(1, D) .* (path_noise .* radius);
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

if mode == "clearing_elite_de_blx"
    elite_count = min(size(all_elites, 1), get_option(options, 'clearing_pool_count', max(24, round(0.46 * NP))));
    raw_pool = all_elites(1:elite_count, :);
    raw_values = (1:elite_count)';
    niche_count = min(elite_count, get_option(options, 'clearing_niche_count', max(8, round(0.16 * NP))));
    niche_radius = get_option(options, 'clearing_radius', 0.018);
    representatives = select_clearing_representatives(raw_pool, raw_values, niche_count, niche_radius, span);
    rep_count = size(representatives, 1);
    if rep_count < 2
        representatives = [best; center];
        rep_count = size(representatives, 1);
    end
    blx_alpha = get_option(options, 'clearing_blx_alpha', 0.30);
    de_weight = get_option(options, 'clearing_de_weight', 0.40);
    block_rate = get_option(options, 'clearing_block_rate', max(0.05, 6 / D));
    for row = start_row:NP
        ids = randperm(rep_count, min(rep_count, 4));
        while numel(ids) < 4
            ids(end + 1) = randi(rep_count); %#ok<AGROW>
        end
        a = representatives(ids(1), :);
        b = representatives(ids(2), :);
        c = representatives(ids(3), :);
        d = representatives(ids(4), :);
        lo = min(a, b);
        hi = max(a, b);
        width = max(hi - lo, 1e-12 .* span);
        child = lo - blx_alpha .* width + rand(1, D) .* ((1 + 2 * blx_alpha) .* width);
        if rand() < get_option(options, 'clearing_de_rate', 0.62)
            mask = rand(1, D) < block_rate;
            if ~any(mask)
                mask(randi(D)) = true;
            end
            child(mask) = child(mask) + de_weight .* (c(mask) - d(mask));
        end
        if rand() < get_option(options, 'clearing_best_pull_rate', 0.32)
            pull = get_option(options, 'clearing_best_pull', 0.16) * rand();
            child = child + pull .* (best - child);
        end
        if rand() < get_option(options, 'clearing_noise_rate', 0.18)
            mask = rand(1, D) < block_rate;
            noise = randn(1, D);
            child(mask) = child(mask) + get_option(options, 'clearing_noise_scale', 0.11) .* noise(mask) .* radius(mask);
        end
        population(row, :) = min(max(child, lb), ub);
    end
    keep = min(NP, rep_count);
    population(1:keep, :) = min(max(representatives(1:keep, :), lb), ub);
    return;
end

if mode == "elite_eda_bridge"
    elite_count = min(size(all_elites, 1), get_option(options, 'eda_bridge_elite_count', max(12, round(0.22 * NP))));
    bridge_center = weighted_elite_center(all_elites, elite_count);
    bridge_center = 0.68 .* best + 0.32 .* bridge_center;
    centered = all_elites(1:elite_count, :) - bridge_center;
    weights = log(elite_count + 0.5) - log(1:elite_count);
    weights = weights ./ sum(weights);
    cov_matrix = centered' * (centered .* weights') + diag((get_option(options, 'eda_bridge_iso_scale', 0.00034) .* span) .^ 2 + 1e-16);
    [R, flag] = chol(cov_matrix, 'upper');
    if flag ~= 0
        R = diag(sqrt(max(diag(cov_matrix), 1e-16)));
    end
    cov_scale = get_option(options, 'eda_bridge_cov_scale', 0.045);
    iso_scale = get_option(options, 'eda_bridge_block_iso_scale', 0.00055) .* span;
    block_rate = get_option(options, 'eda_bridge_block_rate', max(0.035, 5 / D));
    for row = start_row:NP
        if rand() < 0.58
            child = bridge_center + cov_scale .* randn(1, D) * R;
        else
            donor = sample_elite(elite_sets);
            child = 0.74 .* best + 0.26 .* donor;
            mask = rand(1, D) < block_rate;
            if ~any(mask)
                mask(randi(D)) = true;
            end
            if rand() < 0.45
                noise = tan(pi * (rand(1, D) - 0.5));
                noise = min(max(noise, -7), 7);
            else
                noise = randn(1, D);
            end
            child(mask) = child(mask) + noise(mask) .* iso_scale(mask);
        end
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

if mode == "hses_bridge"
    elite_count = min(size(all_elites, 1), get_option(options, 'hses_bridge_elite_count', max(18, round(0.26 * NP))));
    bridge_center = weighted_elite_center(all_elites, elite_count);
    bridge_center = (1 - get_option(options, 'hses_bridge_best_blend', 0.26)) .* bridge_center + ...
        get_option(options, 'hses_bridge_best_blend', 0.26) .* best;
    centered = all_elites(1:elite_count, :) - bridge_center;
    weights = log(elite_count + 0.5) - log(1:elite_count);
    weights = weights(:) ./ sum(weights);
    diag_sigma = sqrt(max(weights' * (centered .^ 2), 0)) + get_option(options, 'hses_bridge_sigma', 0.0022) .* span;
    diag_sigma = max(diag_sigma, get_option(options, 'hses_bridge_min_sigma', 1e-7) .* max(1, span));
    cov_matrix = centered' * (centered .* weights) + diag((get_option(options, 'hses_bridge_cov_ridge', 0.30) .* diag_sigma) .^ 2 + 1e-18);
    cov_matrix = (cov_matrix + cov_matrix') ./ 2;
    [R, flag] = chol(cov_matrix, 'upper');
    if flag ~= 0
        R = diag(max(diag_sigma, 1e-12));
    end
    cov_rate = get_option(options, 'hses_bridge_cov_rate', 0.44);
    uni_rate = get_option(options, 'hses_bridge_uni_rate', 0.40);
    for row = start_row:NP
        cov_sample = bridge_center + get_option(options, 'hses_bridge_cov_scale', 0.70) .* randn(1, D) * R;
        uni_sample = bridge_center + get_option(options, 'hses_bridge_uni_scale', 0.92) .* randn(1, D) .* diag_sigma;
        mode_draw = rand();
        if mode_draw < cov_rate
            child = cov_sample;
        elseif mode_draw < cov_rate + uni_rate
            child = uni_sample;
        else
            mask = rand(1, D) < get_option(options, 'hses_bridge_hybrid_mask_rate', 0.38);
            if ~any(mask)
                mask(randi(D)) = true;
            end
            child = cov_sample;
            child(mask) = uni_sample(mask);
        end
        if rand() < get_option(options, 'hses_bridge_elite_recomb_rate', 0.20)
            donor = all_elites(randi(elite_count), :);
            mask = rand(1, D) < get_option(options, 'hses_bridge_elite_mask_rate', 0.20);
            child(mask) = donor(mask);
        end
        if rand() < get_option(options, 'hses_bridge_best_pull_rate', 0.24)
            pull = get_option(options, 'hses_bridge_best_pull', 0.16) .* rand();
            child = child + pull .* (best - child);
        end
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

if mode == "bbo_tlbo_bridge"
    elite_count = min(size(all_elites, 1), get_option(options, 'bbo_bridge_elite_count', max(16, round(0.24 * NP))));
    bridge_center = weighted_elite_center(all_elites, elite_count);
    mean_x = mean(all_elites(1:elite_count, :), 1);
    teacher = best;
    block_min = get_option(options, 'bbo_tlbo_block_min', max(4, round(0.04 * D)));
    block_max = get_option(options, 'bbo_tlbo_block_max', max(block_min, round(0.12 * D)));
    block_passes = get_option(options, 'bbo_tlbo_block_passes', 2);
    teacher_scale = get_option(options, 'bbo_tlbo_teacher_scale', 0.46);
    learner_scale = get_option(options, 'bbo_tlbo_learner_scale', 0.22);
    mutation_scale = get_option(options, 'bbo_tlbo_mutation_scale', 0.20);
    for row = start_row:NP
        rank = (row - start_row) / max(1, NP - start_row);
        if rand() < 0.54
            child = population(row, :);
        else
            child = sample_elite(elite_sets);
        end
        immigration_rate = min(0.88, get_option(options, 'bbo_immigration_base', 0.24) + ...
            get_option(options, 'bbo_immigration_span', 0.54) * rank);
        mutation_rate = min(0.48, get_option(options, 'bbo_mutation_base', 0.08) + ...
            get_option(options, 'bbo_mutation_span', 0.18) * rank);
        for pass = 1:block_passes
            block_len = min(D, max(1, randi([block_min, block_max])));
            dims = randperm(D, block_len);
            if rand() < immigration_rate
                source = sample_elite(elite_sets);
                child(dims) = source(dims);
            end
            if rand() < get_option(options, 'tlbo_bridge_rate', 0.72)
                teaching_factor = 1 + double(rand() < 0.5);
                step = teacher_scale .* rand(1, block_len) .* (teacher(dims) - teaching_factor .* mean_x(dims));
                child(dims) = child(dims) + step;
            end
            if rand() < get_option(options, 'tlbo_learner_rate', 0.42)
                donor_a = sample_elite(elite_sets);
                donor_b = sample_elite(elite_sets);
                child(dims) = child(dims) + learner_scale .* rand(1, block_len) .* (donor_a(dims) - donor_b(dims));
            end
            if rand() < mutation_rate
                if rand() < 0.45
                    noise = tan(pi * (rand(1, block_len) - 0.5));
                    noise = min(max(noise, -7), 7);
                else
                    noise = randn(1, block_len);
                end
                child(dims) = child(dims) + mutation_scale .* noise .* radius(dims);
            end
        end
        if rand() < get_option(options, 'bbo_bridge_center_blend_rate', 0.34)
            blend = get_option(options, 'bbo_bridge_center_blend', 0.22) * rand();
            child = (1 - blend) .* child + blend .* bridge_center;
        end
        if rand() < get_option(options, 'bbo_bridge_best_pull_rate', 0.28)
            pull = get_option(options, 'bbo_bridge_best_pull', 0.18) * rand();
            child = child + pull .* (teacher - child);
        end
        if rand() < get_option(options, 'bbo_linewell_rate', 0)
            donor_a = all_elites(randi(elite_count), :);
            donor_b = all_elites(randi(elite_count), :);
            direction = donor_a - donor_b;
            if norm(direction) < eps || rand() < get_option(options, 'bbo_linewell_sparse_rate', 0.42)
                mask = rand(1, D) < get_option(options, 'bbo_linewell_block_rate', max(0.035, 5 / D));
                if ~any(mask)
                    mask(randi(D)) = true;
                end
                direction(~mask) = 0;
                if norm(direction) < eps
                    direction(mask) = randn(1, sum(mask)) .* span(mask);
                end
            end
            direction = direction ./ max(norm(direction), eps);
            step_scales = get_option(options, 'bbo_linewell_step_scales', [0.0020, 0.0040, 0.0070]);
            jump = step_scales(randi(numel(step_scales))) .* norm(span);
            if rand() < 0.5
                jump = -jump;
            end
            well_child = teacher + jump .* direction;
            well_blend = get_option(options, 'bbo_linewell_blend', 0.62);
            child = (1 - well_blend) .* child + well_blend .* well_child;
        end
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

if mode == "latent_elite_de_blx"
    top_rate = get_option(options, 'latent_top_rate', 0.36);
    top_sets = cell(size(elite_sets));
    for s = 1:numel(elite_sets)
        take = max(3, round(top_rate * size(elite_sets{s}, 1)));
        take = min(size(elite_sets{s}, 1), take);
        top_sets{s} = elite_sets{s}(1:take, :);
    end
    secondary_best = best;
    if numel(scouts) > 1
        secondary_best = scouts{2}.best_position;
    end
    for row = start_row:NP
        donor_a = sample_ranked_elite(elite_sets, top_sets, get_option(options, 'latent_top_sample_rate', 0.72));
        donor_b = sample_ranked_elite(elite_sets, top_sets, get_option(options, 'latent_top_sample_rate', 0.72));
        donor_c = sample_ranked_elite(elite_sets, top_sets, get_option(options, 'latent_top_sample_rate', 0.72));
        op = mod(row - start_row, 3) + 1;
        switch op
            case 1
                alpha = -0.16 + 1.34 * rand(1, D);
                child = donor_a + alpha .* (donor_b - donor_a);
                child = child + randn(1, D) .* (get_option(options, 'latent_blx_noise', 0.22) .* radius);
            case 2
                F = 0.26 + 0.46 * rand();
                G = 0.06 + 0.20 * rand();
                H = 0.04 + 0.16 * rand();
                child = donor_a + F .* (best - donor_a) + G .* (secondary_best - donor_a) + H .* (donor_b - donor_c);
                mask = rand(1, D) < get_option(options, 'latent_center_block_rate', 0.12);
                if any(mask)
                    child(mask) = center(mask) + randn(1, nnz(mask)) .* (0.65 .* radius(mask));
                end
            otherwise
                pull = 0.18 + 0.36 * rand();
                diff = donor_b - donor_c;
                child = (1 - pull) .* donor_a + pull .* center + (0.08 + 0.18 * rand()) .* diff;
                block_rate = get_option(options, 'latent_microblock_rate', max(0.045, 5 / D));
                mask = rand(1, D) < block_rate;
                if ~any(mask)
                    mask(randi(D)) = true;
                end
                child(mask) = best(mask) + randn(1, nnz(mask)) .* (get_option(options, 'latent_micro_sigma', 0.42) .* radius(mask));
        end
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

if mode == "scout_disagreement_de_blx"
    if numel(scouts) > 1
        secondary_best = scouts{2}.best_position;
    else
        secondary_best = center;
    end
    scout_delta = best - secondary_best;
    delta_scale = max(abs(scout_delta), get_option(options, 'disagreement_min_radius', 0.15) .* radius);
    for row = start_row:NP
        donor_a = sample_elite(elite_sets);
        donor_b = sample_elite(elite_sets);
        donor_c = sample_elite(elite_sets);
        op = mod(row - start_row, 4) + 1;
        if op <= 2
            F = 0.24 + 0.46 * rand();
            G = 0.08 + 0.22 * rand();
            line = (-0.18 + 0.72 * rand()) .* scout_delta;
            child = donor_a + F .* (best - donor_a) + G .* (donor_b - donor_c) + line;
        elseif op == 3
            alpha = -0.18 + 1.36 * rand(1, D);
            child = donor_a + alpha .* (donor_b - donor_a);
            mask = rand(1, D) < get_option(options, 'disagreement_block_rate', max(0.055, 6 / D));
            if any(mask)
                child(mask) = child(mask) + (-0.35 + 0.70 * rand(1, nnz(mask))) .* scout_delta(mask);
            end
            child = child + randn(1, D) .* (get_option(options, 'disagreement_blx_noise', 0.18) .* radius);
        else
            child = 0.5 .* (best + secondary_best);
            mask = rand(1, D) < get_option(options, 'disagreement_block_rate', max(0.055, 6 / D));
            if ~any(mask)
                mask(randi(D)) = true;
            end
            step = tan(pi * (rand(1, nnz(mask)) - 0.5));
            step = min(max(step, -5), 5);
            child(mask) = child(mask) + step .* (get_option(options, 'disagreement_cauchy_scale', 0.10) .* delta_scale(mask));
        end
        population(row, :) = min(max(child, lb), ub);
    end
    population(1, :) = min(max(primary.best_position, lb), ub);
    if numel(scouts) > 1
        population(2, :) = min(max(scouts{2}.best_position, lb), ub);
    end
    return;
end

for row = start_row:NP
    donor_a = sample_elite(elite_sets);
    donor_b = sample_elite(elite_sets);
    donor_c = sample_elite(elite_sets);
    if mode == "block_elite_de_blx"
        child = donor_a;
        block_min = get_option(options, 'block_blx_min', max(4, round(0.05 * D)));
        block_max = get_option(options, 'block_blx_max', max(block_min, round(0.14 * D)));
        block_len = min(D, max(1, randi([block_min, block_max])));
        cols = randperm(D, block_len);
        if rand() < 0.55
            alpha = -0.10 + 1.20 * rand(1, block_len);
            child(cols) = donor_a(cols) + alpha .* (donor_b(cols) - donor_a(cols));
        else
            F = 0.28 + 0.42 * rand();
            G = 0.08 + 0.20 * rand();
            child(cols) = donor_a(cols) + F .* (best(cols) - donor_a(cols)) + G .* (donor_b(cols) - donor_c(cols));
        end
        if rand() < 0.35
            child(cols) = child(cols) + randn(1, block_len) .* (0.22 .* radius(cols));
        end
        if rand() < 0.18
            micro_len = min(D, max(1, round(0.04 * D)));
            micro = randperm(D, micro_len);
            child(micro) = center(micro) + randn(1, micro_len) .* (0.35 .* radius(micro));
        end
        population(row, :) = min(max(child, lb), ub);
        continue;
    end
    op = mod(row - start_row, 4) + 1;
    if mode == "elite_de_blx"
        op = min(op, 2);
    elseif mode == "bestpoint_elite_de" || mode == "secondary_bestpoint_elite_de"
        op = min(op, 2);
    elseif mode == "soft_obl_snap"
        op = max(2, op);
    end

    switch op
        case 1
            alpha = -0.12 + 1.24 * rand(1, D);
            child = donor_a + alpha .* (donor_b - donor_a);
            child = child + randn(1, D) .* (0.30 .* radius);
        case 2
            F = 0.30 + 0.45 * rand();
            G = 0.08 + 0.22 * rand();
            child = donor_a + F .* (best - donor_a) + G .* (donor_b - donor_c);
            mask = rand(1, D) < 0.18;
            child(mask) = center(mask) + randn(1, nnz(mask)) .* radius(mask);
        case 3
            q = 0.25 + 0.55 * rand(1, D);
            opposite = domain_center + q .* (domain_center - donor_a);
            mask = rand(1, D) < 0.28;
            child = donor_a;
            child(mask) = opposite(mask);
            child = child + randn(1, D) .* (0.18 .* radius);
        otherwise
            grid = max(1e-12, grid_scale .* span);
            snapped = round(donor_a ./ grid) .* grid;
            mask = rand(1, D) < 0.32;
            child = donor_a;
            child(mask) = snapped(mask);
            child = child + randn(1, D) .* (0.12 .* radius);
    end
    population(row, :) = min(max(child, lb), ub);
end
if get_option(options, 'guarded_nullspace_infill', false)
    population = guarded_nullspace_infill(population, scouts, radius_scale, lb, ub, options);
end
population(1, :) = min(max(primary.best_position, lb), ub);
if numel(scouts) > 1
    population(2, :) = min(max(scouts{2}.best_position, lb), ub);
end
end

function population = guarded_nullspace_infill(population, scouts, radius_scale, lb, ub, options)
[NP, D] = size(population);
replace_count = min(NP - 3, max(2, round(get_option(options, 'nullspace_infill_rate', 0.09) * NP)));
if replace_count <= 0
    return;
end
span = ub - lb;
radius = radius_scale .* span;
elite_sets = cellfun(@get_elites, scouts, 'UniformOutput', false);
all_elites = vertcat(elite_sets{:});
if size(all_elites, 1) < 4
    return;
end
primary = scouts{1};
best = primary.best_position;
normed = (all_elites - mean(all_elites, 1)) ./ max(1e-12, span);
dim_std = std(normed, 0, 1);
[~, dim_order] = sort(dim_std, 'ascend');
pool_count = max(6, round(get_option(options, 'nullspace_dim_pool_rate', 0.42) * D));
dim_pool = dim_order(1:min(D, pool_count));
block_min = max(2, round(get_option(options, 'nullspace_block_min_rate', 0.035) * D));
block_max = max(block_min, round(get_option(options, 'nullspace_block_max_rate', 0.10) * D));
start_row = NP - replace_count + 1;
for row = start_row:NP
    if rand() < 0.55
        child = best;
    else
        child = sample_elite(elite_sets);
    end
    block_len = min(numel(dim_pool), randi([block_min, block_max]));
    dims = dim_pool(randperm(numel(dim_pool), block_len));
    donor_a = sample_elite(elite_sets);
    donor_b = sample_elite(elite_sets);
    donor_c = sample_elite(elite_sets);
    if rand() < get_option(options, 'nullspace_blx_rate', 0.45)
        lo = min(donor_a(dims), donor_b(dims));
        hi = max(donor_a(dims), donor_b(dims));
        width = max(hi - lo, 1e-12 .* span(dims));
        alpha = get_option(options, 'nullspace_blx_alpha', 0.18);
        child(dims) = lo - alpha .* width + rand(1, block_len) .* ((1 + 2 * alpha) .* width);
    else
        F = get_option(options, 'nullspace_de_weight', 0.34) + 0.18 * rand();
        pull = get_option(options, 'nullspace_best_pull', 0.18) * rand();
        child(dims) = child(dims) + F .* (donor_a(dims) - donor_b(dims)) + ...
            pull .* (best(dims) - child(dims));
        if rand() < 0.28
            child(dims) = child(dims) + 0.16 .* (donor_c(dims) - child(dims));
        end
    end
    if rand() < get_option(options, 'nullspace_noise_rate', 0.34)
        child(dims) = child(dims) + randn(1, block_len) .* ...
            (get_option(options, 'nullspace_noise_scale', 0.24) .* radius(dims));
    end
    population(row, :) = min(max(child, lb), ub);
end
end

function elite = sample_elite(elite_sets)
set_id = randi(numel(elite_sets));
pool = elite_sets{set_id};
elite = pool(randi(size(pool, 1)), :);
end

function elite = sample_ranked_elite(elite_sets, top_sets, top_rate)
if rand() < top_rate
    set_id = randi(numel(top_sets));
    pool = top_sets{set_id};
else
    set_id = randi(numel(elite_sets));
    pool = elite_sets{set_id};
end
elite = pool(randi(size(pool, 1)), :);
end

function center = weighted_elite_center(elites, elite_count)
elite_count = min(size(elites, 1), max(1, elite_count));
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites(1:elite_count, :);
end

function [population, curve, eval_count, iter_count] = success_filtered_fusion_population(problem, scouts, NP, radius_scale, options)
lb = problem.lb;
ub = problem.ub;
pool_multiplier = get_option(options, 'fusion_pool_multiplier', 3.0);
pool_size = max(NP, round(pool_multiplier * NP));
modes = ["coordinate", "elite_de_blx", "soft_obl_snap"];
segments = cell(1, numel(modes));
remaining = pool_size;
for m = 1:numel(modes)
    seg_size = floor(pool_size / numel(modes));
    if m == numel(modes)
        seg_size = remaining;
    end
    remaining = remaining - seg_size;
    mode_options = options;
    mode_options.crossover_mode = char(modes(m));
    if numel(scouts) > 2
        segments{m} = crossover_population_multi(scouts, seg_size, radius_scale, lb, ub, mode_options);
    else
        segments{m} = crossover_population(scouts{1}, scouts{2}, seg_size, radius_scale, lb, ub, mode_options);
    end
end
pool = vertcat(segments{:});
for s = 1:min(numel(scouts), size(pool, 1))
    pool(s, :) = scouts{s}.best_position;
end
values = SOP_cec_evaluate(pool, problem);
eval_count = numel(values);
iter_count = 1;
[values, order] = sort(values(:));
pool = pool(order, :);
elite_keep = min(NP, max(2, round(get_option(options, 'fusion_fitness_keep_rate', 0.62) * NP)));
population = pool(1:elite_keep, :);
if size(population, 1) < NP
    candidate_limit = min(size(pool, 1), max(NP, round(get_option(options, 'fusion_diversity_pool_rate', 1.8) * NP)));
    candidate_pool = pool(1:candidate_limit, :);
    span = max(ub - lb, eps);
    while size(population, 1) < NP
        remaining_idx = (size(population, 1) + 1):size(candidate_pool, 1);
        if isempty(remaining_idx)
            population(end + 1, :) = pool(randi(size(pool, 1)), :); %#ok<AGROW>
            continue;
        end
        selected = population;
        distances = zeros(numel(remaining_idx), 1);
        for k = 1:numel(remaining_idx)
            x = candidate_pool(remaining_idx(k), :);
            distances(k) = min(sqrt(mean(((selected - x) ./ span) .^ 2, 2)));
        end
        fitness_rank = remaining_idx(:) ./ max(1, size(candidate_pool, 1));
        score = get_option(options, 'fusion_diversity_weight', 0.42) .* distances - ...
            (1 - get_option(options, 'fusion_diversity_weight', 0.42)) .* fitness_rank;
        [~, idx] = max(score);
        population(end + 1, :) = candidate_pool(remaining_idx(idx), :); %#ok<AGROW>
    end
end
population = min(max(population(1:NP, :), lb), ub);
curve = cummin(values(1:min(numel(values), max(1, NP))));
end

function elites = get_elites(result)
if isfield(result, 'final_population') && ~isempty(result.final_population)
    elites = result.final_population;
else
    elites = result.best_position;
end
end

function options = apply_jso_rsp(options)
options.include_center = get_option(options, 'jso_rsp_include_center', false);
options.ranked_r1 = true;
options.rank_pressure = get_option(options, 'rank_pressure', 1.7);
options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
options.p_rate_start = get_option(options, 'p_rate_start', 0.20);
options.p_rate_end = get_option(options, 'p_rate_end', 0.040);
options.weight_start = get_option(options, 'weight_start', 0.62);
options.weight_end = get_option(options, 'weight_end', 1.36);
options.archive_factor_start = get_option(options, 'archive_factor_start', 1.8);
options.archive_factor_end = get_option(options, 'archive_factor_end', 3.2);
end

function label = label_for(key)
switch key
    case "operator_pool"
        label = 'DE/GSK/RIME/CMA operator-pool';
    case "operator_pool_gsk"
        label = 'GSK/RIME-biased operator-pool';
    case "operator_pool_cma"
        label = 'DE/CMA-biased operator-pool';
    case "jso_rsp"
        label = 'jSO ranked-r1/RSP';
    case "code_epsde"
        label = 'CoDE/EPSDE multi-strategy DE pool';
    case "quantum_snap"
        label = 'Quantum/opposition snap Differential Evolution';
    case "lshade_cma"
        label = 'L-SHADE-CMA';
    case "lshade"
        label = 'L-SHADE';
    case "spacma"
        label = 'L-SHADE/SPACMA cooperative archive';
    case "gsk"
        label = 'Gaining-Sharing Knowledge (GSK)';
    otherwise
        label = upper(char(key));
end
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

function count = sum_cell_eval(results)
count = 0;
for i = 1:numel(results)
    count = count + results{i}.evaluation_count;
end
end

function count = sum_cell_iter(results)
count = 0;
for i = 1:numel(results)
    count = count + results{i}.iteration;
end
end

function curve = cell_curve_concat(results, use_raw)
curve = [];
for i = 1:numel(results)
    if use_raw
        part = raw_curve_for(results{i});
    elseif isfield(results{i}, 'convergence_curve') && ~isempty(results{i}.convergence_curve)
        part = results{i}.convergence_curve(:);
    else
        part = results{i}.record_value;
    end
    curve = [curve; part(:)]; %#ok<AGROW>
end
end

function representatives = select_clearing_representatives(pool, values, target_count, radius_scale, span)
[values, order] = sort(values(:));
pool = pool(order, :);
span = max(span, eps);
representatives = zeros(0, size(pool, 2));
for i = 1:size(pool, 1)
    x = pool(i, :);
    if isempty(representatives)
        representatives = x;
    else
        distances = sqrt(mean(((representatives - x) ./ span) .^ 2, 2));
        if min(distances) >= radius_scale
            representatives(end + 1, :) = x; %#ok<AGROW>
        end
    end
    if size(representatives, 1) >= target_count
        break;
    end
end
if size(representatives, 1) < target_count
    for i = 1:size(pool, 1)
        x = pool(i, :);
        if ~ismembertol(x, representatives, 1e-12, 'ByRows', true)
            representatives(end + 1, :) = x; %#ok<AGROW>
        end
        if size(representatives, 1) >= target_count
            break;
        end
    end
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
