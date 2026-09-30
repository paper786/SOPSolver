function result = SOP_agent_fno_lshade_cma(problem, seed, options)
% FNO global far/near scout followed by L-SHADE-CMA exploitation.
%
% FNO is used as a literature-grounded basin finder for composition
% functions; L-SHADE-CMA then performs success-history DE plus covariance
% sampling around the selected basin.
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

fno_fraction = get_option(options, 'fno_fraction', 0.18);
fno_options = options;
fno_options.population_num = get_option(options, 'fno_population_num', max(50, round(0.42 * NP)));
fno_options.batch_size = get_option(options, 'fno_batch_size', min(fno_options.population_num, 12));
fno_options.update_mode = get_option(options, 'fno_update_mode', 'batch');
fno_options.boundary_mode = get_option(options, 'fno_boundary_mode', 'half');
fno_options.include_center = get_option(options, 'fno_include_center', true);
fno_options.max_fes = max(1000, floor(fno_fraction * max_fes));
fno_options.max_runtime_sec = max(1, fno_fraction * max_runtime_sec);
fno_options.verbose = false;
fno = SOP_agent_fno(problem, double(seed), fno_options);

remaining_fes = max(1000, max_fes - fno.evaluation_count);
remaining_time = max(1, max_runtime_sec - toc(t_start));
refine_options = options;
refine_options.max_fes = remaining_fes;
refine_options.max_runtime_sec = remaining_time;
refine_options.population_num = get_option(options, 'local_population_num', max(72, round(0.50 * NP)));
refine_options.initial_point = fno.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.010);
refine_options.initial_cauchy = get_option(options, 'initial_cauchy', true);
refine_options.verbose = false;

if isfield(fno, 'final_population') && ~isempty(fno.final_population)
    refine_options.initial_population = fno_seed_population(fno, refine_options.population_num, problem, options);
    refine_options.preserve_initial_population_after_radius = true;
end
refiner = lower(string(get_option(options, 'refiner', 'lshade_cma')));
if refiner == "lshade"
    refine = SOP_agent_lshade(problem, double(seed) + 7919, refine_options);
    refiner_label = 'L-SHADE success-history DE exploitation';
else
    refine_options.cma_rate = get_option(options, 'cma_rate', 0.16);
    refine_options.elite_rate = get_option(options, 'elite_rate', 0.22);
    refine_options.cma_interval = get_option(options, 'cma_interval', 12);
    refine = SOP_agent_lshade_cma(problem, double(seed) + 7919, refine_options);
    refiner_label = 'L-SHADE-CMA success-history DE and covariance exploitation';
end

if refine.record_value < fno.record_value
    result = refine;
else
    result = fno;
end
result.runtime = toc(t_start);
result.evaluation_count = fno.evaluation_count + refine.evaluation_count;
result.iteration = fno.iteration + refine.iteration;
result.convergence_curve = [fno.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [fno.raw_convergence_curve(:); refine.raw_convergence_curve(:)];
result.algorithm_combination = sprintf('Far and Near Optimization (FNO) far/near basin scout\n%s', refiner_label);
result.combination_number = 2;
result.agent_id = 'Agent1';
result.problem = problem;
end

function population = fno_seed_population(fno, NP, problem, options)
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
D = problem.dimension;
population = repmat(fno.best_position, NP, 1);
elite_count = min(size(fno.final_population, 1), max(4, round(get_option(options, 'fno_seed_elite_rate', 0.35) * NP)));
take = min(NP, elite_count);
population(1:take, :) = fno.final_population(1:take, :);
radius = get_option(options, 'refine_radius', 0.010) .* span;
for row = take + 1:NP
    if row <= round(0.68 * NP)
        donor = fno.final_population(randi(elite_count), :);
        child = 0.70 .* fno.best_position + 0.30 .* donor;
        child = child + randn(1, D) .* (0.55 .* radius);
    else
        donor_a = fno.final_population(randi(elite_count), :);
        donor_b = fno.final_population(randi(elite_count), :);
        child = fno.best_position + (0.20 + 0.35 * rand()) .* (donor_a - donor_b);
        mask = rand(1, D) < max(0.04, 5 / D);
        if any(mask)
            child(mask) = child(mask) + tan(pi * (rand(1, nnz(mask)) - 0.5)) .* (0.16 .* radius(mask));
        end
    end
    population(row, :) = min(max(child, lb), ub);
end
population(1, :) = min(max(fno.best_position, lb), ub);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
