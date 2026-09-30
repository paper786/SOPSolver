function result = SOP_agent_numeric_guided_refine(problem, seed, options)
% Short numerical-optimization guidance refiner for black-box CEC calls.
%
% Supported modes:
%   "bfgs"    - finite-difference quasi-Newton/BFGS with backtracking.
%   "simplex" - subspace Nelder-Mead simplex search around the incumbent.
%   "powell"  - derivative-free Powell/direct-search line probing.
%
% These are used as guidance operators after a metaheuristic has found a
% basin. They do not call CEC internals or analytical derivatives.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isempty(seed)
    rng(double(seed), 'twister');
else
    rng('shuffle');
end

t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
max_fes = get_option(options, 'max_fes', 20000);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
mode = lower(string(get_option(options, 'mode', 'bfgs')));

[best_x, best_raw, population, fitness, eval_count] = initialize_state(problem, options, lb, ub, span);
curve = best_raw;
switch mode
    case "simplex"
        [best_x, best_raw, extra_evals, extra_curve] = simplex_refine(problem, best_x, best_raw, population, fitness, lb, ub, span, max_fes - eval_count, max_runtime_sec - toc(t_start), options);
        label = 'Nelder-Mead simplex numerical guidance';
    case "powell"
        [best_x, best_raw, extra_evals, extra_curve] = powell_refine(problem, best_x, best_raw, population, fitness, lb, ub, span, max_fes - eval_count, max_runtime_sec - toc(t_start), options);
        label = 'Powell/direct-search numerical guidance';
    otherwise
        [best_x, best_raw, extra_evals, extra_curve] = bfgs_refine(problem, best_x, best_raw, lb, ub, span, max_fes - eval_count, max_runtime_sec - toc(t_start), options);
        label = 'Finite-difference quasi-Newton/BFGS guidance';
end
eval_count = eval_count + extra_evals;
curve = [curve(:); extra_curve(:)];

runtime = toc(t_start);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = max(1, numel(curve));
result.population_num = get_option(options, 'population_num', size(population, 1));
result.evaluation_count = eval_count;
result.algorithm_combination = label;
result.combination_number = 2;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = population;
result.final_fitness = fitness;
end

function [best_x, best_raw, population, fitness, eval_count] = initialize_state(problem, options, lb, ub, span)
D = problem.dimension;
NP = get_option(options, 'population_num', 80);
population = [];
if isfield(options, 'initial_population') && ~isempty(options.initial_population) && size(options.initial_population, 2) == D
    population = options.initial_population;
end
initial_point = get_option(options, 'initial_point', []);
if isempty(population)
    if ~isempty(initial_point)
        center = min(max(initial_point(:)', lb), ub);
        radius = make_radius(get_option(options, 'initial_radius', 0.0025), span, D);
        population = repmat(center, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
        if get_option(options, 'initial_cauchy', false)
            cauchy = tan(pi * (rand(NP, D) - 0.5));
            cauchy = min(max(cauchy, -8), 8);
            mask = rand(NP, D) < 0.18;
            center_matrix = repmat(center, NP, 1);
            radius_matrix = repmat(radius, NP, 1);
            population(mask) = center_matrix(mask) + cauchy(mask) .* radius_matrix(mask);
        end
        population(1, :) = center;
    else
        population = lb + rand(NP, D) .* span;
    end
end
if size(population, 1) > NP
    population = population(1:NP, :);
elseif size(population, 1) < NP
    population = [population; lb + rand(NP - size(population, 1), D) .* span]; %#ok<AGROW>
end
population = min(max(population, lb), ub);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
if ~isempty(initial_point)
    f0 = SOP_cec_evaluate(min(max(initial_point(:)', lb), ub), problem);
    eval_count = eval_count + 1;
    population(1, :) = min(max(initial_point(:)', lb), ub);
    fitness(1) = f0;
end
[best_raw, idx] = min(fitness);
best_x = population(idx, :);
[fitness, order] = sort(fitness);
population = population(order, :);
end

function [best_x, best_raw, eval_count, curve] = bfgs_refine(problem, best_x, best_raw, lb, ub, span, max_fes, max_runtime_sec, options)
D = numel(best_x);
eval_count = 0;
curve = [];
if max_fes < 2 * D + 2 || max_runtime_sec <= 1
    return;
end
H = eye(D);
h = get_option(options, 'fd_step_scale', 8e-5) .* max(1, abs(span));
[g, gevals] = finite_gradient(problem, best_x, best_raw, h, lb, ub);
eval_count = eval_count + gevals;
max_iters = get_option(options, 'bfgs_iters', 4);
step_cap = get_option(options, 'bfgs_step_cap', 0.0025) .* max(1, abs(span));
for iter = 1:max_iters
    if eval_count >= max_fes || toc_since(max_runtime_sec) || norm(g) <= eps
        break;
    end
    direction = -(H * g(:))';
    direction = cap_step(direction, step_cap);
    if norm(direction) <= eps
        direction = -g ./ (norm(g) + eps);
        direction = cap_step(direction, step_cap);
    end
    [trial_x, trial_raw, ls_evals, improved] = backtracking(problem, best_x, best_raw, direction, lb, ub, max_fes - eval_count, max_runtime_sec, options);
    eval_count = eval_count + ls_evals;
    if ~improved
        H = 0.35 .* H + 0.65 .* eye(D);
        step_cap = 0.55 .* step_cap;
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
        continue;
    end
    s = (trial_x - best_x)';
    old_g = g(:);
    best_x = trial_x;
    best_raw = trial_raw;
    curve(end + 1, 1) = best_raw; %#ok<AGROW>
    if eval_count + 2 * D + 1 > max_fes || toc_since(max_runtime_sec)
        break;
    end
    [g, gevals] = finite_gradient(problem, best_x, best_raw, h, lb, ub);
    eval_count = eval_count + gevals;
    y = g(:) - old_g;
    ys = y' * s;
    if ys > 1e-12 && all(isfinite(y))
        rho = 1 / ys;
        I = eye(D);
        H = (I - rho * s * y') * H * (I - rho * y * s') + rho * (s * s');
        H = 0.5 .* (H + H');
    else
        H = eye(D);
    end
end
end

function [g, eval_count] = finite_gradient(problem, x, f0, h, lb, ub)
D = numel(x);
points = zeros(2 * D, D);
for j = 1:D
    xp = x; xm = x;
    xp(j) = min(ub(j), x(j) + h(j));
    xm(j) = max(lb(j), x(j) - h(j));
    points(2 * j - 1, :) = xp;
    points(2 * j, :) = xm;
end
values = SOP_cec_evaluate(points, problem);
eval_count = numel(values);
g = zeros(1, D);
for j = 1:D
    fp = values(2 * j - 1);
    fm = values(2 * j);
    denom = points(2 * j - 1, j) - points(2 * j, j);
    if abs(denom) <= eps
        g(j) = 0;
    else
        g(j) = (fp - fm) / denom;
    end
end
if ~all(isfinite(g))
    g(~isfinite(g)) = 0;
end
if nargin >= 3 && ~isempty(f0) && ~isfinite(f0)
    g(:) = 0;
end
end

function [best_x, best_raw, eval_count, curve] = simplex_refine(problem, best_x, best_raw, population, fitness, lb, ub, span, max_fes, max_runtime_sec, options)
D = numel(best_x);
eval_count = 0;
curve = [];
k = min(D, get_option(options, 'simplex_dims', 14));
dims = choose_active_dims(population, fitness, k);
step = get_option(options, 'simplex_step_scale', 0.0018) .* span(dims);
simplex = repmat(best_x, k + 1, 1);
for i = 1:k
    simplex(i + 1, dims(i)) = min(max(simplex(i + 1, dims(i)) + step(i), lb(dims(i))), ub(dims(i)));
end
values = SOP_cec_evaluate(simplex, problem);
eval_count = eval_count + numel(values);
[values, order] = sort(values);
simplex = simplex(order, :);
if values(1) < best_raw
    best_raw = values(1);
    best_x = simplex(1, :);
end
alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5;
max_iter = get_option(options, 'simplex_iters', 90);
for iter = 1:max_iter
    if eval_count >= max_fes || toc_since(max_runtime_sec)
        break;
    end
    centroid = mean(simplex(1:k, dims), 1);
    worst = simplex(end, dims);
    xr = simplex(end, :);
    xr(dims) = centroid + alpha .* (centroid - worst);
    xr = min(max(xr, lb), ub);
    fr = SOP_cec_evaluate(xr, problem);
    eval_count = eval_count + 1;
    if fr < values(1) && eval_count < max_fes
        xe = simplex(end, :);
        xe(dims) = centroid + gamma .* (xr(dims) - centroid);
        xe = min(max(xe, lb), ub);
        fe = SOP_cec_evaluate(xe, problem);
        eval_count = eval_count + 1;
        if fe < fr
            simplex(end, :) = xe; values(end) = fe;
        else
            simplex(end, :) = xr; values(end) = fr;
        end
    elseif fr < values(end - 1)
        simplex(end, :) = xr; values(end) = fr;
    else
        xc = simplex(end, :);
        xc(dims) = centroid + rho .* (worst - centroid);
        xc = min(max(xc, lb), ub);
        fc = SOP_cec_evaluate(xc, problem);
        eval_count = eval_count + 1;
        if fc < values(end)
            simplex(end, :) = xc; values(end) = fc;
        else
            for i = 2:k + 1
                simplex(i, dims) = simplex(1, dims) + sigma .* (simplex(i, dims) - simplex(1, dims));
            end
            simplex = min(max(simplex, lb), ub);
            values = SOP_cec_evaluate(simplex, problem);
            eval_count = eval_count + numel(values);
        end
    end
    [values, order] = sort(values);
    simplex = simplex(order, :);
    if values(1) < best_raw
        best_raw = values(1);
        best_x = simplex(1, :);
    end
    curve(end + 1, 1) = best_raw; %#ok<AGROW>
end
end

function [best_x, best_raw, eval_count, curve] = powell_refine(problem, best_x, best_raw, population, fitness, lb, ub, span, max_fes, max_runtime_sec, options)
D = numel(best_x);
eval_count = 0;
curve = [];
k = min(D, get_option(options, 'powell_dirs', 24));
dims = choose_active_dims(population, fitness, k);
directions = eye(D);
directions = directions(dims, :);
if get_option(options, 'powell_random_dirs', true)
    random_dirs = randn(max(2, round(0.25 * k)), D);
    random_dirs = random_dirs ./ max(eps, vecnorm(random_dirs, 2, 2));
    directions = [directions; random_dirs]; %#ok<AGROW>
end
step0 = get_option(options, 'powell_step_scale', 0.0022) .* median(abs(span));
passes = get_option(options, 'powell_passes', 3);
for pass = 1:passes
    step = step0 * (0.45 ^ (pass - 1));
    for d = 1:size(directions, 1)
        if eval_count >= max_fes || toc_since(max_runtime_sec)
            break;
        end
        dir = directions(d, :);
        dir = dir ./ (norm(dir) + eps);
        alphas = step .* [-1.0, -0.5, 0.5, 1.0];
        candidates = repmat(best_x, numel(alphas), 1) + alphas(:) .* dir;
        candidates = min(max(candidates, lb), ub);
        if eval_count + size(candidates, 1) > max_fes
            candidates = candidates(1:max_fes - eval_count, :);
        end
        vals = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(vals);
        [trial_raw, idx] = min(vals);
        if trial_raw < best_raw
            previous = best_x;
            best_raw = trial_raw;
            best_x = candidates(idx, :);
            directions(d, :) = best_x - previous;
        end
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end
end

function dims = choose_active_dims(population, fitness, k)
D = size(population, 2);
if isempty(population) || size(population, 1) < 4
    dims = randperm(D, k);
    return;
end
[~, order] = sort(fitness);
elite_count = min(numel(order), max(4, round(0.25 * numel(order))));
elite = population(order(1:elite_count), :);
scores = var(elite, 0, 1);
if all(scores <= eps)
    dims = randperm(D, k);
else
    [~, dim_order] = sort(scores, 'descend');
    dims = dim_order(1:k);
end
end

function [trial_x, trial_raw, eval_count, improved] = backtracking(problem, x, f0, direction, lb, ub, max_evals, max_runtime_sec, options)
eval_count = 0;
improved = false;
trial_x = x;
trial_raw = f0;
alpha = get_option(options, 'line_alpha0', 1.0);
for i = 1:get_option(options, 'line_steps', 8)
    if eval_count >= max_evals || toc_since(max_runtime_sec)
        break;
    end
    candidate = min(max(x + alpha .* direction, lb), ub);
    value = SOP_cec_evaluate(candidate, problem);
    eval_count = eval_count + 1;
    if value < trial_raw
        trial_x = candidate;
        trial_raw = value;
        improved = true;
        return;
    end
    alpha = alpha * 0.45;
end
end

function tf = toc_since(max_runtime_sec)
tf = max_runtime_sec <= 0;
end

function step = cap_step(step, cap)
step = min(max(step, -cap), cap);
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
else
    radius = reshape(radius, 1, []);
    if numel(radius) ~= D
        radius = median(radius(:)) .* ones(1, D);
    end
end
radius = max(radius, eps);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
