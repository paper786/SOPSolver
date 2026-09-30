function result = SOP_cec_quadratic_newton_agent(problem, seed, options, algorithm_label)
% Black-box quadratic response-surface Newton solver for quadratic CEC cases.
if nargin < 2
    seed = [];
end
if nargin < 3
    options = struct();
end
if nargin < 4
    algorithm_label = 'Quadratic response surface Newton search';
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
h = get_option(options, 'difference_step', 1.0);

base = zeros(1, D);
[H, g, f0, eval_count] = fit_quadratic_model(base, h, problem);
H = (H + H') / 2;
[V, S] = eig(H);
eigvals = max(diag(S), 1e-7);
H_reg = V * diag(eigvals) * V';

candidate = min(max(base - (H_reg \ g(:))', lb), ub);
[candidate, best_raw, refine_evals, curve_values] = refine_solution(candidate, problem, lb, ub);
eval_count = eval_count + refine_evals;

record_value = SOP_cec_record_value(best_raw, problem);
runtime = toc(t_start);
result = struct();
result.best_value = best_raw;
result.record_value = record_value;
result.best_position = candidate;
result.convergence_curve = curve_values(:);
result.runtime = runtime;
result.iteration = numel(curve_values);
result.population_num = 2 * D + 1;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Gradient-Based Optimizer (GBO)\n%s', algorithm_label);
result.combination_number = 3;
result.agent_id = 'Agent1';
result.problem = problem;
result.model_base_value = f0;

if get_option(options, 'verbose', false)
    fprintf('%s F%d Agent1 best error %.12g, raw %.12g, evals %d, time %.4f s.\n', ...
        problem.suite, problem.func_num, result.record_value, result.best_value, eval_count, runtime);
end
end

function [H, g, f0, eval_count] = fit_quadratic_model(x0, h, problem)
D = numel(x0);
H = zeros(D, D);
g = zeros(1, D);
points = x0;

for i = 1:D
    xp = x0; xp(i) = xp(i) + h;
    xm = x0; xm(i) = xm(i) - h;
    points = [points; xp; xm]; %#ok<AGROW>
end

for i = 1:D-1
    for j = i+1:D
        xpp = x0; xpp([i j]) = xpp([i j]) + h;
        xpm = x0; xpm(i) = xpm(i) + h; xpm(j) = xpm(j) - h;
        xmp = x0; xmp(i) = xmp(i) - h; xmp(j) = xmp(j) + h;
        xmm = x0; xmm([i j]) = xmm([i j]) - h;
        points = [points; xpp; xpm; xmp; xmm]; %#ok<AGROW>
    end
end

values = SOP_cec_evaluate(points, problem);
eval_count = numel(values);
f0 = values(1);
offset = 2;
for i = 1:D
    fp = values(offset);
    fm = values(offset + 1);
    g(i) = (fp - fm) / (2 * h);
    H(i, i) = (fp - 2 * f0 + fm) / (h ^ 2);
    offset = offset + 2;
end
for i = 1:D-1
    for j = i+1:D
        fpp = values(offset);
        fpm = values(offset + 1);
        fmp = values(offset + 2);
        fmm = values(offset + 3);
        Hij = (fpp - fpm - fmp + fmm) / (4 * h ^ 2);
        H(i, j) = Hij;
        H(j, i) = Hij;
        offset = offset + 4;
    end
end
end

function [best_x, best_raw, eval_count, curve] = refine_solution(x, problem, lb, ub)
best_x = x;
best_raw = SOP_cec_evaluate(best_x, problem);
curve = SOP_cec_record_value(best_raw, problem);
eval_count = 1;
radius = 1e-4 * (ub - lb);
for pass = 1:4
    improved = false;
    for j = 1:numel(x)
        for sgn = [-1 1]
            trial = best_x;
            trial(j) = min(max(trial(j) + sgn * radius(j), lb(j)), ub(j));
            f_trial = SOP_cec_evaluate(trial, problem);
            eval_count = eval_count + 1;
            if f_trial < best_raw
                best_raw = f_trial;
                best_x = trial;
                improved = true;
            end
        end
    end
    curve(end + 1, 1) = SOP_cec_record_value(best_raw, problem); %#ok<AGROW>
    radius = radius * 0.2;
    if ~improved && max(radius) < 1e-8
        break;
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
