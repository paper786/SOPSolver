function result = SOP_agent_jso_rsp_eigen_refine(problem, seed, options)
% jSO/RSP search followed by elite-covariance stochastic refinement.
%
% The refiner samples objective-only perturbations along covariance
% directions estimated from the final elite population. It is intended for
% rotated or stepped multimodal functions where coordinate-wise local moves
% are brittle.
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
base_fraction = get_option(options, 'base_fraction', 0.82);

base_options = options;
base_options.max_runtime_sec = max(1, base_fraction * max_runtime_sec);
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.verbose = false;
base = SOP_agent_lshade_jso(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
[best_x, best_raw, refine_curve, refine_eval, refine_iter] = eigen_refine(problem, base, ...
    remaining_fes, remaining_time, options);

result = base;
if best_raw < result.best_value
    result.best_value = best_raw;
    result.record_value = SOP_cec_record_value(best_raw, problem);
    result.best_position = best_x;
end
result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + refine_eval;
result.iteration = base.iteration + refine_iter;
result.raw_convergence_curve = [base.raw_convergence_curve(:); refine_curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.algorithm_combination = sprintf('Differential Evolution (DE)\njSO/RSP ranked archive adaptation\nElite covariance stochastic refinement');
result.combination_number = 4;
result.agent_id = 'Agent2';
end

function [best_x, best_raw, curve, eval_count, iter] = eigen_refine(problem, base, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = base.best_position;
best_raw = base.best_value;
eval_count = 0;
iter = 0;
batch = get_option(options, 'eigen_batch', 128);
elite_count = get_option(options, 'eigen_elites', 36);
sigma = get_option(options, 'eigen_sigma', 0.0020);
min_sigma = get_option(options, 'eigen_min_sigma', 1e-7);
top_dirs = get_option(options, 'eigen_top_dirs', min(24, D));
curve = zeros(max(1, ceil(max_fes / max(1, batch))), 1);

if max_fes <= 0 || max_runtime_sec <= 0 || ~isfield(base, 'final_population') || isempty(base.final_population)
    curve = curve([]);
    return;
end

elites = base.final_population(1:min(elite_count, size(base.final_population, 1)), :);
center = mean(elites, 1);
centered = elites - center;
cov_matrix = centered' * centered ./ max(1, size(centered, 1) - 1);
cov_matrix = cov_matrix + diag((1e-6 .* span) .^ 2);
[V, S] = eig((cov_matrix + cov_matrix') ./ 2, 'vector');
[S, order] = sort(max(S, 0), 'descend');
V = V(:, order);
dir_count = max(2, min(top_dirs, size(V, 2)));
V = V(:, 1:dir_count);
dir_scale = sqrt(S(1:dir_count)' + eps);
dir_scale = dir_scale ./ max(dir_scale);
iso_sigma = sigma .* span;
stall = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = repmat(best_x, count, 1);
    for i = 1:count
        active = rand(1, dir_count) < get_option(options, 'eigen_dir_rate', 0.30);
        if ~any(active)
            active(randi(dir_count)) = true;
        end
        coeff = randn(1, dir_count) .* active .* dir_scale .* sigma .* mean(span);
        x = best_x + coeff * V';
        if rand() < 0.35
            mask = rand(1, D) < get_option(options, 'eigen_iso_rate', max(0.02, 3 / D));
            x(mask) = x(mask) + randn(1, sum(mask)) .* iso_sigma(mask);
        end
        candidates(i, :) = min(max(x, lb), ub);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        sigma = max(0.93 * sigma, min_sigma);
        stall = 0;
    else
        sigma = max(0.76 * sigma, min_sigma);
        stall = stall + 1;
        if stall >= 10
            sigma = max(sigma, get_option(options, 'eigen_reset_sigma', 0.0012));
            stall = 0;
        end
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
