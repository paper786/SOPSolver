function result = SOP_agent_sade_gaussian_refine(problem, seed, options)
% SaDE followed by elite Gaussian random-walk exploitation.
%
% Literature basis: self-adaptive Differential Evolution plus evolutionary
% strategy style Gaussian neighborhood mutation. The refinement is purely
% stochastic and only uses benchmark objective calls.
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
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
de_options = options;
de_options.max_runtime_sec = 0.72 * max_runtime_sec;
de_options.verbose = false;
base = SOP_agent1_adaptive_de(problem, seed, de_options);
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;

best_x = base.best_position;
best_raw = base.best_value;
eval_count = base.evaluation_count;
curve = base.convergence_curve(:);
samples = get_option(options, 'refine_samples', 80);
refine_iter = get_option(options, 'refine_iter', 1800);
sigma = get_option(options, 'refine_sigma', 0.025) * span;
min_sigma = get_option(options, 'min_sigma', 1e-8) * span;

for iter = 1:refine_iter
    if toc(t_start) >= max_runtime_sec
        break;
    end
    candidates = repmat(best_x, samples, 1) + randn(samples, D) .* repmat(sigma, samples, 1);
    candidates = min(max(candidates, lb), ub);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        sigma = max(0.985 * sigma, min_sigma);
    else
        sigma = max(0.96 * sigma, min_sigma);
    end
    if mod(iter, 20) == 0
        curve(end + 1, 1) = SOP_cec_record_value(best_raw, problem); %#ok<AGROW>
    end
end

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.convergence_curve = curve;
result.runtime = toc(t_start);
result.iteration = base.iteration + refine_iter;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Self-Adaptive Differential Evolution (SaDE)\nElite Gaussian random-walk refinement');
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
