function value = SOP_cec_record_value(raw_value, problem)
% Convert raw benchmark output to the metric used for the local records.
%
% BestRecord.xlsx currently stores CEC2014 values as errors to the official
% bias (the evidence notes say fmin=0). CEC2017 and CEC2019 records are raw
% objective values. This conversion is only for reporting; optimization is
% always performed on raw objective values.
raw_value = double(raw_value);
if isfield(problem, 'record_metric') && strcmp(problem.record_metric, 'cec2014_error')
    value = raw_value - 100 * double(problem.func_num);
else
    value = raw_value;
end
value(abs(value) < 1e-12) = 0;
end
