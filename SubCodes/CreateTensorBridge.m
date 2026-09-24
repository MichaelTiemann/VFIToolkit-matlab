function TensorFn = CreateTensorBridge(InputFn)

info = functions(InputFn);
func_name = '';

if strcmp(info.type, 'anonymous')
    fn_str = info.function;
    
    % Look for a pattern like @(...) FunctionName(...)
    % Captures the word immediately following the closing parenthesis of the input arguments
    tokens = regexp(fn_str, '@\([^)]*\)\s*([a-zA-Z]\w*)\s*\(', 'tokens');
    
    if ~isempty(tokens)
        extracted_name = tokens{1}{1};
        % Verify the extracted name actually corresponds to an existing .m file
        if exist(extracted_name, 'file') == 2 || exist([extracted_name, '.m'], 'file')
            func_name = extracted_name;
        end
    end
    
    % If it's a pure math expression (no wrapped function file found), use the fallback
    if isempty(func_name)
        % Wrap true anonymous functions in arrayfun to preserve scalar execution logic
        TensorFn = @(varargin) arrayfun(InputFn, varargin{:});
        return;
    end
else
    % Standard named function handle
    func_name = info.function;
end

num_base_args = nargin(func_name);
wrapperName = [func_name, '_AutoBridge'];
fileName = [wrapperName, '.m'];

fid = fopen(fileName, 'w');
if fid == -1
    error('TensorBridge:FileError', 'Could not create harness file %s', fileName);
end

fprintf(fid, 'function F = %s(varargin)\n', wrapperName);
fprintf(fid, '    %% AUTO-GENERATED TENSOR BRIDGE HARNESS (Variadic)\n');
fprintf(fid, '    num_expected = %d;\n', num_base_args);
fprintf(fid, '    num_provided = length(varargin);\n\n');

fprintf(fid, '    if num_provided > num_expected\n');
fprintf(fid, '        args_to_pass = varargin(1:num_expected);\n');
fprintf(fid, '    elseif num_provided < num_expected\n');
fprintf(fid, '        warning(''TensorBridge:MissingArgs'', ''%%s expected %%d arguments but received %%d. Padding with zeros. Check parameter introspection!'', ''%s'', num_expected, num_provided);\n', func_name);
fprintf(fid, '        padding = num2cell(zeros(1, num_expected - num_provided));\n');
fprintf(fid, '        args_to_pass = [varargin, padding];\n');
fprintf(fid, '    else\n');
fprintf(fid, '        args_to_pass = varargin;\n');
fprintf(fid, '    end\n\n');

fprintf(fid, '    F = arrayfun(@%s, args_to_pass{:});\n', func_name);
fprintf(fid, 'end\n');
fclose(fid);

rehash;
TensorFn = str2func(wrapperName);


end