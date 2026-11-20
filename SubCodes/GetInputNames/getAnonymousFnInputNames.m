function fninputnames = getAnonymousFnInputNames(FnToEvaluate)
% GETANONYMOUSFNINPUTNAMES Supports both @(a, b) anonymous handles and @named_fn handles,
% trimming whitespace around parameter names (Octave func2str preserves spaces after commas).
str = strtrim(func2str(FnToEvaluate));
if ~isempty(str) && str(1) == '@'
    open_p  = find(str == '(', 1, 'first');
    close_p = find(str == ')', 1, 'first');
    if isempty(open_p) || isempty(close_p) || close_p <= open_p + 1
        fninputnames = {};
    else
        parts = strsplit(str(open_p+1 : close_p-1), ',');
        fninputnames = cellfun(@strtrim, parts, 'UniformOutput', false);
    end
    return;
end

% Named function handle (@fn -> 'fn'): parse `function ... = fn(arg1, arg2, ...)` header
fpath = which(str);
if isempty(fpath)
    error('getAnonymousFnInputNames: could not locate function file for @%s', str);
end
txt = fileread(fpath);
txt = regexprep(txt, '%[^\r\n]*', '');
txt = regexprep(txt, '\.\.\.\s*[\r\n]+', ' ');
tok = regexp(txt, ['function\s+(?:[^=\(\r\n]+=\s*)?', regexptranslate('escape', str), '\s*\(([^\)]*)\)'], 'tokens', 'once');
if isempty(tok) || isempty(strtrim(tok{1}))
    fninputnames = {};
else
    parts = strsplit(strtrim(tok{1}), ',');
    fninputnames = cellfun(@strtrim, parts, 'UniformOutput', false);
end
end
