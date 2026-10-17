function capture_complex_trig_matrix(outdir, referenceRoot)
% Capture numeric-matrix and callback construction for the complex periodic
% Chebfun2 constructor gap. Prepared only; do not run without parent approval.
% Reference: Chebfun 7574c77680d7e82b79626300bf255498271a72df.

if ~exist(outdir, 'dir'), mkdir(outdir); end
addpath(genpath(referenceRoot));
expectedClassBinding = fullfile(referenceRoot, '@chebfun2', 'chebfun2.m');
expectedConstructor = fullfile(referenceRoot, '@chebfun2', 'constructor.m');
resolvedClassBinding = which('chebfun2');
resolvedConstructor = which('chebfun2/constructor');
if ~strcmp(resolvedClassBinding, expectedClassBinding)
    error('CHEBFUN2:wrongClassBinding', 'Resolved class binding: %s', resolvedClassBinding);
end
if ~strcmp(resolvedConstructor, expectedConstructor)
    error('CHEBFUN2:wrongConstructor', 'Resolved constructor: %s', resolvedConstructor);
end
pref = chebfunpref();
tol = 1e2 * pref.cheb2Prefs.chebfun2eps; % tests/chebfun2/test_constructor.m
cases = [11 9; 12 9; 11 10; 12 10]; % x-count, y-count; A has ny-by-nx layout
dom = [2 7 -3 5];

for icase = 1:size(cases,1)
    nx = cases(icase,1); ny = cases(icase,2);
    xnodes = trigpts(nx, dom(1:2));
    ynodes = trigpts(ny, dom(3:4));
    [XX, YY] = meshgrid(xnodes, ynodes);
    A = field_value(XX, YY, dom, nx, ny);

    % This is the exact public numeric-data route implicated by the symptom.
    fMatrix = chebfun2(A, dom, 'trig');
    fCallback = chebfun2(@(x,y) field_value(x,y,dom,nx,ny), dom, 'trig');
    nativeSourceNorm = norm(fMatrix - fCallback);
    nativeSourcePass = nativeSourceNorm < tol;

    % Dense, non-grid, non-endpoint queries retain the matrix's orientation
    % and complex phase without adding quadrature or interpolation assumptions.
    xq = dom(1) + (dom(2)-dom(1))*[.013 .071 .19 .37 .58 .81 .943];
    yq = dom(3) + (dom(4)-dom(3))*[.027 .11 .29 .52 .73 .91 .977];
    [XQ, YQ] = meshgrid(xq, yq);
    nativeMatrixNodes = fMatrix(XX, YY);
    nativeMatrixDense = fMatrix(XQ, YQ);
    nativeCallbackDense = fCallback(XQ, YQ);
    analyticDense = field_value(XQ, YQ, dom, nx, ny);

    payload = struct();
    payload.source_pin = '7574c77680d7e82b79626300bf255498271a72df';
    payload.source_tol_expression = '100 * pref.cheb2Prefs.chebfun2eps';
    payload.source_tol = tol;
    payload.source_matrix_vs_callback_norm = nativeSourceNorm;
    payload.source_matrix_vs_callback_pass = nativeSourcePass;
    payload.domain = dom;
    payload.nx = nx; payload.ny = ny;
    payload.xnodes = xnodes; payload.ynodes = ynodes;
    payload.A = A;
    payload.native_matrix_nodes = nativeMatrixNodes;
    payload.xq = xq; payload.yq = yq;
    payload.native_matrix_dense = nativeMatrixDense;
    payload.native_callback_dense = nativeCallbackDense;
    payload.analytic_dense = analyticDense;
    payload.matrix_is_complex = ~isreal(A);
    payload.matrix_size = size(A);
    payload.matrix_class = class(A);
    payload.matlab_version = version;
    payload.reference_root = referenceRoot;
    payload.chebfun_class_binding_path = resolvedClassBinding;
    payload.chebfun2_constructor_path = resolvedConstructor;
    payload.trigpts_path = which('trigpts');

    filename = fullfile(outdir, sprintf('case_%02dx%02d.mat', nx, ny));
    save(filename, '-struct', 'payload', '-v7');
end
end

function z = field_value(x, y, dom, nx, ny)
xa = dom(1); xb = dom(2); ya = dom(3); yb = dom(4);
tx = 2*(x-xa)/(xb-xa)-1;
ty = 2*(y-ya)/(yb-ya)-1;
z = (0.7+1.2i) + (1.3-0.4i).*exp(1i*pi*(2*tx-ty)) ...
    + (-0.2+0.9i).*exp(-1i*pi*(tx+2*ty));
if mod(nx,2)==0
    z = z + (0.35+0.8i).*cos((nx/2)*pi*tx);
end
if mod(ny,2)==0
    z = z + (-0.45+0.3i).*cos((ny/2)*pi*ty);
end
end
