function spl = get_spline(a)
% creates a spline representation from an array of control points a
k = 4;
b = repmat(a,1,2);

n = size(a,2);
t = 1:(2*n+k);

spl0 = spmak(t, b);
spl = fnbrk(spl0, [n/2+2 3/2*n+2]);
