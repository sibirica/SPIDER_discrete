X = ncread('u0.000.nc','X');
Y = ncread('u0.000.nc','Y');
Z = ncread('u0.000.nc','Z');
nx = length(X); ny = length(Y); nz = length(Z);
nt = 100;
range = 0:1:nt-1;
dt = 0.01;
T = range*dt;
U = zeros(nx, ny, nz, nt);
V = U; W = U; P = U;
for i=range
    disp(i)
    rstr = sprintf('u0.%02d0.nc',i);
    pstr = sprintf('p0.%02d0.nc',i);
    U(:, :, :, i+1) = ncread(rstr,'Velocity_X');
    V(:, :, :, i+1) = ncread(rstr,'Velocity_Y');
    W(:, :, :, i+1) = ncread(rstr,'Velocity_Z');
    P(:, :, :, i+1) = ncread(pstr,'Component_0');
end
save('channelflow.mat', 'U', 'V', 'W', 'P', 'X', 'Y', 'Z', 'T')
