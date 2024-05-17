% DeepRitz for solving Poisson's eq. in 1D
N = 10;
h = 1/N;
f = @(x) sin(2*pi*x);
x = linspace(0,1,N+1);
u = fminsearch(@(u) energy(u, f), ones(1,N-1));
plot(x, [0,u,0],'MarkerSize',10, 'Marker','o', 'Color','b');
hold on;
x_pts = linspace(0,1,100);
plot(x_pts, f(x_pts)/(4*pi^2), 'LineWidth',2);
legend('Approximate','Exact');

function y = energy(u, f)
    N = length(u)+1;
    h = 1/N;
    xlist = linspace(0, 1-h, N);
    right = [u, 0];
    left = [0, u];
    y = sum((right - left).^2)/h/2 - h*sum(left.*f(xlist));
end