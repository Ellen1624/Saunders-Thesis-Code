function [W] = XOR(n)

%run with: [W] = XOR(10)
% the input n is the gradient descent step-size


% Training Pairs
    % we could insist on these being inputs, but for the purposes of this
    % exercise we won't be varying the training pairs

x1 = [0; 0];
x2 = [0; 1];
x3 = [1; 0];
x4 = [1; 1];
y = [0 1 1 0];

f = @(v) 1./(1 + exp(-v));


% Naive Random Search

naive_iter = 10000;
n_iter = 1;
Ebest = 1000;

while n_iter < naive_iter
    
    i = -20 + 40*rand(9,1);

    % we will say i(1) = w11h, i(2) = w12h, i(3) = w21h, i(4) = w22h, 
    % i(5) = w11o, i(6) = w12o, i(7) = theta_1, i(8) = theta_2, i(9) = theta_3
    
    
    % x1
    
    v11 = i(1)*x1(1) + i(2)*x1(2) - i(7);
    v21 = i(3)*x1(1) + i(4)*x1(2) - i(8);
    z11 = f(v11);
    z21 = f(v21);
    y1 = f(i(5)*z11 + i(6)*z21 - i(9));
    % d11 = (y(1) - y1)*y1*(1-y1);
    
    % x2
    
    v12 = i(1)*x2(1) + i(2)*x2(2) - i(7);
    v22 = i(3)*x2(1) + i(4)*x2(2) - i(8);
    z12 = f(v12);
    z22 = f(v22);
    y2 = f(i(5)*z12 + i(6)*z22 - i(9));
    
    % x3
    
    v13 = i(1)*x3(1) + i(2)*x3(2) - i(7);
    v23 = i(3)*x3(1) + i(4)*x3(2) - i(8);
    z13 = f(v13);
    z23 = f(v23);
    y3 = f(i(5)*z13 + i(6)*z23 - i(9));
    
    % x4
    
    v14 = i(1)*x4(1) + i(2)*x4(2) - i(7);
    v24 = i(3)*x4(1) + i(4)*x4(2) - i(8);
    z14 = f(v14);
    z24 = f(v24);
    y4 = f(i(5)*z14 + i(6)*z24 - i(9));

    E = 1/8*((y1)^2 + (y2 - 1)^2 + (y3 - 1)^2 + (y4)^2);

    if E < Ebest
        Ebest = E;
        W = i;
    end

    n_iter = n_iter + 1

end


% Fixed Step-Size Gradient Descent Algorithm

descent_iter = 10000;
d_iter = 1

while d_iter < descent_iter

    W

    % W(1) = w11h, W(2) = w12h, W(3) = w21h, W(4) = w22h, W(5) = w11o, 
    % W(6) = w12o, W(7) = theta_1, W(8) = theta_2, W(9) = theta_3

    % x1
    
    v11 = W(1)*x1(1) + W(2)*x1(2) - W(7);
    v21 = W(3)*x1(1) + W(4)*x1(2) - W(8);
    z11 = f(v11);
    z21 = f(v21);
    y1 = f(W(5)*z11 + W(6)*z21 - W(9));
    d1 = (y(1) - y1)*y1*(1 - y1);
    
    % x2
    
    v12 = W(1)*x2(1) + W(2)*x2(2) - W(7);
    v22 = W(3)*x2(1) + W(4)*x2(2) - W(8);
    z12 = f(v12);
    z22 = f(v22);
    y2 = f(W(5)*z12 + W(6)*z22 - W(9));
    d2 = (y(2) - y2)*y2*(1 - y2);
    
    % x3
    
    v13 = W(1)*x3(1) + W(2)*x3(2) - W(7);
    v23 = W(3)*x3(1) + W(4)*x3(2) - W(8);
    z13 = f(v13);
    z23 = f(v23);
    y3 = f(W(5)*z13 + W(6)*z23 - W(9));
    d3 = (y(3) - y3)*y3*(1 - y3);
    
    % x4
    
    v14 = W(1)*x4(1) + W(2)*x4(2) - W(7);
    v24 = W(3)*x4(1) + W(4)*x4(2) - W(8);
    z14 = f(v14);
    z24 = f(v24);
    y4 = f(W(5)*z14 + W(6)*z24 - W(9));
    d4 = (y(4) - y4)*y4*(1 - y4);

    % Error

    E = 1/8*((y1)^2 + (y2 - 1)^2 + (y3 - 1)^2 + (y4)^2)

    % So this prints the E for the previous weights, but if you want the
    % error for the weights produced after 10,000 iterations, set the
    % iteration number to 10,001.  It seems highly computationally
    % inefficient to compute all the v's and z's and d's to find Wnew, then
    % recompute them for Wnew just to find the error when they will be
    % recomputed on the next iteration.

    % Gradient Components

    % dE1 = dE/dwo11, dE2 = dE/dwo12, dE3 = dE/dtheta3, dE4 = dE/dwh11,
    % dE5 = dE/dwh12, dE6 = dE/dwh21, dE7 = dE/dwh22,
    % dE8 = dE/dtheta1, dE9 = dE/dtheta2

    dE1 = -d1*z11 - d2*z12 - d3*z13 - d4*z14;
    dE2 = -d1*z21 - d2*z22 - d3*z23 - d4*z24;
    dE3 = d1 + d2 + d3 + d4;
    dE4 = -d1*W(5)*z11*(1-z11)*x1(1) - d2*W(5)*z12*(1-z12)*x2(1) - d3*W(5)*z13*(1-z13)*x3(1) - d4*W(5)*z14*(1-z14)*x4(1);
    dE5 = -d1*W(5)*z11*(1-z11)*x1(2) - d2*W(5)*z12*(1-z12)*x2(2) - d3*W(5)*z13*(1-z13)*x3(2) - d4*W(5)*z14*(1-z14)*x4(2);
    dE6 = -d1*W(6)*z21*(1-z21)*x1(1) - d2*W(6)*z22*(1-z22)*x2(1) - d3*W(6)*z23*(1-z23)*x3(1) - d4*W(6)*z24*(1-z24)*x4(1);
    dE7 = -d1*W(6)*z21*(1-z21)*x1(2) - d2*W(6)*z22*(1-z22)*x2(2) - d3*W(6)*z23*(1-z23)*x3(2) - d4*W(6)*z24*(1-z24)*x4(2);
    dE8 = d1*W(5)*z11*(1-z11) + d2*W(5)*z12*(1-z12) + d3*W(5)*z13*(1-z13) + d4*W(5)*z14*(1-z14);
    dE9 = d1*W(6)*z21*(1-z21) + d2*W(6)*z22*(1-z22) + d3*W(6)*z23*(1-z23) + d4*W(6)*z24*(1-z24);

    % New Weights

    % n = step size = 10.0 <- this is an input

    % Wnew(1) = w11h, Wnew(2) = w12h, Wnew(3) = w21h, Wnew(4) = w22h, Wnew(5) = w11o, 
    % Wnew(6) = w12o, Wnew(7) = theta_1, Wnew(8) = theta_2, Wnew(9) = theta_3

    Wnew(1) = W(1) - n*dE4;
    Wnew(2) = W(2) - n*dE5;
    Wnew(3) = W(3) - n*dE6;
    Wnew(4) = W(4) - n*dE7;
    Wnew(5) = W(5) - n*dE1;
    Wnew(6) = W(6) - n*dE2;
    Wnew(7) = W(7) - n*dE8;
    Wnew(8) = W(8) - n*dE9;
    Wnew(9) = W(9) - n*dE3;

    % these indexes may haunt my nightmares

    W = Wnew;
    
    d_iter = d_iter + 1

end


% Graph?  If you dare
% I do dare, it finally worked

wo = [W(5) W(6)];
wh = [W(1) W(2); W(3) W(4)];
theta = [W(7) W(8) W(9)];


resolution = 50;
f = @(v) 1./(1 + exp(-v));
    
    % Generate grid of points in the unit square [0, 1] x [0, 1]
    [X1, X2] = meshgrid(linspace(0, 1, resolution), linspace(0, 1, resolution));
        
    % Initialize the output grid
    Y = zeros(size(X1));
        
    % Compute the network's output for each point on the grid
    for i = 1:resolution
        for j = 1:resolution
            x1 = X1(i, j);
            x2 = X2(i, j);
            %v1 = W(1)*x1 + W(2)*x2
            %v2 = W(3)*x1 + W(4)*x2
            v1 = wh(1,1)*x1 + wh(1,2)*x2 - theta(1);
            v2 = wh(2,1)*x1 + wh(2,2)*x2 - theta(2);
            z1 = f(v1);
            z2 = f(v2);
            y = f(wo(1)*z1 + wo(2)*z2 - theta(3));
            %y = f(W(5)*z1 + W(6)*z2);
            Y(i, j) = y;
        end
    end
    
    % Plot the surface
    %{
    mesh(X1, X2, Y);
    colormap(gray);        % Use grayscale colormap
    %shading interp; 
    xlabel('x(1)');
    ylabel('x(2)');
    zlabel('y(x)');
    title('XOR Neural Network Output');
    %colorbar;
    %}

    mesh(X1, X2, Y, 'EdgeColor', 'k', 'FaceColor', 'none');
    xlabel('x(1)');
    ylabel('x(2)');
    zlabel('y(x)');
    title('XOR Neural Network Output');

end

