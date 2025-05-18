
% Neighbor PINN with correct gradients


% physics informed neural network complete with data loading and formatting

% load data

%{
load('HF_field1/ln_save.mat');
load('HF_field1/lt_save.mat');
load('HF_field1/u_save.mat');
load('HF_field1/v_save.mat');
load('HF_field1/u_matrix_20.mat');
load('HF_field1/v_matrix_20.mat');
%}


% ROMS_field1
load('ROMS_field1/ln_save.mat');
load('ROMS_field1/lt_save.mat');
load('ROMS_field1/u_save.mat');
load('ROMS_field1/v_save.mat');
load('ROMS_field1/u_matrix_15.mat');
load('ROMS_field1/v_matrix_15.mat');


%{
% ROMS_field2
load('ROMS_field2/ln_save.mat');
load('ROMS_field2/lt_save.mat');
load('ROMS_field2/u_save.mat');
load('ROMS_field2/v_save.mat');
load('ROMS_field2/u_matrix_15.mat');
load('ROMS_field2/v_matrix_15.mat');
%}

u_matrix = u_matrix_15;
v_matrix = v_matrix_15;

%%

[rowCount, colCount] = size(u_matrix);

figure;
quiver(ln_save, lt_save, u_save, v_save, 'AutoScale', 'on');
xlabel('Longitude');
ylabel('Latitude');
title('Original Velocity Field');
axis equal;
grid on;

figure;
%[X, Y] = meshgrid(1:colCount, 1:rowCount);
quiver(ln_save, lt_save, u_matrix, v_matrix, 'AutoScale', 'on');
xlabel('Longitude');
ylabel('Latitude');
title('Velocity Field with a Missing Patch');
axis equal;
grid on;

%%

u_min_val = min(u_matrix(:));
u_max_val = max(u_matrix(:));

% Avoid division by zero if all values are the same
if u_min_val == u_max_val
    warning('All elements in the matrix are the same. Returning a matrix of zeros.');
    u_matrix = zeros(size(u_matrix));
else
    %u_scaled_matrix = (u_matrix - u_min_val) / (u_max_val - u_min_val);
    u_matrix = (u_matrix - u_min_val) / (u_max_val - u_min_val);
end

v_min_val = min(v_matrix(:));
v_max_val = max(v_matrix(:));

% Avoid division by zero if all values are the same
if v_min_val == v_max_val
    warning('All elements in the matrix are the same. Returning a matrix of zeros.');
    v_matrix = zeros(size(v_matrix));
else
    %v_scaled_matrix = (v_matrix - v_min_val) / (v_max_val - v_min_val);
    v_matrix = (v_matrix - v_min_val) / (v_max_val - v_min_val);
end

%%

% Get matrix size
[m, n] = size(u_matrix);

% Initialize output variables as empty arrays
interior_u = [];
interior_v = [];
ux_training = [];
vx_training = [];

% Loop over the interior of the matrix (skip boundaries)
for i = 2:m-1
    for j = 2:n-1
        % Grab 3x3 block
        u_block = u_matrix(i-1:i+1, j-1:j+1);
        v_block = v_matrix(i-1:i+1, j-1:j+1);

        % Check for any NaNs in the full 3x3 blocks
        if all(~isnan(u_block(:))) && all(~isnan(v_block(:)))
            % Save center values
            interior_u = [interior_u; u_matrix(i, j)];
            interior_v = [interior_v; v_matrix(i, j)];

            % Save neighbors (excluding center), in specified order:
            % [UL, UM, UR, ML, MR, LL, LM, LR]
            ux_training = [ux_training; ...
                u_matrix(i-1,j-1), u_matrix(i-1,j), u_matrix(i-1,j+1), ...
                u_matrix(i,j-1),                   u_matrix(i,j+1), ...
                u_matrix(i+1,j-1), u_matrix(i+1,j), u_matrix(i+1,j+1)];

            vx_training = [vx_training; ...
                v_matrix(i-1,j-1), v_matrix(i-1,j), v_matrix(i-1,j+1), ...
                v_matrix(i,j-1),                   v_matrix(i,j+1), ...
                v_matrix(i+1,j-1), v_matrix(i+1,j), v_matrix(i+1,j+1)];
        end
    end
end

% Display results
disp('interior_u:');
disp(interior_u);
disp('ux:');
disp(ux_training);
disp(['Size of ux: ', mat2str(size(ux_training))]);

interior_point_count = length(interior_u);


%%


% physics informed neural network

f = @(v) 1./(1 + exp(-v));

n = 0.0005;

% Naive Random Search

naive_iter = 10000;
n_iter = 1;
Ebest = 1000000;
E_vector_n = zeros(1, naive_iter-1);

%initialize uv, uz, and uy
uv = zeros(4,interior_point_count);
uz = zeros(4, interior_point_count);
uy = zeros(interior_point_count);

vv = zeros(4,interior_point_count);
vz = zeros(4, interior_point_count);
vy = zeros(interior_point_count);

%E = 0;

while n_iter < naive_iter

    E = 0;

    % initial random weights
    % 10x9 random matrix (but the last 4 entries d(5,6)-d(5,9) are 0)
    h = rand(10,9);        % adjust according to patch size  
    h(5,6) = 0;
    h(5,7) = 0;
    h(5,8) = 0;
    h(5,9) = 0;
    h(10,6) = 0;
    h(10,7) = 0;
    h(10,8) = 0;
    h(10,9) = 0;
    %W = h;
    
    num_weights = 82;

    for j = 1:interior_point_count

        ux = ux_training(j,:);
        vx = vx_training(j,:);

        for i = 1:4
            uv(i,j) = h(i,1)*ux(1) + h(i,2)*ux(2) + h(i,3)*ux(3) + h(i,4)*ux(4) + h(i,5)*ux(5) + h(i,6)*ux(6) + h(i,7)*ux(7) + h(i,8)*ux(8) - h(i,9);
            uz(i,j) = f(uv(i,j));

            vv(i,j) = h(i+5,1)*vx(1) + h(i+5,2)*vx(2) + h(i+5,3)*vx(3) + h(i+5,4)*vx(4) + h(i+5,5)*vx(5) + h(i+5,6)*vx(6) + h(i+5,7)*vx(7) + h(i+5,8)*vx(8) - h(i+5,9);
            vz(i,j) = f(vv(i,j));
        end

        uy(j) = f( h(5,1)*uz(1,j) + h(5,2)*uz(2,j)  + h(5,3)*uz(3,j) + h(5,4)*uz(4,j) - h(5,5) );
        vy(j) = f( h(10,1)*vz(1,j) + h(10,2)*vz(2,j) + h(10,3)*vz(3,j) + h(10,4)*vz(4,j) - h(10,5) );

        ud(j) = (uy(j) - interior_u(j))*uy(j)*(1 - uy(j));
        vd(j) = (vy(j) - interior_v(j))*vy(j)*(1 - vy(j));
        
        E = E + (uy(j) - interior_u(j))^2 + (vy(j) - interior_v(j))^2;
        
    end

    % Initialize output matrix
    u_original = u_matrix;
    v_original = v_matrix;
    [rowCount, colCount] = size(u_matrix);
    u_new = u_matrix;
    v_new = v_matrix;
        
    spiral_count = 0;        
    max_spirals = 100;  % Set a sensible max to avoid infinite loops
        
    while any(isnan(u_original(:))) || any(isnan(v_original(:)))
        if spiral_count >= max_spirals
            warning('Maximum spiral iterations reached.');
            break;
        end
        
        % Neighbor directions (row_offset, col_offset)
        neighbor_offsets = [
            -1, -1;
            -1,  0;
            -1,  1;
             0, -1;
             0,  1;
             1, -1;
             1,  0;
             1,  1
        ];
        
        for i = 1:rowCount
            for j = 1:colCount
                if isnan(u_original(i,j)) || isnan(v_original(i,j))

                    % -------- Extract neighbors for u --------
                    u_neighbors = zeros(1, 8);
                    count_u = 0;
                    for k = 1:8
                        r = i + neighbor_offsets(k, 1);
                        c = j + neighbor_offsets(k, 2);
                        if r >= 1 && r <= rowCount && c >= 1 && c <= colCount
                            val = u_original(r, c);
                            if isnan(val), val = 0; end
                            u_neighbors(k) = val;
                            if val ~= 0, count_u = count_u + 1; end
                        end
                    end
        
                    % -------- Extract neighbors for v --------
                    v_neighbors = zeros(1, 8);
                    count_v = 0;
                    for k = 1:8
                        r = i + neighbor_offsets(k, 1);
                        c = j + neighbor_offsets(k, 2);
                        if r >= 1 && r <= rowCount && c >= 1 && c <= colCount
                            val = v_original(r, c);
                            if isnan(val), val = 0; end
                            v_neighbors(k) = val;
                            if val ~= 0, count_v = count_v + 1; end
                        end
                    end
        
                    if count_u > 1 && count_v > 1
                        %{
                        % U imputation
                        %im_uv = (8/count_u) * (W(1:4,1:8) * u_neighbors') - W(1:4,9);
                        im_uv = (8/count_u) * (h(1:4,1:8) * u_neighbors') - h(1:4,9);
                        im_uz = f(im_uv);
                        imputed_uy = f(h(5,1:4) * im_uz - h(5,5));
       
                        % V imputation
                        im_vv = (8/count_v) * (h(6:9,1:8) * v_neighbors') - h(6:9,9);
                        im_vz = f(im_vv);
                        imputed_vy = f(h(10,1:4) * im_vz - h(10,5));
        
                        u_new(i,j) = imputed_uy;
                        v_new(i,j) = imputed_vy;
                        %}

                        im_uv = zeros(4);
                        im_uz = zeros(4);
                        im_vv = zeros(4);
                        im_vz = zeros(4);

                        for b = 1:4
                            im_uv(b) = (8/count_u) * (h(b,1)*u_neighbors(1) + h(b,2)*u_neighbors(2) + h(b,3)*u_neighbors(3) + h(b,4)*u_neighbors(4) + h(b,5)*u_neighbors(5) + h(b,6)*u_neighbors(6) + h(b,7)*u_neighbors(7) + h(b,8)*u_neighbors(8) - h(b,9));
                            im_uz(b) = f(im_uv(b));
                
                            im_vv(b) = (8/count_u) * (h(b+5,1)*v_neighbors(1) + h(b+5,2)*v_neighbors(2) + h(b+5,3)*v_neighbors(3) + h(b+5,4)*v_neighbors(4) + h(b+5,5)*v_neighbors(5) + h(b+5,6)*v_neighbors(6) + h(b+5,7)*v_neighbors(7) + h(b+5,8)*v_neighbors(8) - h(b+5,9));
                            im_vz(b) = f(im_vv(b));
                        end
                
                        imputed_uy = f( h(5,1)*im_uz(1) + h(5,2)*im_uz(2)  + h(5,3)*im_uz(3) + h(5,4)*im_uz(4) - h(5,5) );
                        imputed_vy = f( h(10,1)*im_vz(1) + h(10,2)*im_vz(2) + h(10,3)*im_vz(3) + h(10,4)*im_vz(4) - h(10,5) );

                        u_new(i,j) = imputed_uy;
                        v_new(i,j) = imputed_vy;

                    end
                end
            end
        end
        
        % Update for next spiral
        u_original = u_new;
        v_original = v_new;
        spiral_count = spiral_count + 1;
        %fprintf('Completed spiral %d\n', spiral_count)
    end
    
    field_size = size(u_matrix);
    %h_step = abs(lon_max - lon_min)/(field_size(2) - 1);
    %k_step = abs(lat_max - lat_min)/(field_size(1) - 1);

    h_step = 6;
    k_step = 6;

    for i = 2:(field_size(1) - 1)
        for j = 2:(field_size(2) - 1)
            % divergence = 0; physics informed error component
            %E = E + abs((u_new(i,j+1) - u_new(i,j))/h_step + (v_new(i+1,j) - v_new(i,j))/k_step);
            E = E + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;
        end
    end

    % Convert cell array to matrix
    % u_missing_neighbors = cell2mat(neighbor_rows');

    if E < Ebest
        Ebest = E;
        W = h;
    end
    
    E_vector_n(n_iter) = Ebest;
    n_iter = n_iter + 1

end

% plotting error
figure;
plot(1:length(E_vector_n), E_vector_n)
xlabel('Iteration')
ylabel('Error')
title('Naive Error')
grid on

% Fixed Step-Size Gradient Descent Algorithm

descent_iter = 10000;
%descent_iter = 10;
d_iter = 1;

E_vector_g = zeros(1, descent_iter-1);

ud = zeros(interior_point_count);
vd = zeros(interior_point_count);
%dE = zeros(1:num_weights);

while d_iter < descent_iter
    E = 0;
    for j = 1:interior_point_count

        ux = ux_training(j,:);
        vx = vx_training(j,:);

        for i = 1:4
            uv(i,j) = W(i,1)*ux(1) + W(i,2)*ux(2) + W(i,3)*ux(3) + W(i,4)*ux(4) + W(i,5)*ux(5) + W(i,6)*ux(6) + W(i,7)*ux(7) + W(i,8)*ux(8) - W(i,9);
            uz(i,j) = f(uv(i,j));

            vv(i,j) = W(i+5,1)*vx(1) + W(i+5,2)*vx(2) + W(i+5,3)*vx(3) + W(i+5,4)*vx(4) + W(i+5,5)*vx(5) + W(i+5,6)*vx(6) + W(i+5,7)*vx(7) + W(i+5,8)*vx(8) - W(i+5,9);
            vz(i,j) = f(vv(i,j));
        end

        uy(j) = f( W(5,1)*uz(1,j) + W(5,2)*uz(2,j)  + W(5,3)*uz(3,j) + W(5,4)*uz(4,j) - W(5,5) );
        vy(j) = f( W(10,1)*vz(1,j) + W(10,2)*vz(2,j) + W(10,3)*vz(3,j) + W(10,4)*vz(4,j) - W(10,5) );

        ud(j) = (uy(j) - interior_u(j))*uy(j)*(1 - uy(j));
        vd(j) = (vy(j) - interior_v(j))*vy(j)*(1 - vy(j));
        
        E = E + (uy(j) - interior_u(j))^2 + (vy(j) - interior_v(j))^2;

    end

    % Initialize output matrix
    u_original = u_matrix;
    v_original = v_matrix;
    [rowCount, colCount] = size(u_matrix);
    u_new = u_matrix;
    v_new = v_matrix;

    u_count = zeros(field_size(1), field_size(2));
    v_count = zeros(field_size(1), field_size(2));
    ux_grad = zeros(field_size(1), field_size(2),8);
    vx_grad = zeros(field_size(1), field_size(2),8);
    
    spiral_count = 0;
    max_spirals = 100;  % Set a sensible max to avoid infinite loops
    
    while any(isnan(u_original(:))) || any(isnan(v_original(:)))
        if spiral_count >= max_spirals
            warning('Maximum spiral iterations reached.');
            break;
        end
    
        % Neighbor directions (row_offset, col_offset)
        neighbor_offsets = [
            -1, -1;
            -1,  0;
            -1,  1;
             0, -1;
             0,  1;
             1, -1;
             1,  0;
             1,  1
        ];
    
        for i = 1:rowCount
            for j = 1:colCount
                if isnan(u_original(i,j)) || isnan(v_original(i,j))
    
                    % -------- Extract neighbors for u --------
                    u_neighbors = zeros(1, 8);
                    count_u = 0;
                    for k = 1:8
                        r = i + neighbor_offsets(k, 1);
                        c = j + neighbor_offsets(k, 2);
                        if r >= 1 && r <= rowCount && c >= 1 && c <= colCount
                            val = u_original(r, c);
                            if isnan(val), val = 0; end
                            u_neighbors(k) = val;
                            ux_grad(i,j,k) = val;
                            if val ~= 0, count_u = count_u + 1; end
                        end
                    end
                    u_count(i,j) = count_u;
    
                    % -------- Extract neighbors for v --------
                    v_neighbors = zeros(1, 8);
                    count_v = 0;
                    for k = 1:8
                        r = i + neighbor_offsets(k, 1);
                        c = j + neighbor_offsets(k, 2);
                        if r >= 1 && r <= rowCount && c >= 1 && c <= colCount
                            val = v_original(r, c);
                            if isnan(val), val = 0; end
                            v_neighbors(k) = val;
                            vx_grad(i,j,k) = val;
                            if val ~= 0, count_v = count_v + 1; end
                        end
                    end

                    v_count(i,j) = count_v;
    
                    if count_u > 1 && count_v > 1
                        %{
                        % U imputation
                        im_uv = (8/count_u) * (W(1:4,1:8) * u_neighbors') - W(1:4,9);
                        im_uz = f(im_uv);
                        imputed_uy = f(W(5,1:4) * im_uz - W(5,5));
    
                        % V imputation
                        im_vv = (8/count_v) * (W(6:9,1:8) * v_neighbors') - W(6:9,9);
                        im_vz = f(im_vv);
                        imputed_vy = f(W(10,1:4) * im_vz - W(10,5));
    
                        u_new(i,j) = imputed_uy;
                        v_new(i,j) = imputed_vy;
                        %}
                        
                        im_uv = zeros(1,4);
                        im_uz = zeros(1,4);
                        im_vv = zeros(1,4);
                        im_vz = zeros(1,4);

                        heart_u = zeros(field_size(1), field_size(2));
                        heart_v = zeros(field_size(1), field_size(2));

                        %star_u = zeros(field_size(1), field_size(2),b);
                        %star_v = zeros(field_size(1), field_size(2),b);
                        star_u = zeros(field_size(1), field_size(2),4);
                        star_v = zeros(field_size(1), field_size(2),4);

                        for b = 1:4
                            im_uv(b) = (8/count_u) * (W(b,1)*u_neighbors(1) + W(b,2)*u_neighbors(2) + W(b,3)*u_neighbors(3) + W(b,4)*u_neighbors(4) + W(b,5)*u_neighbors(5) + W(b,6)*u_neighbors(6) + W(b,7)*u_neighbors(7) + W(b,8)*u_neighbors(8)) - W(b,9);
                            im_uz(b) = f(im_uv(b));
                
                            im_vv(b) = (8/count_v) * (W(b+5,1)*v_neighbors(1) + W(b+5,2)*v_neighbors(2) + W(b+5,3)*v_neighbors(3) + W(b+5,4)*v_neighbors(4) + W(b+5,5)*v_neighbors(5) + W(b+5,6)*v_neighbors(6) + W(b+5,7)*v_neighbors(7) + W(b+5,8)*v_neighbors(8)) - W(b+5,9);
                            im_vz(b) = f(im_vv(b));

                            %heart_u(i,j) = heart_u(i,j) + im_uv(b);
                            %heart_v(i,j) = heart_v(i,j) + im_vv(b);

                            star_u(i,j,b) = im_uv(b);
                            star_v(i,j,b) = im_vv(b);

                            uz_grad(i,j,b) = im_uz(b);
                            vz_grad(i,j,b) = im_vz(b);
                        end
                
                        imputed_uy = f( W(5,1)*im_uz(1) + W(5,2)*im_uz(2)  + W(5,3)*im_uz(3) + W(5,4)*im_uz(4) - W(5,5) );
                        imputed_vy = f( W(10,1)*im_vz(1) + W(10,2)*im_vz(2) + W(10,3)*im_vz(3) + W(10,4)*im_vz(4) - W(10,5) );

                        heart_u(i,j) = W(5,1)*im_uz(1) + W(5,2)*im_uz(2)  + W(5,3)*im_uz(3) + W(5,4)*im_uz(4) - W(5,5);
                        heart_v(i,j) = W(10,1)*im_vz(1) + W(10,2)*im_vz(2) + W(10,3)*im_vz(3) + W(10,4)*im_vz(4) - W(10,5);

                        u_new(i,j) = imputed_uy;
                        v_new(i,j) = imputed_vy;
                    end
                end
            end
        end
    
        % Update for next spiral
        u_original = u_new;
        v_original = v_new;
        spiral_count = spiral_count + 1;
        %fprintf('Completed spiral %d\n', spiral_count);
    end

    u_prime_outer = zeros(field_size(1), field_size(2),4);
    v_prime_outer = zeros(field_size(1), field_size(2),4);
    u_prime_hidden = zeros(field_size(1), field_size(2),4,8);
    v_prime_hidden = zeros(field_size(1), field_size(2),4,8);
    u_prime_outer_bias = zeros(field_size(1), field_size(2));
    v_prime_outer_bias = zeros(field_size(1), field_size(2));
    u_prime_hidden_bias = zeros(field_size(1), field_size(2),4);
    v_prime_hidden_bias = zeros(field_size(1), field_size(2),4);

    
    for i = 2:(field_size(1) - 1)
        for j = 2:(field_size(2) - 1)
            % divergence = 0; physics informed error component
            E = E + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;

            % partial derivative components
            if isnan(u_matrix(i,j))
                for a = 1:4
                    u_prime_outer(i,j,a) = f(heart_u(i,j))*(1-f(heart_u(i,j)))*uz_grad(i,j,a);
                    v_prime_outer(i,j,a) = f(heart_v(i,j))*(1-f(heart_v(i,j)))*vz_grad(i,j,a);

                    u_prime_outer_bias(i,j) = -f(heart_u(i,j))*(1-f(heart_u(i,j)));
                    v_prime_outer_bias(i,j) = -f(heart_v(i,j))*(1-f(heart_v(i,j)));
                    for b = 1:8
                        u_prime_hidden(i,j,a,b) = f(heart_u(i,j))*(1-f(heart_u(i,j)))*(W(5,1)*f(star_u(i,j,1))*(1-f(star_u(i,j,1))) + W(5,2)*f(star_u(i,j,2))*(1-f(star_u(i,j,2))) + W(5,3)*f(star_u(i,j,3))*(1-f(star_u(i,j,3))) + W(5,4)*f(star_u(i,j,4))*(1-f(star_u(i,j,4))))*(8/u_count(i,j))*ux_grad(i,j,b);
                        v_prime_hidden(i,j,a,b) = f(heart_v(i,j))*(1-f(heart_v(i,j)))*(W(10,1)*f(star_v(i,j,1))*(1-f(star_v(i,j,1))) + W(10,2)*f(star_v(i,j,2))*(1-f(star_v(i,j,2))) + W(10,3)*f(star_v(i,j,3))*(1-f(star_v(i,j,3))) + W(10,4)*f(star_v(i,j,4))*(1-f(star_v(i,j,4))))*(8/v_count(i,j))*vx_grad(i,j,b);

                        u_prime_hidden_bias(i,j,a) = -f(heart_u(i,j))*(1-f(heart_u(i,j)))*W(5,a)*f(star_u(i,j,a))*(1-f(star_u(i,j,a)));
                        v_prime_hidden_bias(i,j,a) = -f(heart_v(i,j))*(1-f(heart_v(i,j)))*W(10,a)*f(star_v(i,j,a))*(1-f(star_v(i,j,a)));
                    end
                end
            end


        end
    end
    
    udE1_outer_div = zeros(1,4);
    udE2_outer_div = zeros(1,4);
    udE3_outer_div = zeros(1,4);
    udE1_hidden_div = zeros(4,8);
    udE2_hidden_div = zeros(4,8);
    udE3_hidden_div = zeros(4,8);

    vdE1_outer_div = zeros(1,4);
    vdE2_outer_div = zeros(1,4);
    vdE3_outer_div = zeros(1,4);
    vdE1_hidden_div = zeros(4,8);
    vdE2_hidden_div = zeros(4,8);
    vdE3_hidden_div = zeros(4,8);

    udE1_outer_bias_div = 0;
    udE2_outer_bias_div = 0;
    udE3_outer_bias_div = 0;
    udE1_hidden_bias_div = zeros(1,4);
    udE2_hidden_bias_div = zeros(1,4);
    udE3_hidden_bias_div = zeros(1,4);

    vdE1_outer_bias_div = 0;
    vdE2_outer_bias_div = 0;
    vdE3_outer_bias_div = 0;
    vdE1_hidden_bias_div = zeros(1,4);
    vdE2_hidden_bias_div = zeros(1,4);
    vdE3_hidden_bias_div = zeros(1,4);

    uE_div_outer = zeros(1,4);
    uE_div_outer_bias = 0;
    uE_div_hidden = zeros(4,8);
    uE_div_hidden_bias = zeros(1,4);

    vE_div_outer = zeros(1,4);
    vE_div_outer_bias = 0;
    vE_div_hidden = zeros(4,8);
    vE_div_hidden_bias = zeros(1,4);
    
    for a = 1:4
        for b = 1:8
            for i = 2:(field_size(1)-1)
                for j = 2:(field_size(2)-1)
                    if isnan(u_matrix(i,j))
                        if isnan(u_matrix(i,j-1)) && isnan(u_matrix(i,j+1))
                            % outer weights u
                            udE1_outer_div(a) = 2*u_new(i,j+1)*u_prime_outer(i,j+1,a) + 2*u_new(i,j-1)*u_prime_outer(i,j-1,a) - 2*(u_new(i,j+1)*u_prime_outer(i,j-1,a) + u_prime_outer(i,j+1,a)*u_new(i,j-1));
                            udE2_outer_div(a) = 0;
                            udE3_outer_div(a) = u_prime_outer(i,j+1,a)*v_new(i+1,j) - u_prime_outer(i,j+1,a)*v_new(i-1,j) - u_prime_outer(i,j-1,a)*v_new(i+1,j) + u_prime_outer(i,j-1,a)*v_new(i-1,j);
                            % outer thetas u
                            udE1_outer_bias_div = 2*u_new(i,j+1)*u_prime_outer_bias(i,j+1) + 2*u_new(i,j-1)*u_prime_outer_bias(i,j-1) - 2*(u_new(i,j+1)*u_prime_outer_bias(i,j-1) + u_prime_outer_bias(i,j+1)*u_new(i,j-1));
                            udE2_outer_bias_div = 0;
                            udE3_outer_bias_div = u_prime_outer_bias(i,j+1)*v_new(i+1,j) - u_prime_outer_bias(i,j+1)*v_new(i-1,j) - u_prime_outer_bias(i,j-1)*v_new(i+1,j) + u_prime_outer_bias(i,j-1)*v_new(i-1,j);
                            % hidden weights u
                            udE1_hidden_div(a,b) = 2*u_new(i,j+1)*u_prime_hidden(i,j+1,a,b) + 2*u_new(i,j-1)*u_prime_hidden(i,j-1,a,b) - 2*(u_new(i,j+1)*u_prime_hidden(i,j-1,a,b) + u_new(i,j-1)*u_prime_hidden(i,j+1,a,b));
                            udE2_hidden_div(a,b) = 0;
                            udE3_hidden_div(a,b) = u_prime_hidden(i,j+1,a,b)*v_new(i+1,j) - u_prime_hidden(i,j+1,a,b)*v_new(i-1,j) - u_prime_hidden(i,j-1,a,b)*v_new(i+1,j) + u_prime_hidden(i,j-1,a,b)*v_new(i-1,j);
                            % hidden thetas u
                            udE1_hidden_bias_div(a) = 2*u_new(i,j+1)*u_prime_hidden_bias(i,j+1,a) + 2*u_new(i,j-1)*u_prime_hidden_bias(i,j-1,a) - 2*(u_new(i,j+1)*u_prime_hidden_bias(i,j-1,a) + u_new(i,j-1)*u_prime_hidden_bias(i,j+1,a));
                            udE2_hidden_bias_div(a) = 0;
                            udE3_hidden_bias_div(a) = u_prime_hidden_bias(i,j+1,a)*v_new(i+1,j) - u_prime_hidden_bias(i,j+1,a)*v_new(i-1,j) - u_prime_hidden_bias(i,j-1,a)*v_new(i+1,j) + u_prime_hidden_bias(i,j-1,a)*v_new(i-1,j);
                        end
                        if isnan(v_matrix(i-1,j)) && isnan(v_matrix(i+1,j))
                            % outer weights v
                            vdE1_outer_div(a) = 0;
                            vdE2_outer_div(a) = 2*v_new(i+1,j)*v_prime_outer(i+1,j,a) + 2*v_new(i-1,j)*v_prime_outer(i-1,j,a) - 2*(v_new(i+1,j)*v_prime_outer(i-1,j,a) + v_new(i-1,j)*v_prime_outer(i+1,j,a));
                            vdE3_outer_div(a) = u_new(i,j+1)*v_prime_outer(i+1,j,a) - u_new(i,j+1)*v_prime_outer(i-1,j,a) - u_new(i,j-1)*v_prime_outer(i+1,j,a) + u_new(i,j-1)*v_prime_outer(i-1,j,a);
                            % outer thetas v
                            vdE1_outer_bias_div = 0;
                            vdE2_outer_bias_div = 2*v_new(i+1,j)*v_prime_outer_bias(i+1,j) + 2*v_new(i-1,j)*v_prime_outer_bias(i-1,j) - 2*(v_new(i+1,j)*v_prime_outer_bias(i-1,j) + v_new(i-1,j)*v_prime_outer_bias(i+1,j));
                            vdE3_outer_bias_div = u_new(i,j+1)*v_prime_outer_bias(i+1,j) - u_new(i,j+1)*v_prime_outer_bias(i-1,j) - u_new(i,j-1)*v_prime_outer_bias(i+1,j) + u_new(i,j-1)*v_prime_outer_bias(i-1,j);
                            % hidden weights v
                            vdE1_hidden_div(a,b) = 0;
                            vdE2_hidden_div(a,b) = 2*v_new(i+1,j)*v_prime_hidden(i+1,j,a,b) + 2*v_new(i-1,j)*v_prime_hidden(i-1,j,a,b) - 2*(v_new(i+1,j)*v_prime_hidden(i-1,j,a,b) + v_new(i-1,j)*v_prime_hidden(i+1,j,a,b));
                            vdE3_hidden_div(a,b) = u_new(i,j+1)*v_prime_hidden(i+1,j,a,b) - u_new(i,j+1)*v_prime_hidden(i-1,j,a,b) - u_new(i,j-1)*v_prime_hidden(i+1,j,a,b) + u_new(i,j-1)*v_prime_hidden(i-1,j,a,b);
                            % hidden thetas v
                            vdE1_hidden_bias_div(a) = 0;
                            vdE2_hidden_bias_div(a) = 2*v_new(i+1,j)*v_prime_hidden_bias(i+1,j,a) + 2*v_new(i-1,j)*v_prime_hidden_bias(i-1,j,a) - 2*(v_new(i+1,j)*v_prime_hidden_bias(i-1,j,a) + v_new(i-1,j)*v_prime_hidden_bias(i+1,j,a));
                            vdE3_hidden_bias_div(a) = u_new(i,j+1)*v_prime_hidden_bias(i+1,j,a) - u_new(i,j+1)*v_prime_hidden_bias(i-1,j,a) - u_new(i,j-1)*v_prime_hidden_bias(i+1,j,a) + u_new(i,j-1)*v_prime_hidden_bias(i-1,j,a);
                        end

                        if isnan(u_matrix(i,j-1)) && ~isnan(u_matrix(i,j+1))
                            % outer weights u
                            udE1_outer_div(a) = 2*u_new(i,j-1)*u_prime_outer(i,j-1,a) - 2*u_new(i,j+1)*u_prime_outer(i,j-1,a);
                            udE2_outer_div(a) = 0;
                            udE3_outer_div(a) = - u_prime_outer(i,j-1,a)*v_new(i+1,j) + u_prime_outer(i,j-1,a)*v_new(i-1,j);
                            % outer thetas u
                            udE1_outer_bias_div = 2*u_new(i,j-1)*u_prime_outer_bias(i,j-1) - 2*u_new(i,j+1)*u_prime_outer_bias(i,j-1);
                            udE2_outer_bias_div = 0;
                            udE3_outer_bias_div = - u_prime_outer_bias(i,j-1)*v_new(i+1,j) + u_prime_outer_bias(i,j-1)*v_new(i-1,j);
                            % hidden weights u
                            udE1_hidden_div(a,b) = 2*u_new(i,j-1)*u_prime_hidden(i,j-1,a,b) - 2*u_new(i,j+1)*u_prime_hidden(i,j-1,a,b);
                            udE2_hidden_div(a,b) = 0;
                            udE3_hidden_div(a,b) = - u_prime_hidden(i,j-1,a,b)*v_new(i+1,j) + u_prime_hidden(i,j-1,a,b)*v_new(i-1,j);
                            % hidden thetas u
                            udE1_hidden_bias_div(a) = 2*u_new(i,j-1)*u_prime_hidden_bias(i,j-1,a) - 2*u_new(i,j+1)*u_prime_hidden_bias(i,j-1,a);
                            udE2_hidden_bias_div(a) = 0;
                            udE3_hidden_bias_div(a) = - u_prime_hidden_bias(i,j-1,a)*v_new(i+1,j) + u_prime_hidden_bias(i,j-1,a)*v_new(i-1,j);
                        end
                        if isnan(v_matrix(i-1,j)) && ~isnan(v_matrix(i+1,j))
                            % outer weights v
                            vdE1_outer_div(a) = 0;
                            vdE2_outer_div(a) = 2*v_new(i-1,j)*v_prime_outer(i-1,j,a) - 2*v_new(i+1,j)*v_prime_outer(i-1,j,a);
                            vdE3_outer_div(a) = - u_new(i,j+1)*v_prime_outer(i-1,j,a) + u_new(i,j-1)*v_prime_outer(i-1,j,a);
                            % outer thetas v
                            vdE1_outer_bias_div = 0;
                            vdE2_outer_bias_div = 2*v_new(i-1,j)*v_prime_outer_bias(i-1,j) - 2*v_new(i+1,j)*v_prime_outer_bias(i-1,j);
                            vdE3_outer_bias_div = - u_new(i,j+1)*v_prime_outer_bias(i-1,j) + u_new(i,j-1)*v_prime_outer_bias(i-1,j);
                            % hidden weights v
                            vdE1_hidden_div(a,b) = 0;
                            vdE2_hidden_div(a,b) = 2*v_new(i-1,j)*v_prime_hidden(i-1,j,a,b) - 2*v_new(i+1,j)*v_prime_hidden(i-1,j,a,b);
                            vdE3_hidden_div(a,b) = - u_new(i,j+1)*v_prime_hidden(i-1,j,a,b) + u_new(i,j-1)*v_prime_hidden(i-1,j,a,b);
                            % hidden thetas v
                            vdE1_hidden_bias_div(a) = 0;
                            vdE2_hidden_bias_div(a) = 2*v_new(i-1,j)*v_prime_hidden_bias(i-1,j,a) - 2*v_new(i+1,j)*v_prime_hidden_bias(i-1,j,a);
                            vdE3_hidden_bias_div(a) = - u_new(i,j+1)*v_prime_hidden_bias(i-1,j,a) + u_new(i,j-1)*v_prime_hidden_bias(i-1,j,a);                    
                        end

                        if ~isnan(u_matrix(i,j-1)) && isnan(u_matrix(i,j+1))
                            % outer weights u
                            udE1_outer_div(a) = 2*u_new(i,j+1)*u_prime_outer(i,j+1,a) - 2*u_prime_outer(i,j+1,a)*u_new(i,j-1);
                            udE2_outer_div(a) = 0;
                            udE3_outer_div(a) = u_prime_outer(i,j+1,a)*v_new(i+1,j) - u_prime_outer(i,j+1,a)*v_new(i-1,j);
                            % outer thetas u
                            udE1_outer_bias_div = 2*u_new(i,j+1)*u_prime_outer_bias(i,j+1) - 2* + u_prime_outer_bias(i,j+1)*u_new(i,j-1);
                            udE2_outer_bias_div = 0;
                            udE3_outer_bias_div = u_prime_outer_bias(i,j+1)*v_new(i+1,j) - u_prime_outer_bias(i,j+1)*v_new(i-1,j);
                            % hidden weights u
                            udE1_hidden_div(a,b) = 2*u_new(i,j+1)*u_prime_hidden(i,j+1,a,b) - 2*u_new(i,j-1)*u_prime_hidden(i,j+1,a,b);
                            udE2_hidden_div(a,b) = 0;
                            udE3_hidden_div(a,b) = u_prime_hidden(i,j+1,a,b)*v_new(i+1,j) - u_prime_hidden(i,j+1,a,b)*v_new(i-1,j);
                            % hidden thetas u
                            udE1_hidden_bias_div(a) = 2*u_new(i,j+1)*u_prime_hidden_bias(i,j+1,a) - 2*u_new(i,j-1)*u_prime_hidden_bias(i,j+1,a);
                            udE2_hidden_bias_div(a) = 0;
                            udE3_hidden_bias_div(a) = u_prime_hidden_bias(i,j+1,a)*v_new(i+1,j) - u_prime_hidden_bias(i,j+1,a)*v_new(i-1,j);
                        end
                        if ~isnan(v_matrix(i-1,j)) && isnan(v_matrix(i+1,j))
                            % outer weights v
                            vdE1_outer_div(a) = 0;
                            vdE2_outer_div(a) = 2*v_new(i+1,j)*v_prime_outer(i+1,j,a) - 2*v_new(i-1,j)*v_prime_outer(i+1,j,a);
                            vdE3_outer_div(a) = u_new(i,j+1)*v_prime_outer(i+1,j,a)- u_new(i,j-1)*v_prime_outer(i+1,j,a);
                            % outer thetas v
                            vdE1_outer_bias_div = 0;
                            vdE2_outer_bias_div = 2*v_new(i+1,j)*v_prime_outer_bias(i+1,j) - 2*v_new(i-1,j)*v_prime_outer_bias(i+1,j);
                            vdE3_outer_bias_div = u_new(i,j+1)*v_prime_outer_bias(i+1,j) - u_new(i,j-1)*v_prime_outer_bias(i+1,j);
                            % hidden weights v
                            vdE1_hidden_div(a,b) = 0;
                            vdE2_hidden_div(a,b) = 2*v_new(i+1,j)*v_prime_hidden(i+1,j,a,b)- 2*v_new(i-1,j)*v_prime_hidden(i+1,j,a,b);
                            vdE3_hidden_div(a,b) = u_new(i,j+1)*v_prime_hidden(i+1,j,a,b) - u_new(i,j-1)*v_prime_hidden(i+1,j,a,b);
                            % hidden thetas v
                            vdE1_hidden_bias_div(a) = 0;
                            vdE2_hidden_bias_div(a) = 2*v_new(i+1,j)*v_prime_hidden_bias(i+1,j,a) - 2*v_new(i-1,j)*v_prime_hidden_bias(i+1,j,a);
                            vdE3_hidden_bias_div(a) = u_new(i,j+1)*v_prime_hidden_bias(i+1,j,a) - u_new(i,j-1)*v_prime_hidden_bias(i+1,j,a);
                        end

                        if ~isnan(u_matrix(i,j-1)) && ~isnan(u_matrix(i,j+1))
                            % outer weights u
                            udE1_outer_div(a) = 0;
                            udE2_outer_div(a) = 0;
                            udE3_outer_div(a) = 0;
                            % outer thetas u
                            udE1_outer_bias_div = 0;
                            udE2_outer_bias_div = 0;
                            udE3_outer_bias_div = 0;
                            % hidden weights u
                            udE1_hidden_div(a,b) = 0;
                            udE2_hidden_div(a,b) = 0;
                            udE3_hidden_div(a,b) = 0;
                            % hidden thetas u
                            udE1_hidden_bias_div(a) = 0;
                            udE2_hidden_bias_div(a) = 0;
                            udE3_hidden_bias_div(a) = 0;
                        end
                        if ~isnan(v_matrix(i-1,j)) && ~isnan(v_matrix(i+1,j))
                            % outer weights v
                            vdE1_outer_div(a) = 0;
                            vdE2_outer_div(a) = 0;
                            vdE3_outer_div(a) = 0;
                            % outer thetas v
                            vdE1_outer_bias_div = 0;
                            vdE2_outer_bias_div = 0;
                            vdE3_outer_bias_div = 0;
                            % hidden weights v
                            vdE1_hidden_div(a,b) = 0;
                            vdE2_hidden_div(a,b) = 0;
                            vdE3_hidden_div(a,b) = 0;
                            % hidden thetas u
                            vdE1_hidden_bias_div(a) = 0;
                            vdE2_hidden_bias_div(a) = 0;
                            vdE3_hidden_bias_div(a) = 0;
                        end
                            
                        uE_div_outer(a) = uE_div_outer(a) + (1/(4*h_step^2))*udE1_outer_div(a) + (1/(4*k_step^2))*udE2_outer_div(a) + (1/(2*h_step*k_step))*udE3_outer_div(a);    
                        uE_div_outer_bias = uE_div_outer_bias + (1/(4*h_step^2))*udE1_outer_bias_div + (1/(4*k_step^2))*udE2_outer_bias_div + (1/(2*h_step*k_step))*udE3_outer_bias_div;
                        uE_div_hidden(a,b) = uE_div_hidden(a,b) + (1/(4*h_step^2))*udE1_hidden_div(a,b) + (1/(4*k_step^2))*udE2_hidden_div(a,b) + (1/(2*h_step*k_step))*udE3_hidden_div(a,b);
                        uE_div_hidden_bias(a) = uE_div_hidden_bias(a) + (1/(4*h_step^2))*udE1_hidden_bias_div(a) + (1/(4*k_step^2))*udE2_hidden_bias_div(a) + (1/(2*h_step*k_step))*udE3_hidden_bias_div(a);

                        vE_div_outer(a) = vE_div_outer(a) + (1/(4*h_step^2))*vdE1_outer_div(a) + (1/(4*k_step^2))*vdE2_outer_div(a) + (1/(2*h_step*k_step))*vdE3_outer_div(a);
                        vE_div_outer_bias = vE_div_outer_bias + (1/(4*h_step^2))*vdE1_outer_bias_div + (1/(4*k_step^2))*vdE2_outer_bias_div + (1/(2*h_step*k_step))*vdE3_outer_bias_div;
                        vE_div_hidden(a,b) = vE_div_hidden(a,b) + (1/(4*h_step^2))*vdE1_hidden_div(a,b) + (1/(4*k_step^2))*vdE2_hidden_div(a,b) + (1/(2*h_step*k_step))*vdE3_hidden_div(a,b);
                        vE_div_hidden_bias(a) = vE_div_hidden_bias(a) + (1/(4*h_step^2))*vdE1_hidden_bias_div(a) + (1/(4*k_step^2))*vdE2_hidden_bias_div(a) + (1/(2*h_step*k_step))*vdE3_hidden_bias_div(a);
                    end
                end
            end
        end
    end
    

    

    %initializing dE matrices (should match with weight matrix - udE w/ first 5 rows, vdE w/ rows 5-10)
    udE = zeros(5,9);
    vdE = zeros(5,9);
    uWnew = zeros(5,9);
    vWnew = zeros(5,9);
    Wnew = zeros(10,9);

    for m = 1:5
        for k = 1:9
            
            if m == 5 && k < 5  % outer weight gradients
                
                for j = 1:interior_point_count
                    udE(m, k) = udE(m, k) - ud(j) * uz(k, j);
                    vdE(m, k) = vdE(m, k) - vd(j) * vz(k, j);
                end
            %udE(m,k) = udE(m,k) + uE_div_outer(k);
            %vdE(m,k) = vdE(m,k) + vE_div_outer(k);
            udE(m,k) = udE(m,k) - uE_div_outer(k);
            vdE(m,k) = vdE(m,k) - vE_div_outer(k);
            end
            if m == 5 && k == 5 % outer weight bias term gradient
                
                for j = 1:interior_point_count
                    udE(m, k) = udE(m, k) + ud(j);
                    vdE(m, k) = vdE(m, k) + vd(j);
                end
            %udE(m,k) = udE(m,k) + uE_div_outer_bias;
            %vdE(m,k) = vdE(m,k) + vE_div_outer_bias;
            udE(m,k) = udE(m,k) - uE_div_outer_bias;
            vdE(m,k) = vdE(m,k) - vE_div_outer_bias;
            end
            if m < 5 && k ~= 9 % hidden weight gradients
                
                for j = 1:interior_point_count
                    udE(m, k) = udE(m, k) - ud(j) * W(5, m) * uz(m, j) * (1 - uz(m, j)) * ux_training(j, k);
                    vdE(m, k) = vdE(m, k) - vd(j) * W(10, m) * vz(m, j) * (1 - vz(m, j)) * vx_training(j, k);
                end
            %udE(m,k) = udE(m,k) + uE_div_hidden(m,k);
            %vdE(m,k) = vdE(m,k) + vE_div_hidden(m,k);
            udE(m,k) = udE(m,k) - uE_div_hidden(m,k);
            vdE(m,k) = vdE(m,k) - vE_div_hidden(m,k);
            end
            if m < 5 && k == 9 % hidden weight bias term gradients
                
                for j = 1:interior_point_count
                    udE(m, k) = udE(m, k) + ud(j) * W(5, m) * uz(m, j) * (1 - uz(m, j));
                    vdE(m, k) = vdE(m, k) + vd(j) * W(10, m) * vz(m, j) * (1 - vz(m, j));
                end
            %udE(m,k) = udE(m,k) + uE_div_hidden_bias(m);
            %vdE(m,k) = vdE(m,k) + vE_div_hidden_bias(m); 
            udE(m,k) = udE(m,k) - uE_div_hidden_bias(m);
            vdE(m,k) = vdE(m,k) - vE_div_hidden_bias(m); 
            end
        end
    end
    
    for m = 1:5
        for k = 1:9
            uWnew(m,k) = W(m,k) + n*udE(m,k);
            vWnew(m,k) = W(m+5,k) + n*vdE(m,k);

            Wnew(m,k) = uWnew(m,k);
            Wnew(m+5,k) = vWnew(m,k);
        end
    end

    W = Wnew;
    
    E_vector_g(d_iter) = E;
    d_iter = d_iter + 1
    %E

end

% plotting error
figure;
plot(1:length(E_vector_g), E_vector_g)
xlabel('Iteration')
ylabel('Error')
title('Gradient Error')
grid on
set(gcf,'papersize',[5 4],'paperposition',[0 0 5 4]);
print('-dpdf','N_PINN_ROMS1_15_grad_err.pdf');


%%

% de-normalizing

% Inverse min-max normalization for u_new
if u_min_val == u_max_val
    warning('u_min_val and u_max_val are the same. u_new will be filled with constant value.');
    u_new = u_min_val * ones(size(u_new));
else
    u_new = u_new * (u_max_val - u_min_val) + u_min_val;
end

% Inverse min-max normalization for v_new
if v_min_val == v_max_val
    warning('v_min_val and v_max_val are the same. v_new will be filled with constant value.');
    v_new = v_min_val * ones(size(v_new));
else
    v_new = v_new * (v_max_val - v_min_val) + v_min_val;
end


%%

format long

% divergence of the original field
Original_Divergence = 0;
for i = 2:(field_size(1) - 1)
    for j = 2:(field_size(2) - 1)
        % divergence = 0; physics informed error component
        Original_Divergence = Original_Divergence + ((u_save(i,j+1) - u_save(i,j-1))/(2*h_step) + (v_save(i+1,j) - v_save(i-1,j))/(2*k_step))^2;
    end
end
Original_Divergence

% divergence of the imputed field
New_Divergence = 0;
for i = 2:(field_size(1) - 1)
    for j = 2:(field_size(2) - 1)
        % divergence = 0; physics informed error component
        New_Divergence = New_Divergence + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;
    end
end
New_Divergence

Accuracy = sum((u_save(:) - u_new(:)).^2 + (v_save(:) - v_new(:)).^2)


% Compute nan mask
nan_mask = isnan(u_matrix) | isnan(v_matrix);

% Extract row-column boundaries (pixel space)
boundaries = bwboundaries(nan_mask, 'noholes');


figure;
%[X, Y] = meshgrid(1:colCount, 1:rowCount);
%quiver(X, Y, u_new, v_new, 'AutoScale', 'on');
quiver(ln_save, lt_save, u_new, v_new, 'AutoScale', 'on');
axis equal;
grid on;
hold on;

% Convert each pixel-space boundary to lat/lon
for k = 1:length(boundaries)
    boundary = boundaries{k};  % N x 2 array [row, col]
    
    % Preallocate longitude and latitude for this boundary
    lon_boundary = zeros(size(boundary,1), 1);
    lat_boundary = zeros(size(boundary,1), 1);
    
    for i = 1:size(boundary,1)
        row = boundary(i,1);
        col = boundary(i,2);
        lon_boundary(i) = ln_save(row, col);
        lat_boundary(i) = lt_save(row, col);
    end
    
    % Plot the boundary in geographic coordinates
    plot(lon_boundary, lat_boundary, 'k-', 'LineWidth', 2);  % Black line
end

xlabel('Longitude');
ylabel('Latitude');
title('Imputed Velocity Field');

%set(gcf,'papersize',[5 4],'paperposition',[0 0 5 4]);
%print('-dpdf','N_PINN_ROMS2_15_imputed.pdf');


