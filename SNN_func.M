function [Accuracy_vector, New_Divergence_vector] = SNN_func(total_iterations)

    Accuracy_vector = zeros(1,total_iterations);
    New_Divergence_vector = zeros(1,total_iterations);    
    
    
    % spatial PINN
    
    % HF_field1 
    load('HF_field1/ln_save.mat');
    load('HF_field1/lt_save.mat');
    load('HF_field1/u_save.mat');
    load('HF_field1/v_save.mat');
    load('HF_field1/u_matrix_15.mat');
    load('HF_field1/v_matrix_15.mat');

    HF_ln_save = ln_save;
    HF_lt_save = lt_save;

    HF_u_save = u_save;
    HF_v_save = v_save;

    HF_u_matrix = u_matrix_15;
    HF_v_matrix = v_matrix_15;
    
    % ROMS_field1
    load('ROMS_field1/ln_save.mat');
    load('ROMS_field1/lt_save.mat');
    load('ROMS_field1/u_save.mat');
    load('ROMS_field1/v_save.mat');
    load('ROMS_field1/u_matrix_15.mat');
    load('ROMS_field1/v_matrix_15.mat');

    ROMS1_ln_save = ln_save;
    ROMS1_lt_save = lt_save;

    ROMS1_u_save = u_save;
    ROMS1_v_save = v_save;
    
    ROMS1_u_matrix = u_matrix_15;
    ROMS1_v_matrix = v_matrix_15;
    
    % ROMS_field2
    load('ROMS_field2/ln_save.mat');
    load('ROMS_field2/lt_save.mat');
    load('ROMS_field2/u_save.mat');
    load('ROMS_field2/v_save.mat');
    load('ROMS_field2/u_matrix_15.mat');
    load('ROMS_field2/v_matrix_15.mat');
    
    ROMS2_ln_save = ln_save;
    ROMS2_lt_save = lt_save;
    
    ROMS2_u_save = u_save;
    ROMS2_v_save = v_save;

    ROMS2_u_matrix = u_matrix_15;
    ROMS2_v_matrix = v_matrix_15;

    total_iteration = 1;
    for total_iteration = 1:total_iterations
        
        if total_iteration < 201
            ln_save = ROMS1_ln_save;
            lt_save = ROMS1_lt_save;
    
            u_save = ROMS1_u_save;
            v_save = ROMS1_v_save;

            u_matrix = ROMS1_u_matrix;
            v_matrix = ROMS1_v_matrix;
        end
        if total_iteration > 200 && total_iteration < 401
            ln_save = ROMS2_ln_save;
            lt_save = ROMS2_lt_save;
    
            u_save = ROMS2_u_save;
            v_save = ROMS2_v_save;

            u_matrix = ROMS2_u_matrix;
            v_matrix = ROMS2_v_matrix;
        end
        if total_iteration > 400 && total_iteration < 601
            ln_save = HF_ln_save;
            lt_save = HF_lt_save;
    
            u_save = HF_u_save;
            v_save = HF_v_save;
            
            u_matrix = HF_u_matrix;
            v_matrix = HF_v_matrix;
        end
    
    %%
    
    % plot the original and holey velocity fields
    
    [rowCount, colCount] = size(u_matrix);
    field_size = size(u_matrix);
    
    %%
    
    % normalize
    
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
    
    % extract interior points
    
    % Find linear indices of non-NaN values
    u_non_nan_idx = find(~isnan(u_matrix));
    v_non_nan_idx = find(~isnan(v_matrix));
    
    % Get corresponding row and column indices
    [u_row_idx, u_col_idx] = ind2sub(size(u_matrix), u_non_nan_idx);
    [v_row_idx, v_col_idx] = ind2sub(size(v_matrix), v_non_nan_idx);
    
    % Create a 2 x M matrix of coordinates: first row is row indices, second is column indices
    u_coordinates = [u_row_idx, u_col_idx];
    v_coordinates = [v_row_idx, v_col_idx];
    
    % Extract the non-NaN values into a column vector
    u_values = u_matrix(u_non_nan_idx);
    v_values = v_matrix(v_non_nan_idx);
    
    interior_point_count = length(u_values);
    
    %%
    
    
    % spatial neural network

    
    
        f = @(v) 1./(1 + exp(-v));
        
        n = 0.005;
        
        % Naive Random Search
        
        naive_iter = 10000;
        n_iter = 1;
        Ebest = 1000;
        
        E_vector_n = zeros(1, naive_iter-1);
        
        while n_iter < naive_iter
            E = 0;
        
            % initial random weights
            % 10x9 random matrix (but the last 4 entries d(5,6)-d(5,9) are 0)
            h = rand(10,5);        % adjust according to patch size  
            for i = [1:4, 6:9]
                h(i, 4) = 0;
                h(i, 5) = 0;
            end
            %W = h;   
            
            num_weights = 20;
        
            %initialize uv, uz, and uy
            uv = zeros(4,interior_point_count);
            uz = zeros(4, interior_point_count);
            uy = zeros(interior_point_count);
        
            vv = zeros(4,interior_point_count);
            vz = zeros(4, interior_point_count);
            vy = zeros(interior_point_count);
            
            for j = 1:interior_point_count
        
                ux = u_coordinates(j,:); % this calls row / training pair j
                vx = v_coordinates(j,:);
        
                for i = 1:4
                    uv(i,j) = h(i,1)*ux(1) + h(i,2)*ux(2) - h(i,3);
                    uz(i,j) = f(uv(i,j));
        
                    vv(i,j) = h(i+5,1)*vx(1) + h(i+5,2)*vx(2) - h(i+5,3);
                    vz(i,j) = f(vv(i,j));
                end
        
                uy(j) = f( h(5,1)*uz(1,j) + h(5,2)*uz(2,j) + h(5,3)*uz(3,j) + h(5,4)*uz(4,j) - h(5,5) );
                vy(j) = f( h(10,1)*vz(1,j) + h(10,2)*vz(2,j) + h(10,3)*vz(3,j) + h(10,4)*vz(4,j) - h(10,5) );
        
                E = E + (uy(j) - u_values(j))^2 + (vy(j) - v_values(j))^2;
            end
            
            if E < Ebest
                Ebest = E;
                W = h;
            end
        
            E_vector_n(n_iter) = Ebest;
            n_iter = n_iter + 1;
        
        end
            
        % Fixed Step-Size Gradient Descent Algorithm
        
        descent_iter = 10000;
        d_iter = 1;
        
        E_vector_g = zeros(1, descent_iter-1);
        
        ud = zeros(interior_point_count);
        vd = zeros(interior_point_count);
        %dE = zeros(1:num_weights);
        %W = h;
        
        while d_iter < descent_iter
            E = 0;
            for j = 1:interior_point_count
                ux = u_coordinates(j,:);
                vx = v_coordinates(j,:);
        
                for i = 1:4
                    uv(i,j) = W(i,1)*ux(1) + W(i,2)*ux(2) - W(i,3);
                    uz(i,j) = f(uv(i,j));
        
                    vv(i,j) = W(i+5,1)*vx(1) + W(i+5,2)*vx(2) - W(i+5,3);
                    vz(i,j) = f(vv(i,j));
                end
        
                uy(j) = f( W(5,1)*uz(1,j) + W(5,2)*uz(2,j)  + W(5,3)*uz(3,j) + W(5,4)*uz(4,j) - W(5,5) );
                vy(j) = f( W(10,1)*vz(1,j) + W(10,2)*vz(2,j) + W(10,3)*vz(3,j) + W(10,4)*vz(4,j) - W(10,5) );
                    
                ud(j) = (uy(j) - u_values(j))*uy(j)*(1 - uy(j));
                vd(j) = (vy(j) - v_values(j))*vy(j)*(1 - vy(j));
        
                E = E + (uy(j) - u_values(j))^2 + (vy(j) - v_values(j))^2;
            end
            
            
            %initializing dE matrices (should match with weight matrix - udE w/ first 5 rows, vdE w/ rows 5-10)
            udE = zeros(5,5);
            vdE = zeros(5,5);
            uWnew = zeros(5,5);
            vWnew = zeros(5,5);
            Wnew = zeros(10,5);
        
            for m = 1:5
                for k = 1:5
                    
                    if m == 5 && k < 5  % outer weight gradients
                        
                        for j = 1:interior_point_count
                            udE(m, k) = udE(m, k) - ud(j) * uz(k, j);
                            vdE(m, k) = vdE(m, k) - vd(j) * vz(k, j);
                        end
                        
                    end
                    if m == 5 && k == 5 % outer weight bias term gradient
                        
                        for j = 1:interior_point_count
                            udE(m, k) = udE(m, k) + ud(j);
                            vdE(m, k) = vdE(m, k) + vd(j);
                        end
                        
                    end
                    if m < 5 && k < 3 % hidden weight gradients
                        
                        for j = 1:interior_point_count
                            udE(m, k) = udE(m, k) - ud(j) * W(5, m) * uz(m, j) * (1 - uz(m, j)) * u_coordinates(j, k);
                            vdE(m, k) = vdE(m, k) - vd(j) * W(10, m) * vz(m, j) * (1 - vz(m, j)) * v_coordinates(j, k);
                        end
                        
                    end
                    if m < 5 && k == 3 % hidden weight bias term gradients
                        
                        for j = 1:interior_point_count
                            udE(m, k) = udE(m, k) + ud(j) * W(5, m) * uz(m, j) * (1 - uz(m, j));
                            vdE(m, k) = vdE(m, k) + vd(j) * W(10, m) * vz(m, j) * (1 - vz(m, j));
                        end
                        
                    end
                    
                end
            end
            
            for m = 1:5
                for k = 1:5
                    uWnew(m,k) = W(m,k) + n*udE(m,k);
                    vWnew(m,k) = W(m+5,k) + n*vdE(m,k);
        
                    Wnew(m,k) = uWnew(m,k);
                    Wnew(m+5,k) = vWnew(m,k);
                end
            end
        
            W = Wnew;
        
            E_vector_g(d_iter) = E;
            d_iter = d_iter + 1;
            %E
        
        end
       
        
        %%
        
        % original
        
        %%
        
        % imputation take 2
        
        %format long
        
        field_size = size(u_matrix);
        u_new = u_matrix;
        v_new = v_matrix;
        
        for i = 1:field_size(1)
            for j = 1:field_size(2)
                if isnan(u_matrix(i,j))
                    % Hidden layer for u
                    for k = 1:4
                        uv(k) = W(k,1)*i + W(k,2)*j - W(k,3);
                        uz(k) = f(uv(k));
                    end
            
                    % Output neuron for u
                    uy = f( W(5,1)*uz(1) + W(5,2)*uz(2) + W(5,3)*uz(3) + W(5,4)*uz(4) - W(5,5) );
                    u_new(i,j) = uy;
            
                    % Hidden layer for v
                    for k = 1:4
                        vv(k) = W(k+5,1)*i + W(k+5,2)*j - W(k+5,3);
                        vz(k) = f(vv(k));
                    end
            
                    % Output neuron for v
                    vy = f( W(10,1)*vz(1) + W(10,2)*vz(2) + W(10,3)*vz(3) + W(10,4)*vz(4) - W(10,5) );
                    v_new(i,j) = vy;
                end
            end
        end
        
        
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
        
        h_step = 6;
        k_step = 6;
        
        % divergence of the imputed field
        New_Divergence = 0;
        for i = 2:(field_size(1) - 1)
            for j = 2:(field_size(2) - 1)
                % divergence = 0; physics informed error component
                New_Divergence = New_Divergence + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;
            end
        end
        %New_Divergence
        
        %disp('Original Divergence:'); disp(Original_Divergence);
        %disp('New Divergence:'); disp(New_Divergence);
        
        
        Accuracy = sum((u_save(:) - u_new(:)).^2 + (v_save(:) - v_new(:)).^2);
        
      
        Accuracy_vector(total_iteration) = Accuracy;
        New_Divergence_vector(total_iteration) = New_Divergence;
        total_iteration = total_iteration + 1
        
    end
end
