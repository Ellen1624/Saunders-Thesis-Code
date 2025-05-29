function [Accuracy_vector, New_Divergence_vector] = Rand_func(total_iterations)

    Accuracy_vector = zeros(1,total_iterations);
    New_Divergence_vector = zeros(1,total_iterations);
        
    % load in the data
        
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

    
    h_step = 6;
    k_step = 6;

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
        
        field_size = size(u_matrix);

        % Inputs: u_matrix and v_matrix (matrices with NaNs)
        % Outputs: u_new and v_new (NaNs replaced with random values between min and max of known values)
        
        % Copy original matrices
        u_new = u_matrix;
        v_new = v_matrix;
        
        % Find known (non-NaN) values
        u_known = u_matrix(~isnan(u_matrix));
        v_known = v_matrix(~isnan(v_matrix));
        
        % Determine min and max of known values
        u_min = min(u_known);
        u_max = max(u_known);
        
        v_min = min(v_known);
        v_max = max(v_known);
        
        % Find indices of NaNs
        u_nan_indices = isnan(u_matrix);
        v_nan_indices = isnan(v_matrix);
        
        % Generate random values between min and max for each NaN
        u_new(u_nan_indices) = u_min + (u_max - u_min) * rand(sum(u_nan_indices(:)), 1);
        v_new(v_nan_indices) = v_min + (v_max - v_min) * rand(sum(v_nan_indices(:)), 1);

        New_Divergence = 0;
        Accuracy = 0;
        for i = 2:(field_size(1) - 1)
            for j = 2:(field_size(2) - 1)
                % divergence = 0; physics informed error component
                New_Divergence = New_Divergence + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;
            end
        end
        %New_Divergence
                
        Accuracy = sum((u_save(:) - u_new(:)).^2 + (v_save(:) - v_new(:)).^2);
        
        Accuracy_vector(total_iteration) = Accuracy;
        New_Divergence_vector(total_iteration) = New_Divergence;


        total_iteration = total_iteration + 1
       
 
    end
end
