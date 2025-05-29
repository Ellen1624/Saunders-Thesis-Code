function [Accuracy, New_Divergence] = Avg_func()
        
    % load in the data
        
    % HF_field1 
    load('HF_field1/ln_save.mat');
    load('HF_field1/lt_save.mat');
    load('HF_field1/u_save.mat');
    load('HF_field1/v_save.mat');
    load('HF_field1/u_matrix_20.mat');
    load('HF_field1/v_matrix_20.mat');

    HF_ln_save = ln_save;
    HF_lt_save = lt_save;

    HF_u_save = u_save;
    HF_v_save = v_save;

    HF_u_matrix = u_matrix_20;
    HF_v_matrix = v_matrix_20;
    
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

    iteration = 1;
    Accuracy_vector_avg = zeros(1,3);
    for iteration = 1:3
        if iteration == 1
            ln_save = ROMS1_ln_save;
            lt_save = ROMS1_lt_save;
    
            u_save = ROMS1_u_save;
            v_save = ROMS1_v_save;

            u_matrix = ROMS1_u_matrix;
            v_matrix = ROMS1_v_matrix;
        end
        if iteration == 2
            ln_save = ROMS2_ln_save;
            lt_save = ROMS2_lt_save;
    
            u_save = ROMS2_u_save;
            v_save = ROMS2_v_save;

            u_matrix = ROMS2_u_matrix;
            v_matrix = ROMS2_v_matrix;
        end
        if iteration == 3
            ln_save = HF_ln_save;
            lt_save = HF_lt_save;
    
            u_save = HF_u_save;
            v_save = HF_v_save;
            
            u_matrix = HF_u_matrix;
            v_matrix = HF_v_matrix;
        end
        
            field_size = size(u_matrix);
            h_step = 6;
            k_step = 6;
            
                
            % Inputs: u_matrix and v_matrix (matrices with NaNs)
            % Outputs: u_new and v_new (NaNs replaced with mean of known values)
            
            % Calculate mean of known values in u_matrix
            u_mean = mean(u_matrix(~isnan(u_matrix)), 'all');
            
            % Create u_new: same as u_matrix but replace NaNs with u_mean
            u_new = u_matrix;
            u_new(isnan(u_new)) = u_mean;
            
            % Calculate mean of known values in v_matrix
            v_mean = mean(v_matrix(~isnan(v_matrix)), 'all');
            
            % Create v_new: same as v_matrix but replace NaNs with v_mean
            v_new = v_matrix;
            v_new(isnan(v_new)) = v_mean;
            
            New_Divergence = 0;
            Accuracy = 0;
            for i = 2:(field_size(1) - 1)
                for j = 2:(field_size(2) - 1);
                    % divergence = 0; physics informed error component
                    New_Divergence = New_Divergence + ((u_new(i,j+1) - u_new(i,j-1))/(2*h_step) + (v_new(i+1,j) - v_new(i-1,j))/(2*k_step))^2;
                end
            end
            %New_Divergence
                    
            Accuracy_vector_avg(iteration) = sum((u_save(:) - u_new(:)).^2 + (v_save(:) - v_new(:)).^2)

            iteration = iteration + 1;

            if iteration == 1
                
    end
    Accuracy = mean(Accuracy_vector_avg)

end
