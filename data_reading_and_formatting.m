
% data reading

% Reading ROMS velocity field data from CENCOOS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the URL for ROMS data on September 2, 2019
%url = 'http://thredds.cencoos.org/thredds/dodsC/cencoos/ccsnrt/2019/2019_09/ccsnrt_2019_09_02.nc';
%url = 'http://thredds.cencoos.org/thredds/dodsC/cencoos/ccsnrt/2023/2023_06/ccsnrt_2023_06_29.nc'; % ROMS field1
url = 'http://thredds.cencoos.org/thredds/dodsC/cencoos/ccsnrt/2019/2019_09/ccsnrt_2022_01_01.nc';

 %Uncomment to display info on all netCDF variables
% ncdisp(url)

 %Uncomment to display info on specific netCDF variable
% ncdisp(url, 'time')

% Set geographical limits
%lat_max = 36.5; lat_min = 34.5;
%lat_max = 35.0; lat_min = 33.0; % ROMS field1 & 2
lat_max = 40.0; lat_min = 30.0; % ROMS field3
%lon_max = -120.5; lon_min = -123.0;
%lon_max = -122.0; lon_min = -124.0; % ROMS field1 & 2
lon_max = -122.0; lon_min = -130.0; % ROMS field3

% Read longitude and latitude data
lon_all = ncread(url, 'lon_rho');  % Full longitude grid
lat_all = ncread(url, 'lat_rho');  % Full latitude grid

% Find indices for the selected region
lon_j = find(lon_all(:,1) >= lon_min & lon_all(:,1) <= lon_max);
lat_j = find(lat_all(1,:) >= lat_min & lat_all(1,:) <= lat_max);

% Read time variable and convert it to MATLAB datenum format
timebase = erase(ncreadatt(url, 'time', 'units'), 'hours since ');
time = ncread(url, 'time') / 24 + datenum(timebase, 'yyyy-mm-dd HH:MM:SS');

% Choose the index for the specified date (September 2, 2019)
target_date = datenum('02-Sep-2019');
[~, t_j] = min(abs(time - target_date)); % Find closest time index

% Read velocity components at the surface (z index 1)
z_j = 1;  % Surface layer
u = ncread(url, 'urot', [lon_j(1) lat_j(1) z_j t_j], [length(lon_j) length(lat_j) 1 1]);
v = ncread(url, 'vrot', [lon_j(1) lat_j(1) z_j t_j], [length(lon_j) length(lat_j) 1 1]);

% Display date string for confirmation
disp(['Data retrieved for: ', datestr(time(t_j))])
%

% Plot the velocity field
[LN, LT] = meshgrid(lon_all(lon_j,1), lat_all(1,lat_j)); % Match lon/lat grid
quiver(LN, LT, u', v', 'k'); % Quiver plot
title(['ROMS Velocity Field on ', datestr(time(t_j))]);
xlabel('Longitude');
ylabel('Latitude');
grid on;
drawnow;


%

% Extract the correct longitude and latitude data for the desired region
[LON, LAT] = meshgrid(lon_all(lon_j), lat_all(1, lat_j)); % Correct indexing for latitude and longitude

% Flatten the data into column vectors
LON = LON(:); 
LAT = LAT(:); 

% Flatten the velocity data into column vectors (ensure the reshaping matches)
u_transpose = u';
v_transpose = v';

u_flat = u_transpose(:); 
v_flat = v_transpose(:);

% Check the dimensions of LON, LAT, u, and v to ensure they match
disp(['Number of points in LON: ', num2str(length(LON))]);
disp(['Number of points in LAT: ', num2str(length(LAT))]);
disp(['Number of points in u_flat: ', num2str(length(u_flat))]);
disp(['Number of points in v_flat: ', num2str(length(v_flat))]);


% Create a table with the extracted and flattened data
velocity_table = table(LON, LAT, u_flat, v_flat);

%%

% format a spatial u and v matrix

% Load the data
%load('velocity_table.mat');

% Extract variables
%LON = data.LON;
LON = velocity_table.LON;
LAT = velocity_table.LAT;

u_flat = velocity_table.u_flat;
v_flat = velocity_table.v_flat;

% Reindex longitude and latitude
unique_lon = unique(LON);
unique_lat = unique(LAT);

min_lon = min(unique_lon);
min_lat = min(unique_lat);

lon_idx = round((LON - min_lon) / 0.1) + 1;
lat_idx = round((LAT - min_lat) / 0.1) + 1;

% Determine grid size
num_lon = length(unique_lon);
num_lat = length(unique_lat);

% Initialize matrices
u_matrix = NaN(num_lat, num_lon);
v_matrix = NaN(num_lat, num_lon);

% Populate matrices
for k = 1:length(LON)
    u_matrix(lat_idx(k), lon_idx(k)) = u_flat(k);
    v_matrix(lat_idx(k), lon_idx(k)) = v_flat(k);
end

u_matrix = flipud(u_matrix);
v_matrix = flipud(v_matrix);

% Display matrix sizes
disp(['Size of u_matrix: ', num2str(size(u_matrix))]);
disp(['Size of v_matrix: ', num2str(size(v_matrix))]);

u_save = u_matrix;
v_save = v_matrix;
lt_save = LT;
ln_save = LN;

% Optional: Save results
%save('reindexed_velocity.mat', 'u_matrix', 'v_matrix');
save('u_save');
save('v_save');
save('ln_save');
save('lt_save');
