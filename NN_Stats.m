
total_iterations = 600;
%{
[avg_accuracy, avg_divergence] = Avg_func();

[Rand_Accuracy, Rand_Divergence] = Rand_func(total_iterations);

avg_rand_accuracy = mean(Rand_Accuracy);
avg_rand_divergence = mean(Rand_Divergence);

[NNN_Accuracy, NNN_Divergence] = NNN_func(total_iterations);

avg_NNN_accuracy = mean(NNN_Accuracy);
avg_NNN_divergence = mean(NNN_Divergence);

[N_PINN_Accuracy, N_PINN_Divergence] = N_PINN_func(total_iterations);

avg_N_PINN_accuracy = mean(N_PINN_Accuracy);
avg_N_PINN_divergence = mean(N_PINN_Divergence);


[SNN_Accuracy, SNN_Divergence] = SNN_func(total_iterations);

avg_SNN_accuracy = mean(SNN_Accuracy);
avg_SNN_divergence = mean(SNN_Divergence);


[S_PINN_Accuracy, S_PINN_Divergence] = S_PINN_func(total_iterations);

avg_S_PINN_accuracy = mean(S_PINN_Accuracy);
avg_S_PINN_divergence = mean(S_PINN_Divergence);
%}

%{
figure;
histogram(Rand_Accuracy, 'BinWidth', 0.1); % Set bin width as appropriate
hold on;
xline(avg_rand_accuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
%xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Accuracy');
ylabel('Frequency');
title('Random Accuracy Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'Rand_Acc_Dist.pdf');

figure;
histogram(Rand_Divergence, 'BinWidth', 0.001); % Set bin width as appropriate
hold on;
xline(avg_rand_divergence, 'r-', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
%xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Divergence');
ylabel('Frequency');
title('Random Divergence Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'Rand_Div_Dist.pdf');


figure;
histogram(NNN_Accuracy, 'BinWidth', 0.1); % Set bin width as appropriate
hold on;
xline(avg_NNN_accuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'NN Avg');
xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Accuracy');
ylabel('Frequency');
title('Neighbor NN Accuracy Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'NNN_Acc_Dist.pdf');


figure;
histogram(NNN_Divergence, 'BinWidth', 0.001); % Set bin width as appropriate
hold on;
xline(avg_NNN_divergence, 'r-', 'LineWidth', 2, 'DisplayName', 'NN Avg');
xline(avg_divergence, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_divergence, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Divergence');
ylabel('Frequency');
title('Neighbor NN Divergence Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'NNN_Div_Dist.pdf');

figure;
histogram(N_PINN_Accuracy, 'BinWidth', 0.1); % Set bin width as appropriate
hold on;
xline(avg_N_PINN_accuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'PINN Avg');
xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Accuracy');
ylabel('Frequency');
title('Neighbor PINN Accuracy Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'N_PINN_Acc_Dist.pdf');

figure;
histogram(N_PINN_Divergence, 'BinWidth', 0.001); % Set bin width as appropriate
hold on;
xline(avg_N_PINN_divergence, 'r-', 'LineWidth', 2, 'DisplayName', 'PINN Avg');
xline(avg_divergence, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_divergence, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Divergence');
ylabel('Frequency');
title('Neighbor PINN Divergence Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'N_PINN_Div_Dist.pdf');
%}
%{
figure;
histogram(SNN_Accuracy, 'BinWidth', 0.1); % Set bin width as appropriate
hold on;
xline(avg_SNN_accuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'NN Avg');
xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Accuracy');
ylabel('Frequency');
title('Spatial NN Accuracy Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'SNN_Acc_Dist.pdf');


figure;
histogram(SNN_Divergence, 'BinWidth', 0.001); % Set bin width as appropriate
hold on;
xline(avg_SNN_divergence, 'r-', 'LineWidth', 2, 'DisplayName', 'NN Avg');
xline(avg_divergence, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_divergence, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Divergence');
ylabel('Frequency');
%ylim([0 55]);
title('Spatial NN Divergence Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'SNN_Div_Dist_Adjusted.pdf');
%}

%{
figure;
histogram(S_PINN_Accuracy, 'BinWidth', 0.1); % Set bin width as appropriate
hold on;
xline(avg_S_PINN_accuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'PINN Avg');
xline(avg_accuracy, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_accuracy, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Accuracy');
ylabel('Frequency');
title('Spatial PINN Accuracy Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'S_PINN_Acc_Dist.pdf');

figure;
histogram(S_PINN_Divergence, 'BinWidth', 0.001); % Set bin width as appropriate
hold on;
xline(avg_S_PINN_divergence, 'r-', 'LineWidth', 2, 'DisplayName', 'PINN Avg');
xline(avg_divergence, 'k-', 'LineWidth', 2, 'DisplayName', 'Baseline Avg');
%xline(avg_rand_divergence, 'k--', 'LineWidth', 2, 'DisplayName', 'Rand Avg');
legend show;
xlabel('Divergence');
ylabel('Frequency');
title('Spatial PINN Divergence Distribution');
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'S_PINN_Div_Dist.pdf');



dual_histogram(NNN_Accuracy, N_PINN_Accuracy, avg_NNN_accuracy, avg_N_PINN_accuracy, avg_accuracy, 'NN', 'PINN', 'Accuracy ', 'Neighbor ', ' Accuracy', 0.1)
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'N_Compare_Acc_Dist.pdf');

dual_histogram(NNN_Divergence, N_PINN_Divergence, avg_NNN_divergence, avg_N_PINN_divergence, avg_divergence, 'NN', 'PINN', 'Divergence ', 'Neighbor ', ' Divergence', 0.001)
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'N_Compare_Div_Dist.pdf');
%}

dual_histogram(SNN_Accuracy, S_PINN_Accuracy, avg_SNN_accuracy, avg_S_PINN_accuracy, avg_accuracy, 'NN', 'PINN', 'Accuracy', 'Spatial ', ' Accuracy', 0.1)
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'S_Compare_Acc_Dist.pdf');


%{
dual_histogram(SNN_Divergence, S_PINN_Divergence, avg_SNN_divergence, avg_S_PINN_divergence, avg_divergence, 'NN', 'PINN', 'Divergence ', 'Spatial ', ' Divergence', 0.001)
set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
print('-dpdf', 'S_Compare_Div_Dist.pdf');
%}

%{
quad_histogram(NNN_Accuracy, N_PINN_Accuracy, SNN_Accuracy, S_PINN_Accuracy, avg_NNN_accuracy, avg_N_PINN_accuracy, avg_SNN_accuracy, avg_S_PINN_accuracy, avg_accuracy, 'Neighbor NN', 'Neighbor PINN', 'Spatial NN', 'Spatial PINN', 'Accuracy', ' Accuracy', 0.1)
set(gcf, 'PaperSize', [7 6], 'PaperPosition', [0 0 7 6]);
print('-dpdf', 'All_Compare_Acc_Dist.pdf');
%}

%{
quad_histogram(NNN_Divergence, N_PINN_Divergence, SNN_Divergence, S_PINN_Divergence, avg_NNN_divergence, avg_N_PINN_divergence, avg_SNN_divergence, avg_S_PINN_divergence, avg_divergence, 'Neighbor NN', 'Neighbor PINN', 'Spatial NN', 'Spatial PINN', 'Divergence', ' Divergence', 0.001)
%ylim([0 125]);
set(gcf, 'PaperSize', [7 6], 'PaperPosition', [0 0 7 6]);
print('-dpdf', 'All_Compare_Div_Dist.pdf');
%}
%{
dual_histogram(N_PINN_Accuracy, S_PINN_Accuracy, avg_N_PINN_accuracy, avg_S_PINN_accuracy, avg_accuracy, 'Neighbor PINN', 'Spatial PINN', 'Accuracy', '', ' Accuracy', 0.1)
set(gcf, 'PaperSize', [7 6], 'PaperPosition', [0 0 7 6]);
print('-dpdf', 'PINN_Compare_Acc_Dist.pdf');



dual_histogram(N_PINN_Divergence, S_PINN_Divergence, avg_N_PINN_divergence, avg_S_PINN_divergence, avg_divergence, 'Neighbor PINN', 'Spatial PINN', 'Divergence ', '', ' Divergence', 0.001)
set(gcf, 'PaperSize', [7 6], 'PaperPosition', [0 0 7 6]);
print('-dpdf', 'PINN_Compare_Div_Dist.pdf');
%}
