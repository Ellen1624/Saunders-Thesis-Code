function dual_histogram(data1, data2, avg1, avg2, base_avg, label1, label2, label3, label4, label5, bin_width)
% dual_histogram plots two overlaid histograms with average lines and saves to PDF.
%
% Parameters:
%   data1, data2        - Vectors of data to histogram
%   avg1, avg2          - Corresponding average values for data1 and data2
%   base_avg, rand_avg  - Baseline and random average values to mark with dashed lines
%   label1, label2      - Labels for legend for data1 and data2
%   filename            - PDF filename to save the figure (e.g., 'plot.pdf')

    % Define consistent colors
    color1 = [0.2, 0.6, 0.8];   % Blueish
    color2 = [0.8, 0.4, 0.4];   % Reddish

    figure;
    hold on;

    % Plot histograms
    histogram(data1, 'FaceColor', color1, 'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
          'BinWidth', bin_width, 'DisplayName', label1);
    histogram(data2, 'FaceColor', color2, 'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
          'BinWidth', bin_width, 'DisplayName', label2);

    % Plot average lines
    xline(avg1, '-', 'Color', color1, 'LineWidth', 2, 'DisplayName', [label1 ' Avg']);
    xline(avg2, '-', 'Color', color2, 'LineWidth', 2, 'DisplayName', [label2 ' Avg']);

    % Plot baseline and random average lines
    xline(base_avg, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Baseline Avg');
    %xline(rand_avg, ':k', 'LineWidth', 1.5, 'DisplayName', 'Random Avg');

    % Labels and legend
    xlabel(label3);
    ylabel('Frequency');
    title([label4 label1 ' vs ' label2 label5 ' Comparison']);
    legend('show');
    box on;

    % Save to PDF
    %set(gcf, 'PaperSize', [5 4], 'PaperPosition', [0 0 5 4]);
    %print('-dpdf', filename);

    hold off;
end
