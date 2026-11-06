%% =========================================================================
% CA2 - M2026: Surface EMG Physical Action Classification
% Full Implementation: Tasks 1, 2, 3 & 4 (High-Accuracy Random Forest)
% =========================================================================
clc;
clear;
close all;

%% -------------------------------------------------------------------------
% PATH DISCOVERY & SETUP
% -------------------------------------------------------------------------
candidate_paths = {
    '/MATLAB Drive/EMGPhysicalActionDataSet-20261029T071445Z-1-001', ...
    fullfile(pwd, 'EMGPhysicalActionDataSet-20261029T071445Z-1-001'), ...
    '/MATLAB Drive', ...
    pwd ...
};

data_root = '';
subject_folders = [];

for i = 1:length(candidate_paths)
    base_check = candidate_paths{i};
    if exist(base_check, 'dir')
        sub_dirs = dir(fullfile(base_check, 'sub*'));
        sub_dirs = sub_dirs([sub_dirs.isdir]);
        if ~isempty(sub_dirs)
            data_root = base_check;
            subject_folders = sub_dirs;
            break;
        end
        all_dirs = dir(base_check);
        all_dirs = all_dirs([all_dirs.isdir] & ~ismember({all_dirs.name}, {'.', '..'}));
        for j = 1:length(all_dirs)
            nested_check = fullfile(base_check, all_dirs(j).name);
            sub_dirs = dir(fullfile(nested_check, 'sub*'));
            sub_dirs = sub_dirs([sub_dirs.isdir]);
            if ~isempty(sub_dirs)
                data_root = nested_check;
                subject_folders = sub_dirs;
                break;
            end
        end
        if ~isempty(subject_folders), break; end
    end
end

if isempty(subject_folders)
    data_root = uigetdir(pwd, 'Please select the EMG Dataset root folder:');
    if data_root == 0, error('Selection cancelled.'); end
    subject_folders = dir(fullfile(data_root, 'sub*'));
    subject_folders = subject_folders([subject_folders.isdir]);
end

[~, sort_idx] = sort({subject_folders.name});
subject_folders = subject_folders(sort_idx);
num_subjects = length(subject_folders);

%% =========================================================================
% TASK 1: READ AND PLOT EMG SIGNALS
% =========================================================================
fprintf('\n--- TASK 1: Inspecting Files and Plotting EMG Signals ---\n');

% 1. Display number of folders in the dataset
head1 = sprintf('Directories (Subjects) Count: %d', num_subjects);
disp(head1);

sample_sub_path = fullfile(data_root, subject_folders(1).name);
find_txt_files = @(base, act) [dir(fullfile(base, act, 'txt', '*.txt')); dir(fullfile(base, act, '*.txt'))];

% 2. List the text files within each sub-directory for Subject 1
norm_files_sample = find_txt_files(sample_sub_path, 'Normal');
agg_files_sample  = find_txt_files(sample_sub_path, 'Aggressive');

fprintf('\nListing Normal Files for Subject 1:\n');
for i = 1:length(norm_files_sample)
    disp([' - ', norm_files_sample(i).name]);
end

fprintf('\nListing Aggressive Files for Subject 1:\n');
for i = 1:length(agg_files_sample)
    disp([' - ', agg_files_sample(i).name]);
end

% 3. Read and display the size of each signal array
norm_sample_file = fullfile(norm_files_sample(1).folder, norm_files_sample(1).name);
agg_sample_file  = fullfile(agg_files_sample(1).folder, agg_files_sample(1).name);

data_norm = readmatrix(norm_sample_file);
data_agg  = readmatrix(agg_sample_file);

disp(' ');
size_data_norm = sprintf('Size of %s: Rows:%d, Cols:%d', norm_files_sample(1).name, size(data_norm,1), size(data_norm,2));
size_data_agg  = sprintf('Size of %s: Rows:%d, Cols:%d', agg_files_sample(1).name, size(data_agg,1), size(data_agg,2));
disp(size_data_norm);
disp(size_data_agg);

% 4. Read and plot a signal from each of the channels
figure('Color', 'w', 'Position', [100, 100, 1100, 750], 'Name', 'Task 1: 8-Channel EMG Signals');
for ch = 1:8
    subplot(8, 2, 2*ch - 1);
    plot(data_norm(:, ch), 'b', 'LineWidth', 0.8);
    ylabel(sprintf('Ch %d', ch), 'FontWeight', 'bold');
    if ch == 1
        title(sprintf('Normal Action: %s', norm_files_sample(1).name), 'Interpreter', 'none');
    end
    grid on; xlim([1, size(data_norm, 1)]);
    
    subplot(8, 2, 2*ch);
    plot(data_agg(:, ch), 'r', 'LineWidth', 0.8);
    if ch == 1
        title(sprintf('Aggressive Action: %s', agg_files_sample(1).name), 'Interpreter', 'none');
    end
    grid on; xlim([1, size(data_agg, 1)]);
end
xlabel('Time Samples');
sgtitle('Task 1: Multi-Channel EMG Waveform Comparison', 'FontSize', 12, 'FontWeight', 'bold');
drawnow;

%% =========================================================================
% TASK 2 & 3: FEATURE EXTRACTION (Mean, Var + Advanced) & LABELS
% =========================================================================
fprintf('\n--- TASK 2 & 3: Feature & Label Extraction ---\n');

num_trials   = 15;  
N_seg        = 3;   
num_channels = 8;   
num_stats    = 5;   % Mean, Variance (Required) + MAV, RMS, WL (Accuracy boost)
Nf           = num_channels * N_seg * num_stats; 

all_features = [];
all_labels   = [];
action_types = {'Normal', 'Aggressive'};

for s_idx = 1:num_subjects
    sub_path = fullfile(data_root, subject_folders(s_idx).name);
    fprintf('Processing features for Subject %d...\n', s_idx);
    
    for a_type = 1:2
        txt_files = find_txt_files(sub_path, action_types{a_type});
        [~, file_sort_idx] = sort({txt_files.name});
        txt_files = txt_files(file_sort_idx);
        
        for f_idx = 1:length(txt_files)
            % Task 3.1: Generate class labels (1 to 20)
            class_label = f_idx + (a_type - 1) * 10;
            file_full_path = fullfile(txt_files(f_idx).folder, txt_files(f_idx).name);
            emg_matrix = readmatrix(file_full_path);
            
            if size(emg_matrix, 2) > 8
                emg_matrix = emg_matrix(:, end-7:end);
            end
            
            trial_len = floor(size(emg_matrix, 1) / num_trials);
            
            % Task 2.2-2.7: Segment, extract, concatenate
            for tr = 1:num_trials
                trial_start = (tr - 1) * trial_len + 1;
                trial_end   = tr * trial_len;
                trial_signal = emg_matrix(trial_start:trial_end, :);
                
                trial_feature_vec = zeros(1, Nf);
                feat_ptr = 1;
                
                for ch = 1:num_channels
                    ch_signal = trial_signal(:, ch);
                    seg_len = floor(length(ch_signal) / N_seg);
                    
                    for seg = 1:N_seg
                        seg_data = ch_signal((seg-1)*seg_len+1 : min(seg*seg_len, end));
                        
                        % Required base features
                        feat_mean = mean(seg_data);
                        feat_var  = var(seg_data);
                        % Advanced features for Random Forest accuracy
                        feat_mav  = mean(abs(seg_data));
                        feat_rms  = sqrt(mean(seg_data.^2));
                        feat_wl   = sum(abs(diff(seg_data)));
                        
                        trial_feature_vec(feat_ptr:feat_ptr+4) = [feat_mean, feat_var, feat_mav, feat_rms, feat_wl];
                        feat_ptr = feat_ptr + 5;
                    end
                end
                all_features = [all_features; trial_feature_vec]; %#ok<AGROW>
                all_labels   = [all_labels; class_label];         %#ok<AGROW>
            end
        end
    end
end
% Task 3.2: Concatenate target vectors
all_labels = all_labels(:);

fprintf('Feature Matrix Size: %d samples x %d features\n', size(all_features, 1), size(all_features, 2));

% Task 2.9: Save data to matlab array
save('emg_features_all.mat', 'all_features', 'all_labels', '-v7.3');
fprintf('Saved feature matrix and targets to "emg_features_all.mat"\n');

% Task 2.8: Display feature vectors of any two classes
figure('Color', 'w', 'Position', [150, 150, 950, 450], 'Name', 'Task 2: Feature Profiles');
c1_idx = find(all_labels == 1, 1, 'first');
c11_idx = find(all_labels == 11, 1, 'first');
plot(all_features(c1_idx, :), 'b-o', 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'Class 1 (Normal)');
hold on;
plot(all_features(c11_idx, :), 'r-s', 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'Class 11 (Aggressive)');
hold off;
grid on;
xlabel('Feature Index (8 Ch \times 3 Segments \times 5 Statistics)');
ylabel('Feature Value');
title('Task 2: Feature Vector Comparison (Class 1 vs Class 11)');
legend('Location', 'northeast');
drawnow;

%% =========================================================================
% TASK 4: CLASSIFICATION WITH RANDOM FOREST ENSEMBLE
% =========================================================================
fprintf('\n--- TASK 4: Classification Analysis (Random Forest / Bagged Trees) ---\n');

% Q1: Binary Classification
c_A = 1; c_B = 11; 
bin_idx = (all_labels == c_A) | (all_labels == c_B);
X_bin = all_features(bin_idx, :);
Y_bin = all_labels(bin_idx);

cv_bin = cvpartition(Y_bin, 'KFold', 5);
rf_bin_model = fitcensemble(X_bin, Y_bin, 'Method', 'Bag', 'NumLearningCycles', 50, 'CVPartition', cv_bin);
bin_acc  = (1 - kfoldLoss(rf_bin_model)) * 100;
fprintf('Binary Random Forest Accuracy: %.2f%%\n\n', bin_acc);

% Q2: Multi-Class Analysis (M=2 to 19)
fprintf('Starting Multi-Class Evaluation...\n');
num_classes_range = 2:19;
N_mc = 10; 
acc_matrix = zeros(length(num_classes_range), N_mc);
all_class_ids = unique(all_labels);

for m_idx = 1:length(num_classes_range)
    M = num_classes_range(m_idx);
    all_combos = nchoosek(all_class_ids, M);
    
    if size(all_combos, 1) <= N_mc
        combo_indices = 1:size(all_combos, 1);
    else
        combo_indices = randperm(size(all_combos, 1), N_mc);
    end
    
    fprintf('Running M = %2d classes... ', M);
    
    for mc = 1:length(combo_indices)
        combo = all_combos(combo_indices(mc), :);
        mask = ismember(all_labels, combo);
        X_sub = all_features(mask, :);
        Y_sub = all_labels(mask);
        
        cv_sub = cvpartition(Y_sub, 'KFold', 5);
        rf_model = fitcensemble(X_sub, Y_sub, 'Method', 'Bag', 'NumLearningCycles', 40, 'CVPartition', cv_sub);
        
        acc_matrix(m_idx, mc) = (1 - kfoldLoss(rf_model)) * 100;
    end
    
    if length(combo_indices) < N_mc
        acc_matrix(m_idx, length(combo_indices)+1:N_mc) = mean(acc_matrix(m_idx, 1:length(combo_indices)));
    end
    fprintf('Mean Acc: %.2f%%\n', mean(acc_matrix(m_idx, :)));
end

mean_acc_curve = mean(acc_matrix, 2);

%% =========================================================================
% VISUALIZATIONS (TASK 4 - HEATMAP & DEGRADATION)
% =========================================================================
figure('Color', 'w', 'Position', [150, 100, 850, 550], 'Name', 'Task 4: Accuracy Heatmap');
imagesc(1:N_mc, num_classes_range, acc_matrix);
colormap(parula); colorbar; caxis([min(acc_matrix(:)), 100]);
xlabel('Monte Carlo Run Index'); ylabel('Number of Classes (M)');
title('Task 4(b): Cross-Validation Accuracy Heatmap (Random Forest)');
set(gca, 'YDir', 'normal', 'YTick', num_classes_range);
for i = 1:size(acc_matrix, 1)
    for j = 1:size(acc_matrix, 2)
        text(j, num_classes_range(i), sprintf('%.1f', acc_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
    end
end

figure('Color', 'w', 'Position', [200, 150, 800, 480], 'Name', 'Task 4: Degradation Curve');
plot(num_classes_range, mean_acc_curve, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b'); hold on;
plot(num_classes_range, (1 ./ num_classes_range) * 100, 'r--', 'LineWidth', 1.5);
grid on; xlabel('Number of Classes (M)'); ylabel('Mean CV Accuracy (%)');
title('Task 4(c): Degradation of Accuracy vs Number of Classes');
legend({'Random Forest Mean Accuracy', 'Theoretical Random Chance (100/M)'}); xlim([2, 19]); ylim([0, 105]);

fprintf('\nAll Tasks Complete!\n');