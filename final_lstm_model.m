clear; clc; close all;

%% Configuration
rng(42);
file_path       = 'C:/Users/maxcr/OneDrive/Documenten/Research assignment/Data/Site15_29m.xlsx';
parameters      = {'s_wht','mean_fr','wind_speed'};
years           = 2001:2010;
training_years  = 2001:2008;
testing_years   = 2009:2010;

sequence_length   = 48;     
hours_per_week    = 168;    
rolling_weeks     = 12;     

maxEpochsInitial  = 40;     
maxEpochsRefit    = 5;      
numHiddenUnits    = 35;     
miniBatchSize     = 72;     
learnRate         = 0.0005; 

hours_per_refit   = 5 * 24; 

%% Read & Prepare Data
disp('Loading data...');
data = struct();
for sheetIdx = 1:length(years)
    try
        tempTable = readtable(file_path, 'Sheet', sheetIdx);
    catch ME
        error('Error reading sheet %d: %s', sheetIdx, ME.message);
    end
    for p = 1:length(parameters)
        pName = parameters{p};
        if ismember(pName, tempTable.Properties.VariableNames)
            temp_data = tempTable.(pName);
            temp_data = fillmissing(temp_data,'linear');
            data(sheetIdx).(pName) = temp_data;
        else
            data(sheetIdx).(pName) = nan(height(tempTable),1);
        end
    end
end

%% Split and Normalize
disp('Splitting and normalizing data...');
trainDataConcat  = struct();
testDataConcat   = struct();
normalization_params = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    trainValues = [];
    for y = training_years
        idx = find(years == y);
        trainValues = [trainValues; data(idx).(pName)];
    end
    testValues = [];
    for y = testing_years
        idx = find(years == y);
        testValues = [testValues; data(idx).(pName)];
    end
    
    [trainNorm, ps] = mapminmax(trainValues', 0, 1);
    trainNorm = trainNorm';
    testNorm  = mapminmax('apply', testValues', ps);
    testNorm  = testNorm';
    
    trainDataConcat.(pName) = trainNorm;
    testDataConcat.(pName)  = testNorm;
    normalization_params.(pName) = ps;
end

%% Helper Function for LSTM Data
createLSTMData = @(series, seqLen) deal( ...
    arrayfun(@(i) series(i:i+seqLen-1)', 1:(length(series)-seqLen), ...
             'UniformOutput', false)', ...
    series(seqLen+1:end) );

%% Initial LSTM Training
disp('Initial training of LSTM models...');
models = struct();
for p = 1:length(parameters)
    pName = parameters{p};
    [XTrain, YTrain] = createLSTMData(trainDataConcat.(pName), sequence_length);
    layers = [
        sequenceInputLayer(1)
        lstmLayer(numHiddenUnits, 'OutputMode','last')
        fullyConnectedLayer(1)
        regressionLayer
    ];
    opts = trainingOptions('adam',...
        'MaxEpochs', maxEpochsInitial,...
        'MiniBatchSize', miniBatchSize,...
        'InitialLearnRate', learnRate,...
        'GradientThreshold', 1,...
        'Shuffle','once', ...
        'Verbose', false,...
        'Plots','none');
    net = trainNetwork(XTrain, YTrain, layers, opts);
    models.(pName).net = net;
end

%% Forecast & Refit (Every 5 Days)
disp('Starting forecasting in 5-day blocks with refit...');
forecasts = struct();
residuals = struct();
metrics   = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    nTest = length(testDataConcat.(pName));
    forecasts.(pName) = zeros(nTest,1);
    residuals.(pName) = zeros(nTest,1);
    metrics.(pName)   = struct('MSE',[],'MAE',[],'R2',[],'RMSE',[]);
end

numTestSamples = length(testDataConcat.(parameters{1}));
max_num_refits = ceil(numTestSamples / hours_per_refit);

for block = 1:max_num_refits
    fprintf('\n====== 5-Day Block %d of %d ======\n', block, max_num_refits);
    startIdx_block = (block - 1)*hours_per_refit + 1;
    endIdx_block   = min(block*hours_per_refit, numTestSamples);
    
    for p = 1:length(parameters)
        pName = parameters{p};
        nTest_p   = length(testDataConcat.(pName));
        max_block_p = ceil(nTest_p / hours_per_refit);
        if block > max_block_p
            continue;
        end
        endIdx_p   = min(endIdx_block, nTest_p);
        startIdx_p = startIdx_block;
        current_p_hrs = endIdx_p - startIdx_p + 1;
        net = models.(pName).net;
        net = resetState(net);
        currentTrainData = trainDataConcat.(pName);
        YF_lstm = zeros(current_p_hrs, 1);
        
        if length(currentTrainData) < sequence_length
            YF_lstm(:) = NaN;
        else
            lastSeq = currentTrainData(end-sequence_length+1 : end)';
            [net, firstPred] = predictAndUpdateState(net, {lastSeq});
            YF_lstm(1) = firstPred;
            for h = 2:current_p_hrs
                [net, predStep] = predictAndUpdateState(net, {YF_lstm(h-1)});
                YF_lstm(h) = predStep;
            end
        end
        YF_lstm(YF_lstm < 0) = 0;
        
        Y_actual = testDataConcat.(pName)(startIdx_p:endIdx_p);
        forecasts.(pName)(startIdx_p:endIdx_p) = YF_lstm;
        residuals.(pName)(startIdx_p:endIdx_p) = Y_actual - YF_lstm;
        
        mse_  = mean((Y_actual - YF_lstm).^2, 'omitnan');
        mae_  = mean(abs(Y_actual - YF_lstm), 'omitnan');
        rmse_ = sqrt(mse_);
        ss_res= sum((Y_actual - YF_lstm).^2, 'omitnan');
        ss_tot= sum((Y_actual - mean(Y_actual,'omitnan')).^2, 'omitnan');
        r2_   = 1 - (ss_res / ss_tot);
        
        metrics.(pName).MSE  = [metrics.(pName).MSE;  mse_];
        metrics.(pName).MAE  = [metrics.(pName).MAE;  mae_];
        metrics.(pName).RMSE = [metrics.(pName).RMSE; rmse_];
        metrics.(pName).R2   = [metrics.(pName).R2;   r2_];
        
        updatedTrainDataFull = [currentTrainData; Y_actual];
        keepSize = rolling_weeks*hours_per_week + sequence_length;
        if length(updatedTrainDataFull) > keepSize
            updatedTrainDataFull(1:end-keepSize) = [];
        end
        trainDataConcat.(pName) = updatedTrainDataFull;
        
        try
            [XRefit, YRefit] = createLSTMData(updatedTrainDataFull, sequence_length);
            refitOpts = trainingOptions('adam',...
                'MaxEpochs', maxEpochsRefit,...
                'MiniBatchSize', miniBatchSize,...
                'InitialLearnRate', learnRate,...
                'GradientThreshold',1,...
                'Shuffle','once',...
                'Verbose',false,...
                'Plots','none');
            net = trainNetwork(XRefit, YRefit, net.Layers, refitOpts);
            models.(pName).net = net;
        catch
        end
    end
end

%% Final Evaluation & Plots
disp('Computing final evaluation metrics & creating final plots...');
forecast_s_wht      = [];
forecast_mean_fr    = [];
forecast_wind_speed = [];

for p = 1:length(parameters)
    pName = parameters{p};
    Y_actual_full   = testDataConcat.(pName);
    Y_forecast_full = forecasts.(pName);
    nTest_p = length(Y_actual_full);
    Y_forecast_full = Y_forecast_full(1:nTest_p);
    ps = normalization_params.(pName);
    Y_actual_denorm   = mapminmax('reverse', Y_actual_full', ps)';
    Y_forecast_denorm = mapminmax('reverse', Y_forecast_full', ps)';
    Y_forecast_denorm(Y_forecast_denorm < 0) = 0;
    
    mse_  = mean((Y_actual_full - Y_forecast_full).^2,'omitnan');
    mae_  = mean(abs(Y_actual_full - Y_forecast_full),'omitnan');
    rmse_ = sqrt(mse_);
    ss_res= sum((Y_actual_full - Y_forecast_full).^2,'omitnan');
    ss_tot= sum((Y_actual_full - mean(Y_actual_full,'omitnan')).^2,'omitnan');
    r2_   = 1 - (ss_res / ss_tot);
    
    fprintf('\n=== Final Results for %s (2009-2010) ===\n', pName);
    fprintf('Average MSE : %.4f\n',  mse_);
    fprintf('Average MAE : %.4f\n',  mae_);
    fprintf('Average RMSE: %.4f\n',  rmse_);
    fprintf('Average R^2 : %.4f\n',  r2_);
    
    switch pName
        case 's_wht'
            forecast_s_wht = Y_forecast_denorm;
        case 'mean_fr'
            forecast_mean_fr = Y_forecast_denorm;
        case 'wind_speed'
            forecast_wind_speed = Y_forecast_denorm;
    end
    
    figure('Name',['Final Scatter ', pName],'NumberTitle','off');
    scatter(Y_actual_denorm, Y_forecast_denorm, 10, 'filled'); hold on;
    minVal = min([Y_actual_denorm; Y_forecast_denorm]);
    maxVal = max([Y_actual_denorm; Y_forecast_denorm]);
    plot([minVal,maxVal],[minVal,maxVal],'r--','LineWidth',2);
    title(['Scatter: Actual vs Forecast ', pName,' (2009-2010)']);
    xlabel('Actual'); ylabel('Forecast'); grid on; hold off;
    
    figure('Name',['Final LinePlot ', pName],'NumberTitle','off');
    plot(Y_actual_denorm, 'b', 'LineWidth',1.5); hold on;
    plot(Y_forecast_denorm, 'r--','LineWidth',1.5);
    legend('Actual','Forecast');
    title(['Actual vs Forecast ', pName,' (2009-2010)']);
    xlabel('Hour'); ylabel(pName);
    grid on; hold off;
    
    figure('Name',['Final Residuals ', pName],'NumberTitle','off');
    residuals_final = Y_actual_denorm - Y_forecast_denorm;
    plot(residuals_final,'k');
    xlabel('Hour'); ylabel('Residual');
    title(['Residuals ', pName,' (2009-2010)']);
    grid on; hold off;
end

disp('Forecasting completed with only final plots for 2009-2010.');
disp('Workspace variables: forecast_s_wht, forecast_mean_fr, forecast_wind_speed.');
