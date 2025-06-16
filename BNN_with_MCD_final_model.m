% Ensure a parallel pool is available
if isempty(gcp('nocreate'))
    parpool('local');  % starts a pool of workers with default settings
end

clc; close all;

%% 1. Configuration

file_path  = %fill in the file path here
parameters = {'s_wht','mean_fr','wind_speed'};
years      = 2001:2010;
training_years = 2001:2008;
testing_years  = 2009:2010;

hours_per_day   = 24;
hours_per_refit = 5 * hours_per_day;  

hours_per_week      = 7 * hours_per_day;
rolling_weeks       = 20;
rolling_window_size = rolling_weeks * hours_per_week;

modelConfigs = struct;

modelConfigs.s_wht.windowSize   = 48;
modelConfigs.s_wht.hiddenLayer1 = 128;
modelConfigs.s_wht.hiddenLayer2 = 64;
modelConfigs.s_wht.dropoutRate  = 0.08;
modelConfigs.s_wht.mcIterations = 10;
modelConfigs.s_wht.initialEpochs= 80;
modelConfigs.s_wht.refitEpochs  = 40;

modelConfigs.mean_fr.windowSize   = 48;
modelConfigs.mean_fr.hiddenLayer1 = 128;
modelConfigs.mean_fr.hiddenLayer2 = 64;
modelConfigs.mean_fr.dropoutRate  = 0.01;
modelConfigs.mean_fr.mcIterations = 10;
modelConfigs.mean_fr.initialEpochs= 80;
modelConfigs.mean_fr.refitEpochs  = 40;

modelConfigs.wind_speed.windowSize   = 48;
modelConfigs.wind_speed.hiddenLayer1 = 128;
modelConfigs.wind_speed.hiddenLayer2 = 16;
modelConfigs.wind_speed.dropoutRate  = 0.08;
modelConfigs.wind_speed.mcIterations = 10;
modelConfigs.wind_speed.initialEpochs= 80;
modelConfigs.wind_speed.refitEpochs  = 40;

%% 2. Data Loading & Preprocessing
disp('Loading and preprocessing data...');
allData = struct();
for iYear = 1:length(years)
    yearVal = years(iYear);
    try
        tbl = readtable(file_path, 'Sheet', num2str(yearVal));
    catch ME
        error('Error reading sheet %s: %s', num2str(yearVal), ME.message);
    end
    for p = 1:length(parameters)
        pName = parameters{p};
        if ismember(pName, tbl.Properties.VariableNames)
            tmp = tbl.(pName);
            tmp = fillmissing(tmp,'linear');
        else
            tmp = nan(height(tbl),1);
        end
        allData(iYear).(pName) = tmp;
    end
end

trainData   = struct();
testData    = struct();
normParams  = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    concatTrain = [];
    for y = training_years
        idxYear = find(years == y, 1);
        concatTrain = [concatTrain; allData(idxYear).(pName)];
    end
    concatTest = [];
    for y = testing_years
        idxYear = find(years == y, 1);
        concatTest = [concatTest; allData(idxYear).(pName)];
    end
    
    [trainNorm, ps] = mapminmax(concatTrain', 0, 1);
    trainNorm = trainNorm';
    testNorm  = mapminmax('apply', concatTest', ps);
    testNorm  = testNorm';
    
    trainData.(pName) = trainNorm;
    testData.(pName)  = testNorm;
    normParams.(pName)= ps;
end

%% 3. Initial BNN Training (Per Parameter)
disp('Initial BNN training for each parameter...');
models = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    cfg   = modelConfigs.(pName);
    [X_train, Y_train] = createANNData(trainData.(pName), cfg.windowSize);
    
    layers = [
        featureInputLayer(cfg.windowSize, 'Name','input')
        fullyConnectedLayer(cfg.hiddenLayer1, 'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(cfg.dropoutRate, 'Name','dropout1')
        fullyConnectedLayer(cfg.hiddenLayer2, 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(1, 'Name','fc3')
        regressionLayer('Name','output')
    ];
    
    opts = trainingOptions('adam', ...
        'MaxEpochs', cfg.initialEpochs, ...
        'MiniBatchSize', 64, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'Plots','none', ...
        'ExecutionEnvironment','parallel'); % <--- use parallel CPU workers
    
    bnn_net = trainNetwork(X_train, Y_train, layers, opts);
    
    models.(pName).bnn = bnn_net;
    models.(pName).cfg = cfg;
    models.(pName).trainingSeries = trainData.(pName);
end

%% 4. Rolling 5-Day Forecast + Refit
disp('Rolling 5-day forecast + BNN refit...');
forecasts = struct();
residuals = struct();
metrics   = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    nTest_p = length(testData.(pName));
    forecasts.(pName) = nan(nTest_p,1);
    residuals.(pName) = nan(nTest_p,1);
    metrics.(pName)   = struct('MSE',[],'MAE',[],'RMSE',[],'R2',[]);
end

nTest = length(testData.(parameters{1}));
numBlocks = ceil(nTest / hours_per_refit);

for blockIdx = 1:numBlocks
    fprintf('\n=== 5-Day Block %d of %d ===\n', blockIdx, numBlocks);
    startIdx = (blockIdx-1)*hours_per_refit + 1;
    endIdx   = min(blockIdx*hours_per_refit, nTest);
    
    for p = 1:length(parameters)
        pName = parameters{p};
        cfg   = models.(pName).cfg;
        net   = models.(pName).bnn;
        currTrainSeries = models.(pName).trainingSeries;
        nTest_p = length(testData.(pName));
        if startIdx > nTest_p, continue; end
        
        endIdx_p = min(endIdx, nTest_p);
        blockLen = endIdx_p - startIdx + 1;
        if blockLen < 1, continue; end
        
        if length(currTrainSeries) < cfg.windowSize
            continue;
        end
        
        initWindow = currTrainSeries(end-cfg.windowSize+1 : end);
        mcPredictions = zeros(blockLen, cfg.mcIterations);
        
        parfor mc = 1:cfg.mcIterations
            mcPredictions(:,mc) = multiStepForecastDropout(net, initWindow, blockLen, cfg.dropoutRate);
        end
        YF_block = mean(mcPredictions, 2, 'omitnan');
        YF_block = max(min(YF_block, 1), 0);
        
        forecasts.(pName)(startIdx:endIdx_p) = YF_block;
        Y_actual_block = testData.(pName)(startIdx:endIdx_p);
        residuals.(pName)(startIdx:endIdx_p) = Y_actual_block - YF_block;
        
        mse_  = mean((Y_actual_block - YF_block).^2, 'omitnan');
        mae_  = mean(abs(Y_actual_block - YF_block), 'omitnan');
        rmse_ = sqrt(mse_);
        ss_res= sum((Y_actual_block - YF_block).^2, 'omitnan');
        ss_tot= sum((Y_actual_block - mean(Y_actual_block,'omitnan')).^2, 'omitnan');
        if ss_tot>0
            r2_ = 1 - (ss_res/ss_tot);
        else
            r2_ = double(ss_res==0);
        end
        
        metrics.(pName).MSE  = [metrics.(pName).MSE;  mse_];
        metrics.(pName).MAE  = [metrics.(pName).MAE;  mae_];
        metrics.(pName).RMSE = [metrics.(pName).RMSE; rmse_];
        metrics.(pName).R2   = [metrics.(pName).R2;   r2_];
        
        updatedSeries = [currTrainSeries; Y_actual_block];
        if length(updatedSeries) > rolling_window_size
            updatedSeries(1:end-rolling_window_size) = [];
        end
        models.(pName).trainingSeries = updatedSeries;
        
        if length(updatedSeries) >= cfg.windowSize
            [X_upd, Y_upd] = createANNData(updatedSeries, cfg.windowSize);
            refitLayers = net.Layers;
            refitOpts = trainingOptions('adam', ...
                'MaxEpochs', cfg.refitEpochs, ...
                'MiniBatchSize', 64, ...
                'Shuffle','every-epoch', ...
                'Verbose',false, ...
                'Plots','none', ...
                'ExecutionEnvironment','parallel'); % <--- also parallel
            try
                net_refit = trainNetwork(X_upd, Y_upd, refitLayers, refitOpts);
                models.(pName).bnn = net_refit;
            catch ME
                warning('BNN refit failed for %s in block %d: %s', pName, blockIdx, ME.message);
            end
        end
    end
end

%% 5. Final Evaluation & Plots
disp('Final evaluation and plots...');
for p = 1:length(parameters)
    pName = parameters{p};
    Y_actual_norm   = testData.(pName);
    Y_forecast_norm = forecasts.(pName);
    nTest_p = length(Y_actual_norm);
    Y_forecast_norm = Y_forecast_norm(1:nTest_p);
    
    mse_  = mean((Y_actual_norm - Y_forecast_norm).^2, 'omitnan');
    mae_  = mean(abs(Y_actual_norm - Y_forecast_norm), 'omitnan');
    rmse_ = sqrt(mse_);
    ss_res= sum((Y_actual_norm - Y_forecast_norm).^2, 'omitnan');
    ss_tot= sum((Y_actual_norm - mean(Y_actual_norm,'omitnan')).^2, 'omitnan');
    if ss_tot>0
        r2_ = 1 - (ss_res / ss_tot);
    else
        r2_ = double(ss_res==0);
    end
    
    fprintf('\n=== Final Results for %s ===\n', pName);
    fprintf('MSE  = %.4f\n',  mse_);
    fprintf('MAE  = %.4f\n',  mae_);
    fprintf('RMSE = %.4f\n',  rmse_);
    fprintf('R^2  = %.4f\n',  r2_);
    
    ps = normParams.(pName);
    Y_actual_denorm   = mapminmax('reverse', Y_actual_norm', ps)';
    Y_forecast_denorm = mapminmax('reverse', Y_forecast_norm', ps)';
    Y_forecast_denorm(Y_forecast_denorm<0) = 0;
    
    figure('Name',['Scatter: ',pName],'NumberTitle','off');
    scatter(Y_actual_denorm, Y_forecast_denorm, 10, 'filled'); hold on;
    minVal = min([Y_actual_denorm; Y_forecast_denorm]);
    maxVal = max([Y_actual_denorm; Y_forecast_denorm]);
    plot([minVal,maxVal],[minVal,maxVal],'r--','LineWidth',2);
    title(['Scatter: Actual vs Forecast (',pName,')']);
    xlabel('Actual'); ylabel('Forecast');
    grid on; hold off;
    
    figure('Name',['TimeSeries: ',pName],'NumberTitle','off');
    plot(Y_actual_denorm,'b','LineWidth',1.5); hold on;
    plot(Y_forecast_denorm,'r--','LineWidth',1.5);
    legend('Actual','Forecast','Location','best');
    title(['Actual vs Forecast (',pName,')']);
    xlabel('Hour'); ylabel(pName);
    grid on; hold off;
    
    figure('Name',['Residuals: ',pName],'NumberTitle','off');
    plot(Y_actual_denorm - Y_forecast_denorm,'k');
    xlabel('Hour'); ylabel('Residual');
    title(['Residuals (',pName,')']);
    grid on; hold off;
end

disp('BNN MCD forecasting completed.');

%% ---------------- Helper Functions ----------------
function [X, Y] = createANNData(dataSeries, windowSize)
    n = length(dataSeries);
    numSamples = n - windowSize;
    if numSamples < 1
        error('Not enough data. n=%d, windowSize=%d', n, windowSize);
    end
    X = zeros(numSamples, windowSize);
    Y = zeros(numSamples, 1);
    for i = 1:numSamples
        X(i, :) = dataSeries(i : i+windowSize-1);
        Y(i)   = dataSeries(i+windowSize);
    end
end

function YF = multiStepForecastDropout(net, initWindow, steps, dropoutRate)
    windowSize = length(initWindow);
    YF = zeros(steps,1);
    currentWindow = initWindow;
    for s = 1:steps
        predVal = forwardPassDropout(net, currentWindow, dropoutRate);
        predVal = max(predVal, 0);
        YF(s) = predVal;
        currentWindow = [currentWindow(2:end); predVal];
    end
end

function outVal = forwardPassDropout(net, inputVec, dropoutRate)
    fc1 = net.Layers(2);
    x = fc1.Weights * inputVec + fc1.Bias;
    x = max(x,0);
    dropMask = rand(size(x)) > dropoutRate;
    x = x .* dropMask;
    
    fc2 = net.Layers(5);
    x = fc2.Weights * x + fc2.Bias;
    x = max(x,0);
    
    fc3 = net.Layers(7);
    rawOut = fc3.Weights * x + fc3.Bias;
    outVal = rawOut;
end
