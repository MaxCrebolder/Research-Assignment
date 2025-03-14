clc; close all;

%% Configuration
file_path       = "C:\Users\maxcr\OneDrive\Documenten\Research assignment\Data\Site15_29m.xlsx";
parameters      = {'s_wht','mean_fr','wind_speed'};
years           = 2001:2010;
training_years  = 2001:2008;
testing_years   = 2009:2010;

hours_per_day   = 24;
hours_per_refit = 28 * hours_per_day;  
refitFrequency  = 1; 
rollingWindowHours = 150 * 24;  

modelConfigs = struct;
modelConfigs.s_wht.arimaOrder      = [2 0 3];
modelConfigs.s_wht.windowSize      = 149;
modelConfigs.s_wht.hiddenLayerSize = 5;
modelConfigs.s_wht.trainFcn        = 'trainlm';
modelConfigs.s_wht.maxEpochs       = 15;
modelConfigs.s_wht.goal            = 1e-6;

modelConfigs.mean_fr.arimaOrder      = [2 0 3];
modelConfigs.mean_fr.windowSize      = 149;
modelConfigs.mean_fr.hiddenLayerSize = 15;
modelConfigs.mean_fr.trainFcn        = 'trainlm';
modelConfigs.mean_fr.maxEpochs       = 30;
modelConfigs.mean_fr.goal            = 1e-6;

modelConfigs.wind_speed.arimaOrder      = [2 0 3];
modelConfigs.wind_speed.windowSize      = 48;
modelConfigs.wind_speed.hiddenLayerSize = 15;
modelConfigs.wind_speed.trainFcn        = 'trainlm';
modelConfigs.wind_speed.maxEpochs       = 25;
modelConfigs.wind_speed.goal            = 1e-6;

%% Load and Preprocess Data
disp('Loading and preprocessing data...');
allData = struct();
for iYear = 1:length(years)
    try
        tbl = readtable(file_path, 'Sheet', num2str(years(iYear)));
    catch ME
        error('Error reading sheet %d (%d): %s', iYear, years(iYear), ME.message);
    end
    for p = 1:length(parameters)
        pName = parameters{p};
        if ismember(pName, tbl.Properties.VariableNames)
            tempData = tbl.(pName);
            tempData = fillmissing(tempData, 'linear');
        else
            tempData = nan(height(tbl),1);
        end
        allData(iYear).(pName) = tempData;
    end
end

trainData   = struct();
testData    = struct();
normParams  = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    trainVals = [];
    for y = training_years
        idx = find(years == y);
        trainVals = [trainVals; allData(idx).(pName)];
    end
    testVals = [];
    for y = testing_years
        idx = find(years == y);
        testVals = [testVals; allData(idx).(pName)];
    end

    [trainNorm, ps] = mapminmax(trainVals', 0, 1);
    trainNorm = trainNorm';
    testNorm  = mapminmax('apply', testVals', ps);
    testNorm  = testNorm';

    trainData.(pName)   = trainNorm;
    testData.(pName)    = testNorm;
    normParams.(pName)  = ps;
end

%% Initial ARIMA + ANN Training
disp('Initial ARIMA + ANN training...');
tempModels = cell(1, length(parameters));
options = optimoptions('fmincon', 'Display','off', 'MaxIterations',100, 'MaxFunctionEvaluations',200);

parfor p = 1:length(parameters)
    pName = parameters{p};
    arima_fit = [];
    try
        arimaOrder = modelConfigs.(pName).arimaOrder;
        if arimaOrder(1) > 0, ARLags = 1:arimaOrder(1); else, ARLags = []; end
        if arimaOrder(3) > 0, MALags = 1:arimaOrder(3); else, MALags = []; end
        arima_spec = arima('Constant', NaN, 'ARLags', ARLags, ...
                           'D', arimaOrder(2), 'MALags', MALags);
        arima_fit = estimate(arima_spec, trainData.(pName), 'Display','off','Options', options);
    catch ME
        if strcmp(pName, 's_wht')
            fallbackOrder = [2 0 1];
            try
                if fallbackOrder(1)>0, ARLags = 1:fallbackOrder(1); else, ARLags=[]; end
                if fallbackOrder(3)>0, MALags = 1:fallbackOrder(3); else, MALags=[]; end
                fallback_spec = arima('Constant', NaN, 'ARLags',ARLags, ...
                                      'D',fallbackOrder(2), 'MALags',MALags);
                arima_fit = estimate(fallback_spec, trainData.(pName), 'Display','off','Options', options);
            catch
                arima_fit = [];
            end
        else
            arima_fit = [];
        end
    end

    windowSize     = modelConfigs.(pName).windowSize;
    hiddenLayerSz  = modelConfigs.(pName).hiddenLayerSize;
    trainFcn       = modelConfigs.(pName).trainFcn;
    maxEpochs      = modelConfigs.(pName).maxEpochs;
    goalVal        = modelConfigs.(pName).goal;
    [Xann, Yann]   = createANNData(trainData.(pName), windowSize);

    net = feedforwardnet(hiddenLayerSz, trainFcn);
    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = maxEpochs;
    net.trainParam.goal   = goalVal;
    net = train(net, Xann', Yann');

    tempModels{p} = struct('arima', arima_fit, ...
                           'ann', net, ...
                           'window', windowSize, ...
                           'trainingData', trainData.(pName), ...
                           'cumulativeErrors', struct('ARIMA',1, 'ANN',1));
end

models = cell2struct(tempModels, parameters, 2);

%% Rolling Forecast & Re-Fit
disp('Starting rolling forecast/refit...');
forecasts = struct(); 
residuals = struct(); 
metrics   = struct();

for p = 1:length(parameters)
    pName = parameters{p};
    nTest = length(testData.(pName));
    forecasts.(pName) = nan(nTest,1);
    residuals.(pName) = nan(nTest,1);
    metrics.(pName).MSE  = [];
    metrics.(pName).MAE  = [];
    metrics.(pName).RMSE = [];
    metrics.(pName).R2   = [];
end

numTestSamples = length(testData.(parameters{1}));
numBlocks = ceil(numTestSamples / hours_per_refit);

for blockIdx = 1:numBlocks
    fprintf('\n--- Block %d of %d ---\n', blockIdx, numBlocks);
    startIdx = (blockIdx-1)*hours_per_refit + 1;
    endIdx   = min(blockIdx*hours_per_refit, numTestSamples);

    blockResults = cell(1, length(parameters));
    parfor pp = 1:length(parameters)
        pName = parameters{pp};
        nTest_p = length(testData.(pName));
        if startIdx > nTest_p
            blockResults{pp} = [];
            continue;
        end
        endIdx_p = min(endIdx, nTest_p);
        blockLen = endIdx_p - startIdx + 1;
        localModel   = models.(pName);
        currentTrain = localModel.trainingData;

        arimaForecast = nan(blockLen,1);
        if ~isempty(localModel.arima)
            try
                [arimaForecast, ~] = forecast(localModel.arima, blockLen, 'Y0', currentTrain);
            catch
                arimaForecast = nan(blockLen,1);
            end
        end

        annForecast = nan(blockLen,1);
        if ~isempty(localModel.ann) && (length(currentTrain) >= localModel.window)
            ann_input = currentTrain(end - localModel.window + 1 : end);
            for h = 1:blockLen
                pred = localModel.ann(ann_input);
                if pred < 0, pred = 0; end
                annForecast(h) = pred;
                ann_input = [ann_input(2:end); pred];
            end
        end

        cumErr_ARIMA = localModel.cumulativeErrors.ARIMA;
        cumErr_ANN   = localModel.cumulativeErrors.ANN;
        sumErr = cumErr_ARIMA + cumErr_ANN;
        w_arima = cumErr_ANN / sumErr;
        w_ann   = cumErr_ARIMA / sumErr;

        YF_hybrid = nan(blockLen,1);
        for i = 1:blockLen
            if isnan(arimaForecast(i))
                YF_hybrid(i) = annForecast(i);
            elseif isnan(annForecast(i))
                YF_hybrid(i) = arimaForecast(i);
            else
                YF_hybrid(i) = w_arima*arimaForecast(i) + w_ann*annForecast(i);
            end
        end
        YF_hybrid(YF_hybrid<0) = 0;

        Y_actual = testData.(pName)(startIdx:endIdx_p);
        err_arima = mean((Y_actual - arimaForecast).^2, 'omitnan');
        err_ann   = mean((Y_actual - annForecast).^2, 'omitnan');
        alpha = 0.1;
        localModel.cumulativeErrors.ARIMA = (1-alpha)*cumErr_ARIMA + alpha*err_arima;
        localModel.cumulativeErrors.ANN   = (1-alpha)*cumErr_ANN   + alpha*err_ann;

        mse_  = mean((Y_actual - YF_hybrid).^2, 'omitnan');
        mae_  = mean(abs(Y_actual - YF_hybrid), 'omitnan');
        rmse_ = sqrt(mse_);
        ss_res= sum((Y_actual - YF_hybrid).^2, 'omitnan');
        ss_tot= sum((Y_actual - mean(Y_actual,'omitnan')).^2, 'omitnan');
        r2_   = 1 - (ss_res/ss_tot);

        blockResults{pp} = struct('forecast', YF_hybrid, 'actual', Y_actual, ...
                                  'model', localModel, ...
                                  'metrics', struct('MSE',mse_,'MAE',mae_,'RMSE',rmse_,'R2',r2_));
    end

    for pp = 1:length(parameters)
        pName = parameters{pp};
        if ~isempty(blockResults{pp})
            out = blockResults{pp};
            YF_block   = out.forecast;
            Y_actual_b = out.actual;
            localMdl   = out.model;
            m          = out.metrics;

            forecasts.(pName)(startIdx:endIdx) = YF_block;
            residuals.(pName)(startIdx:endIdx) = Y_actual_b - YF_block;
            metrics.(pName).MSE  = [metrics.(pName).MSE;  m.MSE];
            metrics.(pName).MAE  = [metrics.(pName).MAE;  m.MAE];
            metrics.(pName).RMSE = [metrics.(pName).RMSE; m.RMSE];
            metrics.(pName).R2   = [metrics.(pName).R2;   m.R2];
            models.(pName)       = localMdl;
        end
    end

    disp(['Refitting ARIMA + ANN after block ', num2str(blockIdx), '...']);
    for pp = 1:length(parameters)
        pName = parameters{pp};
        if startIdx <= length(testData.(pName))
            localMdl = models.(pName);
            newActual = testData.(pName)(startIdx:endIdx);
            localMdl.trainingData = [localMdl.trainingData; newActual];

            if length(localMdl.trainingData) > rollingWindowHours
                localMdl.trainingData = localMdl.trainingData(end-rollingWindowHours+1:end);
            end

            try
                arimaOrder = modelConfigs.(pName).arimaOrder;
                if arimaOrder(1) > 0, ARLags = 1:arimaOrder(1); else, ARLags=[]; end
                if arimaOrder(3) > 0, MALags = 1:arimaOrder(3); else, MALags=[]; end
                refitSpec = arima('Constant',NaN,'ARLags',ARLags,'D',arimaOrder(2),'MALags',MALags);
                localMdl.arima = estimate(refitSpec, localMdl.trainingData, 'Display','off','Options',options);
            catch
                % ARIMA re-fit failed, skipping
            end

            try
                wSize = modelConfigs.(pName).windowSize;
                if length(localMdl.trainingData) > wSize
                    [Xr, Yr] = createANNData(localMdl.trainingData, wSize);
                    netRefit = feedforwardnet(modelConfigs.(pName).hiddenLayerSize, ...
                                              modelConfigs.(pName).trainFcn);
                    netRefit.trainParam.showWindow      = false;
                    netRefit.trainParam.showCommandLine = false;
                    netRefit.trainParam.epochs = modelConfigs.(pName).maxEpochs;
                    netRefit.trainParam.goal   = modelConfigs.(pName).goal;
                    netRefit = train(netRefit, Xr', Yr');
                    localMdl.ann = netRefit;
                end
            catch
                % ANN re-train failed, skipping
            end
            models.(pName) = localMdl;
        end
    end
end

%% Final Evaluation & Plots
disp('Final evaluation and plots...');
forecast_s_wht      = [];
forecast_mean_fr    = [];
forecast_wind_speed = [];

for p = 1:length(parameters)
    pName = parameters{p};
    Y_actual_full   = testData.(pName);
    Y_forecast_full = forecasts.(pName);
    nTest_p = length(Y_actual_full);
    Y_forecast_full = Y_forecast_full(1:nTest_p);

    ps = normParams.(pName);
    Y_actual_denorm   = mapminmax('reverse', Y_actual_full', ps)';
    Y_forecast_denorm = mapminmax('reverse', Y_forecast_full', ps)';
    if strcmp(pName, 's_wht')
        maxTrain = ps.xmax;
        capValue = 1.5 * maxTrain;
        Y_forecast_denorm(Y_forecast_denorm>capValue) = capValue;
    end
    Y_forecast_denorm(Y_forecast_denorm<0) = 0;

    mse_  = mean((Y_actual_full - Y_forecast_full).^2, 'omitnan');
    mae_  = mean(abs(Y_actual_full - Y_forecast_full), 'omitnan');
    rmse_ = sqrt(mse_);
    ss_res= sum((Y_actual_full - Y_forecast_full).^2, 'omitnan');
    ss_tot= sum((Y_actual_full - mean(Y_actual_full,'omitnan')).^2, 'omitnan');
    r2_   = 1 - (ss_res / ss_tot);

    fprintf('\nResults for %s:\nMSE  = %.4f\nMAE  = %.4f\nRMSE = %.4f\nR^2  = %.4f\n', ...
        pName, mse_, mae_, rmse_, r2_);

    switch pName
        case 's_wht'
            forecast_s_wht = Y_forecast_denorm;
        case 'mean_fr'
            forecast_mean_fr = Y_forecast_denorm;
        case 'wind_speed'
            forecast_wind_speed = Y_forecast_denorm;
    end

    figure('Name',['Scatter ',pName],'NumberTitle','off');
    scatter(Y_actual_denorm, Y_forecast_denorm, 10, 'filled'); hold on;
    mnv = min([Y_actual_denorm;Y_forecast_denorm]);
    mxv = max([Y_actual_denorm;Y_forecast_denorm]);
    plot([mnv,mxv],[mnv,mxv],'r--','LineWidth',2);
    title(['Actual vs Forecast (',pName,')']);
    xlabel('Actual'); ylabel('Forecast'); grid on; hold off;

    figure('Name',['LinePlot ',pName],'NumberTitle','off');
    plot(Y_actual_denorm,'b','LineWidth',1.5); hold on;
    plot(Y_forecast_denorm,'r--','LineWidth',1.5);
    legend('Actual','Forecast');
    title(['Actual vs Forecast (',pName,')']);
    xlabel('Hour'); ylabel(pName); grid on; hold off;

    figure('Name',['Residuals ',pName],'NumberTitle','off');
    residDenorm = Y_actual_denorm - Y_forecast_denorm;
    plot(residDenorm,'k');
    title(['Residuals (',pName,')']);
    xlabel('Hour'); ylabel('Residual'); grid on; hold off;
end

disp('Hybrid ARIMA+ANN re-fitting completed.');
disp('Final forecast arrays: forecast_s_wht, forecast_mean_fr, forecast_wind_speed');

function [X, Y] = createANNData(series, windowSize)
    n = length(series);
    numSamples = n - windowSize;
    if numSamples < 1
        error('Not enough data (%d points) with windowSize=%d', n, windowSize);
    end
    X = zeros(numSamples, windowSize);
    Y = zeros(numSamples,1);
    for i = 1:numSamples
        X(i,:) = series(i : i+windowSize-1);
        Y(i)   = series(i+windowSize);
    end
end
