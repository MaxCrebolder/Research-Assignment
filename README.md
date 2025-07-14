**Improving Forecast Accuracy for Offshore Wind Turbine Installation**

The scripts made in this GitHub file are used to model three separate models, namely a Hybrid ARIMA-ANN model (hybrid_final_model), a Bayesian Neural Network with Monte Carlo Dropout (BNN_with_MCD_final_model) and a Long-Short Term Memory Model (final_lstm_model).
Other scripts in the GitHub are used during the hyper-parameter tuning phase of the model.

**Requirements**

MATLAB R2024a or newer with:
- Statistics & Machine Learning Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox

**Data**

Each file loads a single Excel file in which all metocean measurements, of site 15, are combined in 10 different sheets. So sheet 1 will be 2001, sheet 2 the measurements from 2002 and so on. In this case only the Significant Wave Height (s_wht), the Mean Wave Frequency (mean_fr) and the Wind Speed (wind_speed), are used from the data. Make sure all yearly data is combined in one file!

**Hybrid ARIMA-ANN Model**
- ARIMA order tuning, Grid-search of (p,d,q) orders to find the best-AIC model orders.
- Hybrid Neron Sweep and Learning, Hidden-layer sweep for the ANN component of the hybrid model.
- hyrbid_final_model, Train (2001-2008) + test (2009-2010) the Hybrid ARIMA–ANN on with a 5-day rolling refit.

**BNN with MCD**
- BNN_hyperparameter_tuning, 540-run sweep over window size, neuron count, dropout-probability.
- BNN_mc_iterations_batchsize, Sensitivity of MC iterations and batch size on inference time/accuracy.
- BNN_with_MCD_final_model, Train (2001-2008) + test (2009-2010) the BNN with MCD on with a 5-day rolling refit.


**LSTM**
- lstm_hyperparameters_tuning, Sequence length & hidden units sweep for the LSTM.
- final_lstm_model, Train (2001-2008) + test (2009-2010) the LSTM on with a 5-day rolling refit.

**Important Notes**
Due to the large amount of inputted data the final models will take a significant time to run (>12-hours).
