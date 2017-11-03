%%data prep
data_file = csvread('C:/Users/Ben Bowes/Documents/HRSD GIS/Site Data/MMPS_170_no_blanks.csv', 1,1);
HRRR_data = csvread('C:/HRRR/HRRR_Archive_20160920_hr_6/rainfall.csv', 1,1);
SewellsPt_data = csvread('C:/Users/Ben Bowes/Documents/HRSD GIS/Tide_hr_Sewells_Pt/Tide_20160920_hr02.csv', 1,1);

gwl = data_file(:,1);
tide = data_file(:,2);
precip = data_file(:,3);
forecast_precip = HRRR_data(:,6);
forecast_tide = SewellsPt_data(:,1);

%%training data, 70% of time series
gwl_train = gwl(1:round(length(gwl)*0.7),:);
tide_train = tide(1:round(length(tide)*0.7),:);
precip_train = precip(1:round(length(precip)*0.7),:);

%%construct training inputs and targets, put in NN format
inputSeriesTrn = [tide_train,precip_train];
targetSeriesTrn = gwl_train;

inputSeriesTrn = tonndata(inputSeriesTrn, false, false);
targetSeriesTrn = tonndata(targetSeriesTrn, false, false);

%%set up network architecture
Ntrials = 10; %number of neural networks to create
nets = cell(Ntrials,1); 

trainFcn = 'trainlm';
input_delay = 1:72; %delay for exogenous inputs
feedback_delay = 1:72; %delay for targets
Hmin = 1; %min number of neurons
Hmax = 10; %max number of neurons
dH = 1;%increment for number of neurons

%%define network
perfs_train = cell(Ntrials,Hmax);%cell array to store training performance
nets_list = cell(Ntrials,Hmax);%cell array to store all networks
for h = Hmin:dH:Hmax
    net = narxnet(input_delay,feedback_delay,h,'open',trainFcn);
    net.divideFcn = 'divideblock';
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    %net.divideMode = 'none';  % training data is not split into trn/val/tst

    %%train network
    [Xs,Xi,Ai,Ts] = preparets(net,inputSeriesTrn,{},targetSeriesTrn);

    for i = 1:Ntrials
        fprintf('Training %d/%d\n',i,Ntrials)
        nets{i} = train(net,Xs,Ts,Xi,Ai);
        nets_list{i,h} = nets{i};
        Y=nets{i}(Xs,Xi,Ai);
        perfs_train{i,h} = mse(nets{i}, Ts, Y);
    end
end
view(net);

%%Prepare test and evaluation data
N = 18; %number of steps for multi step prediction
shift = 1222;

%%testing data, 30% of time series
gwl_test = gwl(length(gwl_train)+1:end-shift,:);
tide_test = tide(length(tide_train)+1:end-shift,:);
precip_test = precip(length(precip_train)+1:end-shift,:);

%%prediction evaluation data
gwl_eval = gwl(end-shift+1:end-shift+N);
tide_eval = tide(end-shift+1:end-shift+N);
precip_eval = precip(end-shift+1:end-shift+N);

%%construct test inputs and targets, put in NN format
inputSeriesTest = [tide_test,precip_test];
targetSeriesTest = gwl_test;

inputSeriesTest = tonndata(inputSeriesTest, false, false);
targetSeriesTest = tonndata(targetSeriesTest, false, false);

%%construct evaluation inputs and targets, put in NN format
% inputSeriesEval = [tide_eval,precip_eval];
% targetSeriesEval = gwl_eval;
% 
% inputSeriesEval = tonndata(inputSeriesEval, false, false);
% targetSeriesEval = tonndata(targetSeriesEval, false, false);

%%construct forecasted inputs and targets, put in NN format
inputSeriesEval = [forecast_tide,forecast_precip];
targetSeriesEval = gwl_eval;

inputSeriesEval = tonndata(inputSeriesEval, false, false);
targetSeriesEval = tonndata(targetSeriesEval, false, false);

%%test the network
[Xs_t,Xi_t,Ai_t,Ts_t] = preparets(net,inputSeriesTest,{},targetSeriesTest);

perfs_test = cell(Ntrials,Hmax);
YTotal = cell(0);
for i = 1:Ntrials
    for h = 1:Hmax
      neti = nets_list{i,h};
      Y = neti(Xs_t,Xi_t,Ai_t);
      perfs_test{i,h} = mse(neti, Ts_t, Y);
    end
%   YTotal = [YTotal; Y];
end

%%multi-step ahead prediction
[Xs_e,Xio_e,Aio_e,Ts_e] = preparets(net,inputSeriesTest(1:end-input_delay),{},targetSeriesTest(1:end-feedback_delay));

perfs_eval = cell(Ntrials,Hmax);
YPred = cell(0,18);
for i = 1:Ntrials
    for h = 1:Hmax
      neti_eval = nets_list{i,h};
      [Y_eval,Xfo,Afo] = neti_eval(Xs_e,Xio_e,Aio_e);
      [netc,Xic,Aic] = closeloop(neti_eval,Xfo,Afo);
      [yc,Xfc,Afc] = netc(inputSeriesEval,Xic,Aic);
      perfs_eval{i,h} = mse(netc, targetSeriesEval, yc);
      YPred = [YPred; yc];
    end
end
YPred = cell2mat(YPred);
YPred = transpose(YPred);

%%save specific net
%MMPS_043_nets = nets_list{1,1};
%MMPS_153_nets = nets_list;
% save ('MMPS_125_nets.mat');