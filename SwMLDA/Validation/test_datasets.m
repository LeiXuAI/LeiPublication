% This is a main function for testing different feature extraction methods
% and different classifiers

% clc;
% clear all;
% close all;
mydir = pwd;
idcs = strfind(mydir, '/');
newdir = mydir(1:idcs(end)-1);

addpath([newdir,'/Datasets']);
% path(path,'../Datasets');
addpath('../Measures');
addpath('../Classifiers');
addpath('../Transforms');
addpath('../Results');
addpath('../SwLDA');
 
train_cell     = cell(1,50);
test_cell      = cell(1,50);
result_cell    = cell(1,50);
transfrom_cell = cell(1,20);
classifier_cell= cell(1,10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% here to add or replace training, testing data files
%% and their correspondingresult files.
% most better than refs
train_cell {1,1}  = '../Datasets/Yeast_train.mat';
test_cell  {1,1}  = '../Datasets/Yeast_test.mat';
result_cell{1,1}  = '../Results/Yeast';
% 
% % parts better than refs
% train_cell {1,2}  = '../Datasets/Emotions_train.mat';
% test_cell  {1,2}  = '../Datasets/Emotions_test.mat';
% result_cell{1,2}  = '../Results/Emotions';

% far more better than refs
train_cell {1, 2} = '../Datasets/Scene_train.mat';
test_cell {1, 2} = '../Datasets/Scene_test.mat';
result_cell{1,2}  = '../Results/Scene';

% parts better than refs
train_cell {1, 3} = '../Datasets/Image_train.mat';
test_cell {1, 3} = '../Datasets/Image_test.mat';
result_cell{1,4}  = '../Results/Image';

% better than refs
train_cell {1, 4} = '../Datasets/Enron_train.mat';
test_cell {1, 4} = '../Datasets/Enron_test.mat';
result_cell{1,4}  = '../Results/Enron';

% better than refs
train_cell {1, 5} = '../Datasets/Cal500_train.mat';
test_cell {1, 5} = '../Datasets/Cal500_test.mat';
result_cell{1, 5}  = '../Results/Cal500';

% % worse than refs, spend more time
% train_cell {1, 7} = '../Datasets/Bibtex_train.mat';
% test_cell {1, 7} = '../Datasets/Bibtex_test.mat';
% result_cell{1, 7}  = '../Results/Bibtex';

% better than refs.
train_cell {1, 6} = '../Datasets/Corel16k001_train.mat';
test_cell {1, 6} = '../Datasets/Corel16k001_test.mat';
result_cell{1, 6}  = '../Results/Corel16k001';

% far more better than refs
train_cell {1, 7} = '../Datasets/Plant_train.mat';
train_cell {1, 7} = '../Datasets/Plant_test.mat';
test_cell {1, 7}  = '../Results/Plant';

% some parts better than refs
train_cell {1, 8} = '../Datasets/Human_train.mat';
test_cell {1, 8} = '../Datasets/Human_test.mat';
result_cell{1, 8}  = '../Results/Human';

% some parts better than refs
train_cell {1, 9} = '../Datasets/TMC2007_train.mat';
test_cell {1, 9} = '../Datasets/TMC2007_test.mat';
result_cell{1, 9}  = '../Results/TMC2007';


% % some of my methods far more better than refs
train_cell {1, 10} = '../Datasets/Medical_train.mat';
test_cell {1, 10} = '../Datasets/Medical_test.mat';
result_cell{1, 10}  = '../Results/Medical';
% 
% % dimensions of dataset are not consistent  
% train_cell {1, 13} = '../Datasets/Genbase_train.mat';
% test_cell {1, 13} = '../Datasets/Genbase_test.mat';
% result_cell{1, 13}  = '../Results/Genbase';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% here to list compared linear feature extraction methods
%other methods
transform_cell{1,1}  = '-PCA';
transform_cell{1,2}  = '-CCA';
transform_cell{1,3}  = '-MLSI';
transform_cell{1,4}  = '-MDDMp';
transform_cell{1,5}  = '-MDDMf';
transform_cell{1,6}  = '-MVMD';

%referred methods
transform_cell{1,7}  = '-DMLDA';
transform_cell{1,8}  = '-wMLDAc(MLDA)'; %Wang2010
transform_cell{1,9}  = '-wMLDAb';       %Park2007
transform_cell{1,10} = '-wMLDAe';       %Xu2017 & Chen2007
transform_cell{1,11} = '-wMLDAf';       %Xu2017 & Lin2013
transform_cell{1,12} = '-wMLDAd';       %Xu2017

%our methods
transform_cell{1,13} = '-wMLDAs_m';       %Xu2017 & myWork with missclassification V
transform_cell{1,14} = '-wMLDAs_c';      %Xu2017 & myWork with correlation V
transform_cell{1,15} = '-wMLDAs_f';       %Xu2017 & myWork with fuzzy V
transform_cell{1,16} = '-wMLDAs_e';      %Xu2017 & myWork with entropy V
transform_cell{1,17} = '-wMLDAs_b';       %Xu2017 & myWork with binary V
transform_cell{1,18} = '-wMLDAs_d';      %Xu2017 & myWork with dependence V
transform_cell{1,19} = '-wMLDAs_k';      %Xu2017 & myWork with kernel
% transform_type: to detect transform type
%    =0, no tranaform
%    =(1,2,...,12), to select one of the above transforms
% default setting
% transform_type = 8; %% to be selected by user
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%transform_parameter settings for transform
%       .rank: to determine whether to estimate the ranks of matrices 
%           =1 --> yes
%           =0 --> no (default=0)
%       .ratio -- a constant to detect the number of projected features (d)
%                                (i.e., reduced dimensions)
%            =-2 --> = the number of classes (d=q) (CCA, MDDMp and MDDMf)
%            =-1 --> = the number of classes - 1 (d=q-1) (LDA-type)
%            =0  --> = the number of all positive eigenvalues 
%                     (the eigenvalues > 10^(-10)*max(eigenvalues)
%            =(0,1) --> = ratio * the number of all features (d=ratio*D)
%            =(1,2) --> = the minimal d to satisfy 
%                              sum normalized eigenvalues >ratio-1
%            =1,>=2 --> = ratio (d=ratio)
%       .regX: regularization constant for X of size N*D
%              (default=0.1)(MLSI,CCA)
%       .regY: regularization constant for Y of size N*q
%              (default=0.1)(CCA)
%       .regXY: regularization constant for X and Y (default=0.1)(MLSI,LDA)
%       .beta: trade-off constant (default=0.5) (MLSI,MDDMf and MVMD)
%There are two ways to add a regulariztion constant to some matrix A
%%  (1) [A + reg] when reg >= 0
%%  (2) [A + |reg|*max(max(A))] when reg < 0
%---------------------------------------------------------------------------
%% default settings for transform
% transform_parameter.rank =0; % 
% transform_parameter.regX = 0.1; %
% transform_parameter.regY = 0.1; %
% transform_parameter.regXY = 0.1; %
% transform_parameter.beta = 0.5; % 
% transform_parameter.ratio = 1.999; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Here to list two classifiers used in experiments
classifier_cell{1,1} = '-MLkNN';
classifier_cell{1,2} = '-LRR';
% classifier_type: to choose a classifier
%%default setting
classifier_type = 1; %
%% classifier_parameter setting for classifier
% classifier_parameter.MLkNN_k: 
%       the number of nearest nighbour instances (default=15)
% classifier_parameter.RR_reg:
%       the regularization constant for ridge regression (default=0.01)
% default settings
% classifier_parametet.MLkNN_k =15;
% classifier_parameter.RidgeR_reg = 0.01;
%% 
%% here to determine user choice and parameters
% % KBS 
% % 
classifier_type = 1;
transform_type  = 13;
transform_parameter.ratio = 1.999;
transform_parameter.rank = 1;

% % % % Neurocomputing
% classifier_type = 1;
% transform_type  = 17;
% transform_parameter.ratio = -1;
% transform_parameter.rank = 1;

if (classifier_type == 1) %ML-kNN
    classifier_parameter.MLkNN_k  = 16;
else
    classifier_parameter.RidgeR_reg = 0.01;
end

switch(transform_type)
    case 1 %PCA

    case 2 %CCA
        %KBS
        transform_parameter.regY = 0.1;
        transform_parameter.regX = 0.1;
        
        %Neurocomputing
        %transform_parameter.regY = 0.0;
        %transform_parameter.regX = 0.0;
        
    case 3 %MLSI
        transform_parameter.beta = 0.5;
        transform_parameter.regXY = 0.1;
        transform_parameter.regX  = 0.1;
        
    case 4 %MDDMp
        
    case 5 %MDDMf
        %KBS
        transform_parameter.beta = 0.5;
        %Neurocomputing 
        %transform_parameter.beta = 1.0;
        
    case 6 %MVMD
        transform_parameter.beta  = 0.5;
        
    otherwise % wMLDA
        %KBS
        transform_parameter.regXY = 0.1;
        
        %Neurocomputing
        %transform_parameter.regXY = 0.0;
end

% to determine which data sets will be validated in the experiments
for i = [2]
    disp(train_cell{1,i});
    trainfile = train_cell{1,i};
    testfile = test_cell{1,i};
    if(transform_type == 0)
        resultfile = strcat(result_cell{1,i},classifier_cell{1, classifier_type},'.txt');
    else
        resultfile = strcat(result_cell{1,i},transform_cell{1,transform_type}, classifier_cell{1, classifier_type},'.txt');
    end
    
    % the main implementation function
    test_transform_classifier(trainfile, testfile, resultfile,transform_type,transform_parameter,classifier_type,classifier_parameter);

end