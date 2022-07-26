clear all;
cd ..
setpaths3
cd demos


%% load data
try  
    load bal.mat
catch  % if it fails, download it from the web
    disp('Downloading data ...');
    urlwrite('https://dl.dropboxusercontent.com/u/4284723/DATA/bal.mat','bal.mat');
    load bal.mat
end;


%% tune parameters
disp('Setting hyper parameters');
K=25;
knn=5;
maxiter=8;

%% train full muodel
fprintf('Training final model...\n');
[L,Details] = lmnnCG(xTr, yTr,K,'maxiter',maxiter);

testerrEUC=knncl([],xTr,yTr,xTe,yTe,knn,'train',0);
testerrLMNN=knncl(L,xTr,yTr,xTe,yTe,knn,'train',0);
fprintf('Bal data set\n');
fprintf('\n\nTesting error before LMNN: %2.2f%%\n',100.*testerrEUC);
fprintf('Testing error after  LMNN: %2.2f%%\n',100.*testerrLMNN);



