function [] = classif_multiClass(trainHistFile, trainLblFile, testHistFile, testLblFile, outDir, suffix)


classifResult = [];

nb_pat_limit_classif=1000000;

train_interKernFN=sprintf('%s/train_interKern%s.mat', outDir, suffix );
test_interKernFN=sprintf('%s/test_interKern%s.mat',outDir, suffix );
train_interSVMModel_FN=sprintf( '%s/svmModel_interKern_featScale%s', outDir, suffix );
classifResultFN=sprintf('%s/classifResult%s.csv', outDir, suffix );
bestCFN=sprintf('%s/bestC%s.csv', outDir, suffix );

currDir = strrep(mfilename('fullpath'),'classif_multiClass','');
addpath( currDir );
addpath( genpath( strcat( currDir , 'libsvm') ) ) ;

%pkg install -forge io;
pkg load io;

printf( 'Process Classification.\n'); fflush(stdout);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read train file
printf( 'Read train files : \n-%s \n', trainHistFile ); fflush(stdout);
%-------------------------------------------------------------------------------
% Train label
if ( ~exist( sprintf( '%s.mat', trainLblFile ) ) )
  train_label = dlmread( trainLblFile, ',');
  if ( train_label(1) == 0 )
    train_label = train_label+1;
  endif
  save( '-v7', sprintf( '%s.mat', trainLblFile ), 'train_label' );
else
  load( sprintf( '%s.mat', trainLblFile ) );
endif

%-------------------------------------------------------------------------------
% Train hist
if ( ~exist( sprintf( '%s.mat', trainHistFile ) ) )
  train_hist = dlmread( trainHistFile, ',');
  save( '-v7', sprintf( '%s.mat', trainHistFile ), 'train_hist' );
else
  load( sprintf( '%s.mat', trainHistFile ) );
endif
% Check last col to be sure it is not fill by 0 in case last col of csv is class.
if ( sum( train_hist (:, size( train_hist, 2 ) ) ) == 0 )
  train_hist = train_hist(:,1:end-1);
endif
minColTrain = min( train_hist );
maxColTrain = max( train_hist );
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train hist manipulation
%-------------------------------------------------------------------------------
%-------------------------------------------------------------------------------
% Train intersection kernel
printf( 'Compute normalize train kernel.\n' ); fflush(stdout);
if exist( train_interKernFN )
  load( train_interKernFN );
else
  % feature scale normalisation
  printf( 'Compute normalize train hist w/ feature scale.\n' ); fflush(stdout);
  train_hist_normalized  = ( train_hist .- minColTrain ) ./ ( maxColTrain - minColTrain );
  train_Inter = hist_isect( train_hist_normalized, train_hist_normalized  );
  save( '-v7', train_interKernFN, 'train_Inter');
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read test file
printf( 'Read test files : \n-%s \n', testHistFile ); fflush(stdout);
%-------------------------------------------------------------------------------
% Test label
if ( ~exist( sprintf( '%s.mat', testLblFile ) ) )
  test_label = dlmread( testLblFile, ',');
  if ( test_label(1) == 0 )
    test_label = test_label+1;
  endif
  save( '-v7', sprintf( '%s.mat', testLblFile ), 'test_label' );
else
  load( sprintf( '%s.mat', testLblFile ) );
endif

%-------------------------------------------------------------------------------
% Test hist
if ( ~exist( sprintf( '%s.mat', testHistFile ) ) )
  test_hist = dlmread( testHistFile, ',');
  save( '-v7', sprintf( '%s.mat', testHistFile ), 'test_hist' );
else
  load( sprintf( '%s.mat', testHistFile ) );
endif

% Check last col to be sure it is not fill by 0 in case last col of csv is class.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test hist manipulation
%-------------------------------------------------------------------------------
% feature scale normalisation
%-------------------------------------------------------------------------------
% Test intersection kernel
printf( 'Compute normalize test kernel.\n' ); fflush(stdout);
% Compute Test BoWBoG Inter Kernel
if exist( test_interKernFN )
  load( test_interKernFN );
else
  % feature scale normalisation
  printf( 'Compute normalize test hist w/ feature scale.\n' ); fflush(stdout);
  test_hist_normalized = ( test_hist .- minColTrain ) ./ ( maxColTrain - minColTrain );
  test_hist_normalized( find( test_hist_normalized > 1 ) ) = 1 ;
  if ~( exist("train_hist_normalized") )
    train_hist_normalized  = ( train_hist .- minColTrain ) ./ ( maxColTrain - minColTrain );
  endif
  test_Inter  = hist_isect( test_hist_normalized, train_hist_normalized);
  save( '-v7', test_interKernFN, 'test_Inter');
end

clear test_hist
clear train_hist

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test hist file verification
if ( size(train_Inter,1) == size(test_Inter,1) )
  if ( train_Inter == test_Inter )
    printf( 'ERROR : Train inter kernel and test inter kernel are the same !!!!\n');
    printf( 'ERROR : Please, input check files %s, %s.\n', trainHistFile, testHistFile );
    fflush(stdout);
    return
  endif
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data computation for SVM
%-------------------------------------------------------------------------------
% Train index for SVM
index = (1:size(train_Inter,1))';
train_Inter = [ index train_Inter ];
%-------------------------------------------------------------------------------
% Test index for SVM
index = (1:size(test_Inter,1))';
test_Inter = [ index test_Inter ];
%-------------------------------------------------------------------------------
% Check number of class and class ID
class = unique(train_label);
nbClass = numel(class);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training of n-SVM (one per class)
%for idxC = 0:size( train_class, 1 )-1
printf( 'Training Model.\n' );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTER KERN
%-----------------------------------------------------------------------------
% Load svmModel or find best param C and train a SVM model
for idxC = 1:nbClass
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % BoWBoG
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  disp(sprintf('Class: %d', idxC ) );
  fflush(stdout);
  currtrain_interSVMModel_FN=sprintf('%s-Class%d.mat', train_interSVMModel_FN, idxC );

  if ( exist( sprintf( '%s', currtrain_interSVMModel_FN ) ) )
    load ( sprintf( '%s', currtrain_interSVMModel_FN ) );
  else
    % Find best metaparameter C for currecnt class
    disp(sprintf('Search of C on for class %d', idxC ) );
    fflush(stdout);
    bestC = 0;
    bestVal = 0;
    %-------------------------------------------------------------------------
    % Train SVM with probability as result with multiple value of C
    for k = -10:1:10
      disp(sprintf('c=%g', 2^k ) );
      fflush(stdout);
      cmd = ['-v 10 -t 4 -q  -c ', num2str(2^k)];
      val = svmtrain( double(train_label==idxC), train_Inter, cmd );
      if ( val > bestVal )
        bestVal = val;
        bestC = 2^k;
      end
    end
    % Train SVM on BoW with probability estimation
    printf('C = %d \n' , bestC );
    fflush(stdout);
  
    cmd = ['-t 4 -q -b 1 -c ', num2str(bestC) ];
    fid = fopen( bestCFN,'w' );
    fprintf( fid, 'bestC, %f \n', bestC );
    fclose( fid );
    %---------------------------------------------------------------------------
    % Train SVM with bestC with probability estimation and save this model
    svmModel = svmtrain( double(train_label==idxC), train_Inter, cmd );
    save( '-v7', sprintf( '%s', currtrain_interSVMModel_FN ), 'svmModel' );
  end

  % Predict SVM on Train
  [Lbl,Acc,Prev] = svmpredict( double(train_label==idxC), train_Inter, svmModel, sprintf('-q -b 1'));
  allBoWBoGProbTrain(:,idxC) = Prev(:,svmModel.Label==1);    %# probability of class==idxC+1

  % Predict SVM on Test
  [Lbl,Acc,Prev] = svmpredict( double(test_label==idxC), test_Inter, svmModel, sprintf('-q -b 1'));
  allBoWBoGProbTest(:,idxC) = Prev(:,svmModel.Label==1);    %# probability of class==idxC+1
end

% Compute result with NSVM voting on BoWBoG Train
[~,predBoWBoGTrain]       = max(allBoWBoGProbTrain,[],2);
classifResult(1) = sum( predBoWBoGTrain == train_label ) ./ numel( train_label ) * 100 ;
% Compute result with NSVM voting on BoWBoG Test
[~,predBoWBoGTest]       = max(allBoWBoGProbTest,[],2);
classifResult(2) = sum( predBoWBoGTest == test_label ) ./ numel( test_label ) * 100 ;


 
%    clear confus
%    for idxC = 1:nbClass
%      for idxC2 = 1:nbClass
%        confus(idxC,idxC2)=sum( predBoWBoGTest( 
%              1+(idxC-1)*(size(test_label,1)/nbClass):
%              idxC*(size(test_label,1)/nbClass) ) == idxC2 );
%      endfor
%    endfor
%    transpose(confus/5)

disp( sprintf( '%2.2f, %2.2f \n', classifResult(:) ) );
disp( sprintf( '%s------------------------\n', classifResultFN ) );
fflush(stdout);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save prediction results
fid = fopen( classifResultFN,'w' );
fprintf( fid, 'train, test \n' );
fprintf( fid, '%2.2f, %2.2f \n', classifResult(:) );
fclose( fid );

endfunction
