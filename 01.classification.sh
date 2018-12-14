#!/bin/bash

SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TIMEOUT=10
TIME_LIM=0
NBPAT_LIM_COUNT=100000000
NBPAT_LIM_COUNT_MATLAB=250000

DATASET="15Scene-Prod-0"
#DATASET="Caltech_80ImgClass-Prod-1"
#DATASET="Flowers-Prod-1"

# Set the frequency threshold you want to test
for iFreq in 30
do
  # Loop to process all folds
  for iFold in 0 1 2 3 4 5 6 7 8 9 
  do
    # Loop to process all vocabulary size
    for iWords in 100 #1000 2000 4000
    do
      ## !!!! EXCEPT for the iFold loop, the other ones are not especially
      # needed as you will probably test with the parameter we present in our
      # paper. The loop here were used to test multiple frequency threshold and
      # multiple vocabulary size in once script.
      # I think it will be better for you to copy this script for each set of
      # parameter we present in our article and run them individually (or with
      # a "master" script) in order to not loose to much time by testing every
      # possible set of parameter as we did.

      # Directory where graphs files are, comment out the one not need for the
      # dataset
      DIR="${SCRIPT}/data/${DATASET}/Folds${iFold}/train100-test50/Dense-step8-SIFT-binSize4-patchSize16-${iWords}Words"
      #DIR="${SCRIPT}/data/${DATASET}/Folds${iFold}/train72-test8/Dense-step8-SIFT-binSize4-patchSize16-${iWords}Words"

      # Directory where histogram, SVM models, etc. will be stored
      OUT="${SCRIPT}/data/${DATASET}/Folds${iFold}/train100-test50/Dense-step8-SIFT-binSize4-patchSize16-${iWords}Words/${iFreq}"
      #OUT="${SCRIPT}/data/${DATASET}/Folds${iFold}/train72-test8/Dense-step8-SIFT-binSize4-patchSize16-${iWords}Words/${iFreq}"
      mkdir -p $OUT

      # Train graph file
      TRAIN_FILE="${DIR}/Train-GraphDB.gri"
      # Histogramm of the train set
      TRAIN_CSV_FILE="${OUT}/train.csv"
      # Class of train graph
      TRAIN_LBL_FILE="${DIR}/train.lbl.csv"

      # Test graph file
      TEST_FILE="${DIR}/Test-GraphDB.gri"
      # Histogramm of the test set
      TEST_CSV_FILE="${OUT}/test.csv"
      # Class of test graph
      TEST_LBL_FILE="${DIR}/test.lbl.csv"

      # Pattern file that will be outputed by Grima
      PAT_FILE="${OUT}/pattern.gri"

      # Results file
      RES_FILE="${OUT}/classifResult${iFreq}.csv"

      # If pattern file does not exit process mining
      if ! [[ -e $PAT_FILE ]]
      then
        echo "~/thesis_alg_src/hologram2D/bin/hologram2D -T $TIMEOUT -f $iFreq -P $NBPAT_LIM_COUNT -o $OUT $TRAIN_FILE"
        time /home/tm08317s/thesis_alg_src/hologram2D/bin/hologram2D -T $TIMEOUT -f $iFreq -P $NBPAT_LIM_COUNT -o $OUT $TRAIN_FILE
      fi

      # Check that pattern file exist to be sure and output the number of
      # pattern for information
      if [[ -e $PAT_FILE ]]
      then
        NB_PAT=$( head -5 "$PAT_FILE" | grep NbFreqPattern | cut -d: -f2 )
        echo "NB PAT : $NB_PAT"
        # If train histograms does not exist, then count pattern in train set
        if ! [[ -e $TRAIN_CSV_FILE ]]
        then
          echo "./bin/countPattern2D -DW -O 0 -p $PAT_FILE -c $TRAIN_CSV_FILE -l $TRAIN_LBL_FILE $TRAIN_FILE"
          ./bin/countPattern2D -DW -O 0 -p $PAT_FILE -c $TRAIN_CSV_FILE -l $TRAIN_LBL_FILE $TRAIN_FILE
        fi
        # If test histograms does not exist, then count pattern in train set
        if ! [[ -e $TEST_CSV_FILE ]]
        then
          echo "./bin/countPattern2D -DW -O 0 -p $PAT_FILE -c $TEST_CSV_FILE -l $TEST_LBL_FILE $TEST_FILE"
          ./bin/countPattern2D -DW -O 0 -p $PAT_FILE -c $TEST_CSV_FILE -l $TEST_LBL_FILE $TEST_FILE
        fi
      fi

      # If results file does not exist, run classification task

      if ! [[ -e ${RES_FILE} ]]
      then
        cd bin
        echo "octave --eval \"classif_multiClass('$TRAIN_CSV_FILE','$TRAIN_LBL_FILE','$TEST_CSV_FILE','$TEST_LBL_FILE','$OUT', '${iFreq}' );\""
        octave --eval "classif_multiClass('$TRAIN_CSV_FILE','$TRAIN_LBL_FILE','$TEST_CSV_FILE','$TEST_LBL_FILE','$OUT', '${iFreq}' );"
        cd "$SCRIPT"
      else
        cat $RES_FILE
      fi

    done
  done
done
