#!/bin/bash

# Direct FourierFT Evaluation Script (No Soft Inference)
# This script evaluates FourierFT orthogonal training results directly
# using fixed ood_weight=[1,1] without OOD detection

BASE_MODEL="gcyzsl/O3_LLAMA2_ScienceQA"

echo "=== Direct FourierFT Evaluation (No Soft Inference) ==="

for SCALE in 0.1
do
  for SEED in 0 1 2
  do
    for LABEL_K in "force"
    do
      OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_5/lora_${LABEL_K}_random"
      TYPE=""
      for UNLEAN_D in "biology" "physics" "chemistry"
      do
        OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
        TYPE+="_${UNLEAN_D}"

        echo "Evaluating: SEED=${SEED}, SCALE=${SCALE}, TYPE=${TYPE}"
        echo "Adapter path: ${OUTPUT_1}"
        
        # Test on Retain Data (RD) - should maintain high performance
        TESTPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}_test_RD.json"
        echo "  -> Testing RD (Retain Data): ${TESTPATH_1}"
        python eval_direct_fourierft.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --fourierft_weights ${OUTPUT_1} \
          --seed ${SEED}

        # Test on Specific Domain train (SD) - should show unlearning effect
        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_train_SD.json"
        echo "  -> Testing SD train (Specific Domain): ${TESTPATH_1}"
        python eval_direct_fourierft.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --fourierft_weights ${OUTPUT_1} \
          --seed ${SEED}

        # Test on Specific Domain test (SD) - should show unlearning effect  
        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_test_SD.json"
        echo "  -> Testing SD test (Specific Domain): ${TESTPATH_1}"
        python eval_direct_fourierft.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --fourierft_weights ${OUTPUT_1} \
          --seed ${SEED}

        # Test on CommonQA - should maintain general capability
        TESTPATH_1="./data/commonqa/commonqa_test.json"
        echo "  -> Testing CommonQA (General Knowledge): ${TESTPATH_1}"
        python eval_direct_fourierft.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --fourierft_weights ${OUTPUT_1} \
          --seed ${SEED}

        # Test on OpenBookQA - should maintain general capability
        TESTPATH_1="./data/openbookqa/openbookqa_test.json"
        echo "  -> Testing OpenBookQA (General Knowledge): ${TESTPATH_1}"
        python eval_direct_fourierft.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --fourierft_weights ${OUTPUT_1} \
          --seed ${SEED}

        echo "Completed evaluation for ${TYPE}"
        echo "----------------------------------------"
      done
    done
  done
done

echo "=== All Direct FourierFT Evaluations Completed ==="
