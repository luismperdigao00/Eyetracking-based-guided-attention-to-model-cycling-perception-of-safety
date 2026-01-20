#!/usr/bin/env bash
set -e

# ----------------------------------
# Environment
# ----------------------------------
VENV_PATH=".venv"
#export CUDA_VISIBLE_DEVICES=0

# ----------------------------------
# Test configuration
# ----------------------------------
RUN_NAME="2ainhlx4"
CHECKPOINT="treasured-sweep-1_best_model_8_0.7292.pt"
#CHECKPOINT="quiet-sweep-2_best_model_3_0.7382.pt"   #"vgg_syn+ber.pt" #fluent-sweep-18_model_2_0.7382.pt"
#CHECKPOINT="resilient-sweep-3_best_model_5_0.7333.pt" #GAZE
#TEST_SET="splits/comparisons_df_test.pkl"
TEST_SET="build_datasets/comparisons_tests.pkl"

PYTHON_SCRIPT="test.py"

# ----------------------------------
# Activate virtual environment
# ----------------------------------
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: virtualenv not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# ----------------------------------
# Run test (CORRECT ARGS)
 #--wandb_run_id "$RUN_NAME" \
# ----------------------------------
python "$PYTHON_SCRIPT" \
    --comparisons "$TEST_SET" \
    --dataset images/printart/subjectivesafety_images \
    --cities "berlin" \
    --wandb_run_id "$RUN_NAME" \
    --checkpoint "$CHECKPOINT" \
    --cuda \
    --cuda_id 0

echo "Test finished: $RUN_NAME ($TEST_SET)"

#    --backbone "vgg" \
#    --model rsscnn \
#    --gaze off \
#    --ties \