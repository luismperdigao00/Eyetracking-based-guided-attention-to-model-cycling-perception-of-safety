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
RUN_NAME="61441nj1"
CHECKPOINT="crimson-sweep-3_best_model_5_0.7689.pt"
#TEST_SET="comparisons_df_with_synthetic_berlin.pickle"
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
    --cuda_id 1

echo "Test finished: $RUN_NAME ($TEST_SET)"

#python "$PYTHON_SCRIPT" \
#    --comparisons "$TEST_SET" \
#    --dataset images/printart/subjectivesafety_images \
#    --cities berlin \
#    --cnn_pool "flatten" \
#    --backbone "vgg" \
#    --checkpoint "vgg_syn+ber.pt" \
#    --model rsscnn \
#    --gaze "off" \
#    --ties \
#    --cuda \
#    --cuda_id 0

#python "$PYTHON_SCRIPT" \
#    --comparisons "$TEST_SET" \
#    --dataset images/printart/subjectivesafety_images \
#    --cities "paris, barcelona, munich" \
#    --wandb_run_id "$RUN_NAME" \
#    --checkpoint "$CHECKPOINT" \
#    --cuda \
#    --cuda_id 0


#BEST NON GAZE:
#RUN_NAME="mbg8xvwg"
#CHECKPOINT="lemon-sweep-1_best_model_10_0.7847.pt"
#Best gaze:
#RUN_NAME="dw1th1vv"
#CHECKPOINT="confused-sweep-2_best_model_8_0.7899.pt"

#RUN_NAME="rwoa07zg"
#CHECKPOINT="divine-sweep-1_best_model_6_0.8021.pt"