# Examples of commands to train FormulaNet models.
# This command trains FormulaNet for unconditional premise seltection.
python batch_train.py \
--log ../logs/log_bi_s3_uc.txt \ # path to log file
--output ../models/model_bi_s3_uc \ # path to save models
--record ../logs/record_bi_s3_uc \ # path to records
--short_cut \ # add shortcut between steps
--nSteps 3 \ # number of steps
--max_pair 40000 \ # maximum number of edges to update
--uncondition \ # for uncondition problem
--binary # FormulaNet model, otherwise FormulaNet-basic model.

# This command trains FormulaNet-basic with 2 updating steps for conditional premise selection.
python batch_train.py \
--log ../logs/log_dir_s2.txt \
--output ../models/model_dir_s2 \
--record ../logs/record_dir_s2 \
--short_cut \
--nSteps 2 \
--max_pair 100000 \
--direction
