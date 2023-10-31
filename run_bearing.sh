ENTITY=rul-adapt
PROJECT=benchmark
APPROACHES=("no_adaption" "adarul" "conditional_dann" "conditional_mmd" "consistency" "dann" "latent_align" "mmd" "pseudo_labels")
DATASETS=("femto" "xjtu-sy")
TASKS="one2two,one2three,two2one,two2three,three2one,three2two"
REPLICATIONS=5

for APPROACH in "${APPROACHES[@]}"; do
for DATASET in "${DATASETS[@]}"; do
poetry run python train.py \
       --multirun hydra/launcher=ray \
       +hydra.launcher.num_gpus=0.5 \
       +task="$TASKS" \
       +approach="$APPROACH" \
       +feature_extractor=cnn \
       +dataset="$DATASET" \
       evaluation_mode=degraded_only \
       test=True \
       logger.entity=$ENTITY \
       logger.project=$PROJECT \
       +logger.tags="[$DATASET,cnn,transductive]" \
       replications=$REPLICATIONS
done
done
