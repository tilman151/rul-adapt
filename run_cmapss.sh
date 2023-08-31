ENTITY=rul-adapt
PROJECT=benchmark
APPROACHES=("adarul" "conditional_dann" "conditional_mmd" "consistency dann" "latent_align" "mmd" "pseudo_labels")
REPLICATIONS=5

for APPROACH in "${APPROACHES[@]}"; do
poetry run python train.py \
       --multirun hydra/launcher=ray \
       +hydra.launcher.num_gpus=0.25 \
       +task="glob(*)" \
       +approach="$APPROACH" \
       +feature_extractor=cnn \
       +dataset=cmapss \
       test=True \
       logger.entity=$ENTITY \
       logger.project=$PROJECT \
       +logger.tags="[cmapss,cnn,transductive]" \
       replications=$REPLICATIONS
poetry run python train.py \
        --multirun hydra/launcher=ray \
        +hydra.launcher.num_gpus=0.25 \
        +task="glob(*)" \
        +approach="$APPROACH" \
        +feature_extractor=cnn \
        +dataset=cmapss \
       adaption_mode=inductive \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[cmapss,cnn,inductive]" \
        replications=$REPLICATIONS
poetry run python train.py \
        --multirun hydra/launcher=ray \
        +hydra.launcher.num_gpus=0.25 \
        +task="glob(*)" \
        +approach="$APPROACH" \
        +feature_extractor=cnn \
        +dataset=cmapss \
       ++target.reader.percent_broken=1.0 \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[cmapss,cnn,complete]" \
        replications=$REPLICATIONS
poetry run python train.py \
         --multirun hydra/launcher=ray \
         +hydra.launcher.num_gpus=0.25 \
         +task="glob(*)" \
         +approach="$APPROACH" \
         +feature_extractor=lstm \
         +dataset=cmapss \
         test=True \
         logger.entity=$ENTITY \
         logger.project=$PROJECT \
         +logger.tags="[cmapss,lstm,transductive]" \
         replications=$REPLICATIONS
done
