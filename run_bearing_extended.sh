ENTITY=rul-adapt
PROJECT=benchmark
APPROACHES=("no_adaption" "adarul" "conditional_dann" "conditional_mmd" "consistency" "dann" "latent_align" "mmd" "pseudo_labels")
XJTUSY_TASKS="one2two,one2three,two2one,two2three,three2one,three2two"
FEMTO_TASKS_12="one2two,one2three,two2one,two2three"
FEMTO_TASKS_3="three2one,three2two"
REPLICATIONS=5

for APPROACH in "${APPROACHES[@]}"; do
poetry run python train.py \
       --multirun hydra/launcher=ray \
       +hydra.launcher.num_gpus=0.5 \
       +task="$FEMTO_TASKS_12" \
       +approach="$APPROACH" \
       +feature_extractor=cnn \
       +dataset=femto \
       test=True \
       logger.entity=$ENTITY \
       logger.project=$PROJECT \
       +logger.tags="[femto,cnn,transductive,extended]" \
       replications=$REPLICATIONS \
       +source.reader.run_split_dist="{dev:[1,2,4,5,6,7],val:[3],test:[]}"
poetry run python train.py \
       --multirun hydra/launcher=ray \
       +hydra.launcher.num_gpus=0.5 \
       +task="$FEMTO_TASKS_3" \
       +approach="$APPROACH" \
       +feature_extractor=cnn \
       +dataset=femto \
       test=True \
       logger.entity=$ENTITY \
       logger.project=$PROJECT \
       +logger.tags="[femto,cnn,transductive,extended]" \
       replications=$REPLICATIONS \
       +source.reader.run_split_dist="{dev:[1,3],val:[2],test:[]}"
poetry run python train.py \
       --multirun hydra/launcher=ray \
       +hydra.launcher.num_gpus=0.5 \
       +task="$XJTUSY_TASKS" \
       +approach="$APPROACH" \
       +feature_extractor=cnn \
       +dataset="xjtu-sy" \
       test=True \
       logger.entity=$ENTITY \
       logger.project=$PROJECT \
       +logger.tags="[xjtu-sy,cnn,transductive,extended]" \
       replications=$REPLICATIONS \
       +source.reader.run_split_dist="{dev:[1,2,4,5],val:[3],test:[]}"
done
