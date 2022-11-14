#!/bin/bash


# ref: https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
# ref: https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $(run_ts)"

echo "START TIME: $(date)"

ROOT_DIR=/home/ubuntu/cloudfs/saved_models/stable_diffusion/finetune_redbook_ocr/

if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi


config_yaml="$ROOT_DIR/base_config.yaml"


# dataset gen'd by ghostai_training/ocr_title_to_image/gen_data.ipynb
# from :https://github.com/invoke-ai/InvokeAI/blob/main/configs/stable-diffusion/v1-finetune.yaml
cat <<EOT > $config_yaml
model:
  base_learning_rate: 5.0e-03
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: caption
    image_size: 64
    channels: 4
    cond_stage_trainable: true   # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    personalization_config:
      target: ldm.modules.embedding_manager.EmbeddingManager
      params:
        placeholder_strings: ["*"]
        initializer_words: ["sculpture"]
        per_image_tokens: false
        num_vectors_per_token: 1
        progressive_words: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 8
    num_workers: 8
    num_val_workers: 0 # Avoid a weird val dataloader issue
    train:
      target: ldm.data.simple.hf_dataset
      params:
        name: /home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/txt2image_dataset
        image_transforms:
        - target: torchvision.transforms.Resize
          params:
            size: 512
            interpolation: 3
        - target: torchvision.transforms.RandomCrop
          params:
            size: 512
        - target: torchvision.transforms.RandomHorizontalFlip
    validation:
      target: ldm.data.simple.TextOnly
      params:
        captions:
        - "A pokemon with green eyes, large wings, and a hat"
        - "A cute bunny rabbit"
        - "Yoda"
        - "An epic landscape photo of a mountain"
        output_size: 512
        n_gpus: 2 # small hack to make sure we see all our samples
lightning:
  modelcheckpoint:
    params:
      every_n_train_steps: 500
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 500
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
    max_steps: 4000000
EOT

BATCH_SIZE=4
N_GPUS=8
ACCUMULATE_BATCHES=1
TRAIN_NAME=finetune_redbook_ocr
gpu_list=0,1,2,3,4,5,6,7

ckpt_path=/home/ubuntu/cloudfs/saved_models/models--CompVis--stable-diffusion-v-1-4-original/snapshots/f0bb45b49990512c454cf2c5670b0952ef2f9c71/sd-v1-4-full-ema.ckpt

# testing setup
#    --every_n_train_steps 100 \

python main.py \
    -t \
    --base $config_yaml \
    -l $ROOT_DIR \
    --name "$TRAIN_NAME"\
    --gpus $gpu_list \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --resume_from_checkpoint $ckpt_path \
    --every_n_train_steps 100 \
    data.params.batch_size=$BATCH_SIZE \
    lightning.trainer.accumulate_grad_batches=$ACCUMULATE_BATCHES \
    data.params.validation.params.n_gpus=$N_GPUS