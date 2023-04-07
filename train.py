# from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# cifa_data = "/home/hjx/Diffusion-Probabilistic-Models"

# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# ).cuda()

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 32,
#     timesteps = 1000,           # number of steps
#     sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
#     loss_type = 'l1'            # L1 or L2
# ).cuda()

# trainer = Trainer(
#     diffusion,
#     cifa_data,
#     train_batch_size = 32,
#     train_lr = 8e-5,
#     train_num_steps = 700000,         # total training steps
#     gradient_accumulate_every = 2,    # gradient accumulation steps
#     ema_decay = 0.995,                # exponential moving average decay
#     amp = True                        # turn on mixed precision
# )

# trainer.train()

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
# print(tokenizer.get_vocab())
input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text).input_ids
print(input_ids)
# s = tokenizer.convert_tokens_to_string(input_ids)
# print(s)