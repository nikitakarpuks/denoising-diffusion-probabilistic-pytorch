from tqdm.auto import tqdm
from torchvision.utils import save_image
from models import unet
from pathlib import Path
import random
import utils
from models.schedule import *


def extract(vector_in_time, t, x_shape):
    # timestamp here is a tensor which size is a batch size and which value is a timestep
    batch_size = t.shape[0]
    out = vector_in_time.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_inv_alphas_t = extract(sqrt_alphas_inv, t, x.shape)

    model_mean = sqrt_inv_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    # Algorithm 2.3
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        # Algorithm 2.4
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(model, image_size, batch_size=256, channels=1):
    shape = (batch_size, channels, image_size, image_size)
    batch_size = shape[0]

    # Algorithm 2.1
    img = torch.randn(shape, device=device)

    imgs = []
    # Algorithm 2.2
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())

    return imgs


def select_random_image(samples, n, step):
    image_index = random.randint(0, len(samples[0]) - 1)
    selected_images = [samples[i][image_index] for i in range(0, n, step)]
    stacked_images = torch.stack(selected_images, dim=0)

    return stacked_images


# import config
cfg = config.load("./config/config.yaml")


# hyperparameters
timesteps = cfg.timesteps
image_size = cfg.image_size
device = cfg.device
channels = cfg.channels

# define model's arch
model = unet.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

path_to_ckpt = './ckpts/ckpt_cifar10-300.pt'
checkpoint = torch.load(path_to_ckpt)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

results_folder = Path("./inference_images/")
results_folder.mkdir(exist_ok=True)

# sampling
samples = sample(model, image_size=image_size, batch_size=cfg.sample, channels=channels)

# denoising process vizualization (10 random images)
n = cfg.timesteps
step = n//10
imgs = []
for i in range(10):
    imgs.append(select_random_image(samples, n, step))
imgs = torch.cat(imgs)
save_image(imgs, str(results_folder / f'gradually-{cfg.dataset_name}.png'), nrow=10)

# save all generated images
save_image(samples[-1], str(results_folder / f'{cfg.dataset_name}-big.png'), nrow=int(math.sqrt(cfg.sample)))

# create gif
utils.make_animation(samples, timesteps, image_size, channels)
