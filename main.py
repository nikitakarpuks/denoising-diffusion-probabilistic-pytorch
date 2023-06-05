import collections
import os
import tqdm.auto as tqdm
import numpy as np
from datasets import load_dataset
from models import unet
from torch.optim import AdamW
import data.augmentations
from torch.utils.data import DataLoader
from models.schedule import *


# outputs tensor value in timestep t with previously defined size
def extract(vector_in_time, t, x_shape):
    batch_size = t.shape[0]
    out = vector_in_time.gather(-1, t.cpu())
#     *() is an alternative for join and map function combination
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# outputs q(x_t|x_{t-1}) in timestep t
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # reparametrization trick formula
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# outputs loss function in timestep t
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        # Algorithm 1.4
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_0=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# apply transformations from TransformAugmentations class
def transforms(examples):
    if 'img' in examples:
        img_col_name = 'img'
    elif 'image' in examples:
        img_col_name = 'image'
    else:
        raise NotImplementedError()
        
    examples["pixel_values"] = [data.augmentations.TransformAugmentations()(image) for image in examples[img_col_name]]
    del examples[img_col_name]

    return examples


if __name__ == "__main__":

    # configuration import
    cfg = config.load("./config/config.yaml")

    # hyps
    timesteps = cfg.timesteps
    image_size = cfg.image_size
    device = cfg.device

    dataset = load_dataset(cfg.dataset_name)

    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(transformed_dataset["train"], batch_size=cfg.batch_size, shuffle=True)
    data_len = len(transformed_dataset["train"])

    model = unet.Unet(
        dim=image_size,
        channels=cfg.channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    # continue with saved checkpoint if exists
    checkpoint = None
    start_epoch = 0
    if os.path.isfile("./ckpts/ckpt.pt"):
        print('=> loading checkpoint:\n ckpt.pt')
        checkpoint = torch.load("./ckpts/ckpt.pt", torch.device(cfg.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        start_epoch = checkpoint['epoch']
    else:
        try:
            os.mkdir("./ckpts/")
        except:
            pass

    if cfg.optimizer == "Adamw":
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    else:
        raise NotImplementedError()

    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if cfg.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    else:
        raise NotImplementedError()

    loss_hist = collections.deque(maxlen=20)
    tqdm.tqdm._instances.clear()

    # training
    model.train()
    for epoch in range(start_epoch, cfg.epochs):

        pbar = tqdm.tqdm(enumerate(dataloader, start=1), position=0, leave=True)
        epoch_loss = []
        prc_processed = 0
        for step, batch in pbar:
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1.3
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type=cfg.loss)

            epoch_loss.append(float(loss))
            loss_hist.append(float(loss))

            prc_processed += 100 * batch_size / data_len
            print_text = '[{:.2f}%] | Epoch: {} | Step: {} | Running loss: {:1.5f}'.format(
                prc_processed, epoch, step, np.mean(loss_hist))
            pbar.set_description(print_text)

            optimizer.zero_grad()
            loss.backward()
            # Algorithm 1.5
            optimizer.step()

            del batch
            torch.cuda.empty_cache()

        scheduler.step(np.mean(epoch_loss))

        PATH = "./ckpts/ckpt.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, PATH)
