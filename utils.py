import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_animation(samples, timesteps, image_size, channels):
    # choose number of images to make an animation of their diffusion
    n = 1
    for j in range(n):
        fig = plt.figure()
        ims = []
        for i in range(timesteps):
            im = plt.imshow(samples[i][j].reshape(image_size, image_size, channels), cmap="gray", animated=True)
            ims.append([im])

        animate = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=5000)
        animate.save(f'diffusion-animated-{j}.gif')
