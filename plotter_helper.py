import matplotlib.pyplot as plt


def evaluation_side_by_side_plot(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs[0].permute((1, 2, 0)).numpy().astype(int))
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs[0].view((400, 400)).detach().numpy(), cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth[0].view((400, 400)), cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)


def evaluation_side_by_side_plot_np(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs)
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs, cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth, cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)
