import torch
import matplotlib.pyplot as plt

kit_green = (0, 150/255, 130/255)
kit_blue = (70/255, 100/255, 170/255)
kit_red = (162/255, 34/255, 35/255)
kit_green_50 = (0.5, 0.7941, 0.7549)
kit_blue_50 = (0.6373, 0.6961, 0.8333)
kit_red_50 = (0.8176, 0.5667, 0.5686)

def find_threshold(sigmas):
    # Finds the threshold value for high vs low sigma values
    sorted_sigma, indices = torch.sort(torch.abs(torch.flatten(sigmas)))
    difference = sorted_sigma[1:] - sorted_sigma[:-1]
    thr = (sorted_sigma[torch.argmax(difference) + 1] - sorted_sigma[torch.argmax(difference)]) / 2 + sorted_sigma[
        torch.argmax(difference)]
    return thr


def sigma_weight_plot(weights, sigmas, base_name):
    # Creates scatter plot for a layer's sigma values and weights
    final_weights = (torch.flatten(weights)).tolist()
    final_sigma = (torch.abs(torch.flatten(sigmas))).tolist()

    x1 = final_sigma

    y1 = final_weights

    plt.figure(figsize=(10, 8))
    plt.scatter(x1, y1, c=kit_blue_50,
                linewidths=2,
                marker="s",
                edgecolor=kit_blue,
                s=50)
    # plt.xlim(0, 1)
    # plt.ylim(-0.5, 0.5)

    plt.xlabel("Sigma Values", fontsize=18)
    plt.ylabel("Weights", fontsize=18)

    # plt.show()
    file_name = base_name + '-sigma-values.png'
    plt.savefig(file_name)
    #plt.show()

def determine_prune_weights(init_weights, final_weights, final_sigmas, threshold):
    # param_list should contain only a single torch layer
    final_weights_prune = torch.where((torch.flatten(final_sigmas)) > threshold, (torch.flatten(final_weights)),
                                      0).tolist()
    initial_weights_prune = torch.where((torch.flatten(final_sigmas)) > threshold, (torch.flatten(init_weights)),
                                        0).tolist()
    final_weights_prune = [item for item in final_weights_prune if item != 0]
    initial_weights_prune = [item for item in initial_weights_prune if item != 0]

    final_weights_remain = torch.where((torch.flatten(final_sigmas)) > threshold, 0,
                                       (torch.flatten(final_weights))).tolist()
    initial_weights_remain = torch.where((torch.flatten(final_sigmas)) > threshold, 0,
                                         (torch.flatten(init_weights))).tolist()
    final_weights_remain = [item for item in final_weights_remain if item != 0]
    initial_weights_remain = [item for item in initial_weights_remain if item != 0]

    return (final_weights_prune, initial_weights_prune, final_weights_remain, initial_weights_remain)

def plot_init_final_weights(final_weights_prune, initial_weights_prune, final_weights_remain,
                            initial_weights_remain, base_name):
    x1 = initial_weights_prune
    y1 = final_weights_prune
    x2 = initial_weights_remain
    y2 = final_weights_remain

    plt.figure(figsize=(10, 8))
    pr = plt.scatter(x1, y1, c=kit_red_50,
                     linewidths=2,
                     marker="s",
                     edgecolor=kit_red,
                     s=50)

    re = plt.scatter(x2, y2, c=kit_blue_50,
                     linewidths=2,
                     marker="s",
                     edgecolor=kit_blue,
                     s=50)

    # plt.xlim(0, 1)
    # plt.ylim(-0.5, 0.5)

    plt.xlabel("Initial Weights", fontsize=18)
    plt.ylabel("Final Weights", fontsize=18)
    plt.xlim((-1, 1))  # restricts x axis from 0 to 25
    plt.ylim((-1, 1))  # restricts x axis from 0 to 25
    plt.plot([-1, 2.5], [-1, 2.5])  # plots line y = x

    plt.legend((pr, re),
               ('High Sigma Weights', 'Low Sigma Weights'),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=12)
    plt.axis('square')
    #plt.show()
    file_name = base_name + '-weight-change.png'
    plt.savefig(file_name)
