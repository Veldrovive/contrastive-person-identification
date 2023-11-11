"""
Implements the SupCon loss function from https://arxiv.org/pdf/2004.11362.pdf

"""

import torch
from torch import nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Features are of the size (m*N, D) where m is the number of augmented samples per input sample and N is the batch size.
        Labels are of the size (m*N,). Assumed to be a tensor of integers.

        We assume that the features are already normalized.

        Note: There do not need to be equal numbers of each label, we just assume that for the explanation of the feature size
        """
        mN = features.shape[0]
        # Compute the similarity matrix and label mask
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float()

        # Get the mask that we will use to compute the denominator
        denom_mask = torch.ones_like(label_mask) - torch.eye(mN, device=label_mask.device)

        # Now we compute the denominator using the log-sum-exp trick
        # First step is to compute the similarity matrix masked by the denominator mask
        denom_masked_sim_matrix = similarity_matrix * denom_mask
        # Then we take the max over the rows to prep for the log-sum-exp trick
        sim_max = torch.max(denom_masked_sim_matrix, dim=1, keepdim=True).values
        de_maxed_masked_sim_matrix = denom_masked_sim_matrix - sim_max
        # Now we compute the log-sum-exp
        exp_matrix = torch.exp(de_maxed_masked_sim_matrix) * denom_mask  # We need to remove the diagonal again. Can this be combined?
        exp_sum = torch.sum(exp_matrix, dim=1, keepdim=True)
        log_sum_exp = torch.log(exp_sum)
        # Finally we compute the denominator
        denominator = sim_max + log_sum_exp

        # The numerator is just the similarity matrix 
        numerator = similarity_matrix

        # The final part is taking the mean by using the cardinality of the label mask
        normalizer = torch.sum(label_mask, dim=1, keepdim=True)

        # Our final matrix is (numerator - denominator) / normalizer masked by the label mask
        S = (numerator - denominator) / normalizer
        S = S * label_mask

        # And the loss is the negative sum of every element in the matrix
        loss = -torch.sum(S) / mN
        
        return loss

if __name__ == "__main__":
    # # Set the seed
    # torch.manual_seed(0)
    # # Create example data
    # N = 32
    # D = 128
    # num_labels = 10
    # features = torch.randn(N, D)
    # # Normalize the features
    # features = features / torch.norm(features, dim=1, keepdim=True)
    # labels = torch.randint(0, num_labels, (N,))

    # loss = SupConLoss()
    # output = loss(features, labels)
    # print(output)

    # # Manual creation of some features for checking correctness
    # features = torch.tensor([
    #     [1, 0],
    #     [0, 1],
    #     [1, 0],
    #     [0, 1]
    # ], dtype=torch.float32)
    # labels_1 = torch.tensor([0, 1, 0, 1])  # Should have low loss
    # labels_2 = torch.tensor([0, 0, 1, 1])  # Should have high loss

    # loss = SupConLoss()
    # output_1 = loss(features, labels_1)
    # output_2 = loss(features, labels_2)
    # print(output_1)
    # print(output_2)

    # features = torch.tensor([
    #     [1, 0],
    #     [-1, 0],
    #     [1, 0],
    #     [-1, 0]
    # ], dtype=torch.float32)
    # output_3 = loss(features, labels_1)  # Should be lower than output_1
    # output_4 = loss(features, labels_2)  # Should be higher than output_2
    # print(output_3)
    # print(output_4)

    # assert output_3 < output_1
    # assert output_1 < output_2
    # assert output_2 < output_4


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time
    # features = torch.randn(600, 2)
    # labels = torch.randint(0, 10, (600,))
    # # Now we backprop these features. We expect the features to cluster by label
    # features.requires_grad = True
    # loss = SupConLoss()

    # previous_features = torch.zeros_like(features)

    # plt.ion()  # Turn on interactive mode
    # count = 0
    # while torch.norm(previous_features - features) > 0.01:
    #     previous_features = features.clone()
    #     output = loss(features, labels)
    #     output.backward()
    #     features.data -= 1 * features.grad
    #     features.grad.zero_()
    #     if count % 10 == 0:
    #         plt.clf()  # Clear the current figure
    #         plt.scatter(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), c=labels.detach().numpy())
    #         plt.draw()  # Redraw the current figure
    #         plt.pause(0.01)  # Pause for a brief moment to update the plot
    #         print(output)
    #     if count == 0:
    #         time.sleep(5)
    #     count += 1

    # plt.ioff()  # Turn off interactive mode
    # plt.show()  # Show the final plot


    features = torch.randn(600, 3)
    labels = torch.randint(0, 10, (600,))
    # Now we backprop these features. We expect the features to cluster by label
    features.requires_grad = True
    loss = SupConLoss()

    previous_features = torch.zeros_like(features)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    plt.ion()  # Turn on interactive mode

    count = 0
    while torch.norm(previous_features - features) > 0.01:
        previous_features = features.clone()
        output = loss(features, labels)
        output.backward()
        features.data -= 1 * features.grad
        features.grad.zero_()
        if count % 10 == 0:
            ax.clear()  # Clear the current figure
            ax.scatter(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), features[:, 2].detach().numpy(), c=labels.detach().numpy())
            plt.draw()  # Redraw the current figure
            plt.pause(0.01)  # Pause for a brief moment to update the plot
            print(output)
        if count == 0:
            time.sleep(5)  # Sleep for 5 seconds on the first plot
        count += 1

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot