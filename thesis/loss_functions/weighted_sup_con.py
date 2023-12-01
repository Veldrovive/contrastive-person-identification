"""
Implements the SupCon loss function from https://arxiv.org/pdf/2004.11362.pdf

Adjusted to allow for weighting the positive sets differently.
"""

import torch
from torch import nn

class WeightedSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def _tensor_sim(self, t):
        """
        Assumes t is nxm. Returns a nxn similarity matrix.
        If rows i and j are equal, then the (i, j) and (j, i) entries are 1. Otherwise they are 0.
        """
        t_expanded = t.unsqueeze(1)  # Shape is nx1xm
        t_expanded_T = t.unsqueeze(0)  # Shape is 1xnxm

        eq_mat = torch.eq(t_expanded, t_expanded_T)  # Shape is nxnxm

        # Now we must have that all elements are the row are equal
        sim_mat = torch.all(eq_mat, dim=2).float()  # Shape is nxn

        return sim_mat

    def forward(self, features, labels: list[tuple[float, torch.Tensor]]):
        """
        Features are of the size (m*N, D) where m is the number of augmented samples per input sample and N is the batch size.
        
        Labels are a list of different positive sets. Each set contains a weight as well as the labels.
        The weight scales the loss for that positive set and the labels are used to compute the set label mask.

        We assume that the features are already normalized.
        """
        mN = features.shape[0]
        # Compute the similarity matrix and label mask
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # NEW: Instead of using a single label mask, we have a label tensor of size (mN, mN, num_labels)
        label_masks = torch.zeros((mN, mN, len(labels)), device=features.device)
        for i, (weight, label_set) in enumerate(labels):
            if label_set.ndim == 1:
                label_set = label_set.contiguous().view(-1, 1)
            label_masks[:, :, i] = self._tensor_sim(label_set) * weight
        label_mask = torch.sum(label_masks, dim=2)

        inverse_eye = torch.ones_like(label_mask) - torch.eye(mN, device=label_mask.device)
        label_mask = label_mask * inverse_eye

        # Get the mask that we will use to compute the denominator
        denom_mask = inverse_eye

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

        # The final part is taking the mean by using the weight of the label mask
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
    labels = torch.randint(0, 3, (600,))
    # Now we backprop these features. We expect the features to cluster by label
    features.requires_grad = True
    loss = SupConLoss(temperature=0.1)

    previous_features = torch.zeros_like(features)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    plt.ion()  # Turn on interactive mode

    count = 0
    while torch.norm(previous_features - features) > 0.001:
        previous_features = features.clone()
        normed_features = features / torch.norm(features, dim=1, keepdim=True)
        output = loss(normed_features, labels)
        output.backward()
        features.data -= 0.5 * features.grad
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