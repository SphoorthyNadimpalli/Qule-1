# Library imports
# import math
import random
import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import pennylane as qml
import torch.nn.functional as F
# Pytorch imports
import torch
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader

# from rdkit.Chem import Draw
from utils import *
from models import Discriminator
from sparse_molecular_dataset import SparseMolecularDataset

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


######################################################################
# Data
# ~~~~
#
######################################################################
# As mentioned in the introduction, we will use a `small
# dataset <https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits>`__
# of handwritten zeros. First, we need to create a custom dataloader for
# this dataset.
#
# class MolDataset:
#     def __init__(self, dataset_filepath):
#         self.data = SparseMolecularDataset()
#         self.data.load(dataset_filepath)
#
#     def get_data(self):
#         return self

# class MolDataset():
#     def __init__(self):
#         self.data = dataloader.__getData__()
#         return self.data

# class DigitsDataset(Dataset):
#     """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""
#
#     def __init__(self, csv_file, label=0, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.csv_file = csv_file
#         self.transform = transform
#         self.df = self.filter_by_label(label)
#
#     def filter_by_label(self, label):
#         # Use pandas to return a dataframe of only zeros
#         df = pd.read_csv(self.csv_file)
#         df = df.loc[df.iloc[:, -1] == label]
#         return df
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         image = self.df.iloc[idx, :-1] / 16
#         image = np.array(image)
#         image = image.astype(np.float32).reshape(8, 8)
#
#         if self.transform:
#             image = self.transform(image)
#
#         # Return image and label
#         return image, 0


######################################################################
# Next we define some variables and create the dataloader instance.
#

# image_size = 8  # Height / width of the square images
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = DigitsDataset(csv_file="quantum_gans/optdigits.tra", transform=transform)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=True, drop_last=True
# )

######################################################################
# Let's visualize some of the data.

class DrawMols:

    def __init__(self, mols, fname):
        img = Draw.MolsToGridImage(mols[1:9], molsPerRow=4, subImgSize=(200, 200))
        img.save('images/' + fname)

# plt.figure(figsize=(8, 2))
#
# for i in range(8):
#     # image = dataset[i][0].reshape(image_size, image_size)
#     plt.subplot(1, 8, i+1)
#     plt.axis('off')
#     # plt.imshow(image.numpy(), cmap='gray')
#
# plt.show()
######################################################################
# Implementing the Discriminator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
######################################################################
# For the discriminator, we use a fully connected neural network with two
# hidden layers. A single output is sufficient to represent the
# probability of an input being classified as real.
#


# class Discriminator(nn.Module):
#     """Fully connected classical discriminator"""
#
#     def __init__(self):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             # Inputs to first hidden layer (num_input_features -> 64)
#             nn.Linear(image_size * image_size, 64),
#             nn.ReLU(),
#             # First hidden layer (64 -> 16)
#             nn.Linear(64, 16),
#             nn.ReLU(),
#             # Second hidden layer (16 -> output)
#             nn.Linear(16, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         return self.model(x)
######################################################################
# Implementing the Generator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Each sub-generator, :math:`G^{(i)}`, shares the same circuit
# architecture as shown below. The overall quantum generator consists of
# :math:`N_G` sub-generators, each consisting of :math:`N` qubits. The
# process from latent vector input to image output can be split into four
# distinct sections: state embedding, parameterisation, non-linear
# transformation, and post-processing. Each of the following sections
# below refer to a single iteration of the training process to simplify
# the discussion.
#
# .. figure:: ../demonstrations/quantum_gans/qcircuit.jpeg
#   :width: 90%
#   :alt: quantum_circuit
#   :align: center
#
# **1) State Embedding**
#
#
# A latent vector, :math:`\boldsymbol{z}\in\mathbb{R}^N`, is sampled from
# a uniform distribution in the interval :math:`[0,\pi/2)`. All
# sub-generators receive the same latent vector which is then embedded
# using RY gates.
#
# **2) Parameterised Layers**
#
#
# The parameterised layer consists of parameterised RY gates followed by
# control Z gates. This layer is repeated :math:`D` times in total.
#
# **3) Non-Linear Transform**
#
#
# Quantum gates in the circuit model are unitary which, by definition,
# linearly transform the quantum state. A linear mapping between the
# latent and generator distribution would be suffice for only the most
# simple generative tasks, hence we need non-linear transformations. We
# will use ancillary qubits to help.
#
# For a given sub-generator, the pre-measurement quantum state is given
# by,
#
# .. math:: |\Psi(z)\rangle = U_{G}(\theta)|\boldsymbol{z}\rangle
#
# where :math:`U_{G}(\theta)` represents the overall unitary of the
# parameterised layers. Let us inspect the state when we take a partial
# measurment, :math:`\Pi`, and trace out the ancillary subsystem,
# :math:`\mathcal{A}`,
#
#
# The post-measurement state, :math:`\rho(\boldsymbol{z})`, is dependent
# on :math:`\boldsymbol{z}` in both the numerator and denominator. This
# means the state has been non-linearly transformed! For this tutorial,
# :math:`\Pi = (|0\rangle \langle0|)^{\otimes N_A}`, where :math:`N_A`
# is the number of ancillary qubits in the system.
#
# With the remaining data qubits, we measure the probability of
# :math:`\rho(\boldsymbol{z})` in each computational basis state,
# :math:`P(j)`, to obtain the sub-generator output,
# :math:`\boldsymbol{g}^{(i)}`,
#
# .. math::  \boldsymbol{g}^{(i)} = [P(0), P(1), ... ,P(2^{N-N_A} - 1)]
#
# **4) Post Processing**
#
#
# Due to the normalisation constraint of the measurment, all elements in
# :math:`\boldsymbol{g}^{(i)}` must sum to one. This is
# a problem if we are to use :math:`\boldsymbol{g}^{(i)}` as the pixel
# intensity values for our patch. For example, imagine a hypothetical
# situation where a patch of full intensity pixels was the target. The
# best patch a sub-generator could produce would be a patch of pixels all
# at a magnitude of :math:`\frac{1}{2^{N-N_A}}`. To alleviate this
# constraint, we apply a post-processing technique to each patch,
#
# .. math::  \boldsymbol{\tilde{x}^{(i)}} = \frac{\boldsymbol{g}^{(i)}}{\max_{k}\boldsymbol{g}_k^{(i)}}
#
# Therefore, the final image, :math:`\boldsymbol{\tilde{x}}`, is given by
#
# .. math:: \boldsymbol{\tilde{x}} = [\boldsymbol{\tilde{x}^{(1)}}, ... ,\boldsymbol{\tilde{x}^{(N_G)}}]
#

# Quantum variables
n_qubits = 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 9  # Depth of the parameterised quantum circuit / D
n_generators = 81  # 4  # Number of subgenerators for the patch method / N_G


######################################################################
# Now we define the quantum device we want to use, along with any
# available CUDA GPUs (if available).
#

# Quantum simulator
dev = qml.device("lightning.qubit", wires=n_qubits)
# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Next, we define the quantum circuit and measurement process described above.


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):

    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))


# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven

######################################################################
# Now we create a quantum generator class to use during training.


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, flag, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        # atom or feature matrix shape required by discrim. is 16x9.
        # shape of the tensor output by generator = 1, (2 ** n_qubit - n_aqubits)*n_generators
        if flag:
            n_generators = int(n_generators/9)

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images


######################################################################
# Training
# ~~~~~~~~

######################################################################
# Let's define learning rates and number of iterations for the training process.

# Collect molecules for plotting later
# Results store output of partially trained model for every 50 iterations
results_a, results_x = [], []


class TrainNtest:
    def __init__(self):
        dataset_filepath = "data/gdb9_9nodes.sparsedataset"
        self.data = SparseMolecularDataset()
        self.data.load(dataset_filepath)
        self.num_iter = 500  # Number of training iterations
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.dropout = 0
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.num_iters = 20
        self.batch_size = 16
        self.d_conv_dim = [[128, 64], 128, [128, 64]]
        self.lrG = 0.3  # Learning rate for the generator
        self.lrD = 0.0001  # Learning rate for the discriminator
        self.lambda_gp = 10.0  # lambda parameter for GP
        self.metric = 'validity,sas'
        self.model_save_dir = 'molgan/models'
        self.resume_iters = None
        self.test_iters = 20
        torch.autograd.set_detect_anomaly(True)

        # Iteration counter
        # counter = 0

        ######################################################################
        # Now putting everything together and executing the training process.

        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout).to(device)
        # discriminator = Discriminator().to(device)
        self.generator_a = PatchQuantumGenerator(n_generators, False).to(device)
        self.generator_x = PatchQuantumGenerator(n_generators, True).to(device)

        # Binary cross entropy
        # criterion = nn.BCELoss()

        # Optimisers
        # optD = optim.SGD(discriminator.parameters(), lr=lrD)

        self.Ga_optimizer = torch.optim.Adam(self.generator_a.parameters(), lr=self.lrG)
        self.Gx_optimizer = torch.optim.Adam(self.generator_x.parameters(), lr=self.lrG)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lrD, (self.beta1, self.beta2))

        # real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        # fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

        # Fixed noise allows us to visually track the generated images throughout training
        self.a_fixed_noise = torch.rand(1, n_qubits, device=device) * math.pi / 2
        self.x_fixed_noise = torch.rand(1, n_qubits, device=device) * math.pi / 2

        # Noise following a uniform distribution in range [0,pi/2)
        noise = torch.rand(1, n_qubits, device=device) * math.pi / 2
        self.fake_atom = self.generator_a(noise)
        self.fake_bond = self.generator_x(noise)

    def train(self):
        start_iter = 0
        # Start training.
        for i in range(start_iter, self.num_iters):
            if (i + 1) % 10 == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()

            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)

                # Visualize training dataset samples
                DrawMols(mols, 'train_mol.png')
                print('[Validation Batch Initiated]', '')

            """Data for training the discriminator"""

            a = torch.from_numpy(a).to(device).long()  # Adjacency.
            x = torch.from_numpy(x).to(device).long()  # Nodes.
            a_tensor = self.label2onehot(a, self.m_dim)
            x_tensor = self.label2onehot(x, self.b_dim)

            # Compute loss
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real = - torch.mean(logits_real)

            # data = data.reshape(-1, image_size * image_size)
            # real_data = data.to(device)

            # Compute loss for generator
            # a_fake = torch.randint(0, 4, (16, 9, 9))
            # x_fake = torch.randint(0, 4, (16, 9))

            self.fake_atom = self.fake_atom.reshape(16, 9, 9).long()        # Nodes
            self.fake_bond = self.fake_bond.reshape(16, 9).long()           # Adjacency
            self.fake_atom = self.label2onehot(self.fake_atom, self.m_dim)  # a_faketensor
            self.fake_bond = self.label2onehot(self.fake_bond, self.b_dim)  # x_faketensor
            logits_fake, features_fake = self.D(self.fake_atom, None, self.fake_bond)
            d_loss_fake = torch.mean(logits_fake)

            """ Training discriminator """

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(device)
            x_int0 = (eps * a_tensor + (1. - eps) * self.fake_atom).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * self.fake_bond).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            # Backward and optimize.
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            # self.reset_grad()
            # d_loss_fake.backward(retain_graph=True)
            # self.Ga_optimizer.step()

            """End of discriminator Training """

            #discriminator.zero_grad()
            #outD_real = logits_real
            #outD_real = discriminator(real_data).view(-1)
            #outD_fake = self.D(fake_data.detach()).view(-1)
            #errD_real = criterion(outD_real, real_labels)
            #errD_fake = criterion(outD_fake, fake_labels)

            # Propagate gradients
            #errD_real.backward()
            #errD_fake.backward()

            #errD = errD_real + errD_fake`
            #optD.step()

            """Training the generator"""

            #generator.zero_grad()
            #outD_fake = discriminator(fake_data).view(-1)
            #errG = criterion(outD_fake, real_labels)
            #errG.backward()
            #optG.step()

            # Real Reward
            reward_real = torch.from_numpy(self.reward(mols)).to(device)

            # Fake Reward
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(self.fake_bond, self.fake_atom)]
            reward_fake = torch.from_numpy(self.reward(mols)).to(device)

            # Value loss
            value_logit_real, _ = self.D(a_tensor, None, x_tensor, torch.sigmoid)
            value_logit_fake, _ = self.D(self.fake_atom, None, self.fake_bond, torch.sigmoid)
            g_loss_value = torch.mean((value_logit_real - reward_real) ** 2 + (
                    value_logit_fake - reward_fake) ** 2)

            # rl_loss= -value_logit_fake
            # f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

            # Backward and optimize.
            g_loss = -d_loss_fake + g_loss_value
            self.reset_grad()
            g_loss.backward()
            self.Ga_optimizer.step()
            self.Gx_optimizer.step()

            #counter += 1

            # Show loss values
            if i % 10 == 0:
                print(f'Iteration: {i}, Gradient Penalty: {d_loss:0.3f}')
                test_atom = self.generator_a(self.a_fixed_noise).view(a_tensor.shape).cpu().detach()
                test_bond = self.generator_x(self.x_fixed_noise).view(x_tensor.shape).cpu().detach()

                # Save molecules every 50 iterations
                if i % 50 == 0:
                    results_a.append(test_atom)
                    results_x.append(test_bond)
                    mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                            for e_, n_ in zip(results_a, results_x)]
                    DrawMols(mols, 'training_iteration-'+str(i)+'_mol.png')
    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch()
            logits_fake, features_fake = self.D(self.fake_atom, None, self.fake_bond)
            g_loss_fake = - torch.mean(logits_fake)

            print("Loss: ", g_loss_fake)

            # Fake Reward
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(self.fake_atom, self.fake_bond)]

        DrawMols(mols, 'test_mol.png')

    @staticmethod
    def label2onehot(labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    @staticmethod
    def gradient_penalty(y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        # self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def reward(self, mols):
        rr = 1
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        # G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        # V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))
        # self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))


    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]
######################################################################
#.. note::
#
#    You may have noticed ``errG = criterion(outD_fake, real_labels)`` and
#    wondered why we don’t use ``fake_labels`` instead of ``real_labels``.
#    However, this is simply a trick to be able to use the same
#    ``criterion`` function for both the generator and discriminator.
#    Using ``real_labels`` forces the generator loss function to use the
#    :math:`\log(D(G(z))` term instead of the :math:`\log(1 - D(G(z))` term
#    of the binary cross entropy loss function.
#

######################################################################
# Finally, we plot how the generated images evolved throughout training.

# fig = plt.figure(figsize=(10, 5))
# outer = gridspec.GridSpec(5, 2, wspace=0.1)
# for i, images in enumerate(results):
#     inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
#                     subplot_spec=outer[i])
#
#     images = torch.squeeze(images, dim=1)
#     for j, im in enumerate(images):
#
#         ax = plt.Subplot(fig, inner[j])
#         ax.imshow(im.numpy(), cmap="gray")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if j==0:
#             ax.set_title(f'Iteration {50+i*50}', loc='left')
#         fig.add_subplot(ax)
#
# plt.show()


######################################################################
# Acknowledgements
# ~~~~~~~~~~~~~~~~
#


######################################################################
# Many thanks to Karolis Špukas who I co-developed much of the code with.
# I also extend my thanks to Dr. Yuxuan Du for answering my questions
# regarding his paper. I am also indebited to the Pennylane community for
# their help over the past few years.
#


######################################################################
# References
# ~~~~~~~~~~
#


######################################################################
# [1] Ian J. Goodfellow et al. *Generative Adversarial Networks*.
# `arXiv:1406.2661 <https://arxiv.org/abs/1406.2661>`__ (2014).
#
# [2] He-Liang Huang et al. *Experimental Quantum Generative Adversarial
# Networks for Image Generation*.
# `arXiv:2010.06201 <https://arxiv.org/abs/2010.06201>`__ (2020).
#
