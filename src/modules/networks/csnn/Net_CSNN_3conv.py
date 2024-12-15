import snntorch as snn
import torch
import torch.nn as nn

from modules.networks.csnn.BaseNet_CSNN import BaseNet_CSNN

class Net_CSNN_3conv(nn.Module, BaseNet_CSNN):
    """
    Convolutional Spiking Neural Network with 3 convolutional layers and 1 fully connected layer.

    Parameters
    ----------
    topology : list
        List of tuples with the topology of the model
    params : dict
        Dictionary with the parameters of the model such as beta, slope, step and threshold
    """
    def __init__(self, topology, params):
        super().__init__()
        self.betas = params['beta']
        self.spike_grad = params['slope']
        self.num_steps = params['step']
        self.thresholds = params['threshold']
        # Layer1
        self.conv1 = nn.Conv1d(topology[0][0], topology[0][1], topology[0][2])
        self.mp1 = nn.MaxPool1d(2)
        self.lif1 = snn.Leaky(beta=self.betas[0], spike_grad=self.spike_grad, threshold=self.thresholds[0], learn_beta=True, learn_threshold=True)
        # Layer2
        self.conv2 = nn.Conv1d(topology[1][0], topology[1][1], topology[1][2])
        self.mp2 = nn.MaxPool1d(2)
        self.lif2 = snn.Leaky(beta=self.betas[1], spike_grad=self.spike_grad, threshold=self.thresholds[1], learn_beta=True, learn_threshold=True)
        # Layer3
        self.conv3 = nn.Conv1d(topology[2][0], topology[2][1], topology[2][2])
        self.mp3 = nn.MaxPool1d(2)
        self.lif3 = snn.Leaky(beta=self.betas[2], spike_grad=self.spike_grad, threshold=self.thresholds[2], learn_beta=True, learn_threshold=True)
        # Layer4
        self.fc1 = nn.Linear(topology[3][0], topology[3][1])
        self.lif4 = snn.Leaky(beta=self.betas[3], spike_grad=self.spike_grad, threshold=self.thresholds[3], learn_beta=True, learn_threshold=True, output=True)
        # Architecture
        self.architecture = f"""
        =====================
        Conv1d({topology[0]})
        MaxPool1d(2)
        Leaky({self.betas[0]}, {self.thresholds[0]})
        ---
        Conv1d({topology[1]})
        MaxPool1d(2)
        Leaky({self.betas[1]}, {self.thresholds[1]})
        ---
        Conv1d({topology[2]})
        MaxPool1d(2)
        Leaky({self.betas[2]}, {self.thresholds[2]})
        ---
        Linear({topology[3]})
        Leaky({self.betas[3]}, {self.thresholds[3]}, output=True)
        =====================
        """

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        tuple
            Tuple with the current, spike and membrane potential of the last layer
        """
        cur_last_rec = []
        spk_last_rec = []
        mem_last_rec = []
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem4 = self.lif4.reset_mem()
        for _ in range(self.num_steps):
            # Layer1
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer2
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)   
            # Layer3 
            cur3 = self.mp3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            # Layer4
            cur4 = self.fc1(spk3.flatten(1))
            spk4, mem4 = self.lif4(cur4, mem4)     
            # Record the final layer
            cur_last_rec.append(cur4)
            spk_last_rec.append(spk4)
            mem_last_rec.append(mem4)
        return torch.stack(cur_last_rec), torch.stack(spk_last_rec), torch.stack(mem_last_rec)
        
    
if __name__ == "__main__":
    SM_2H_K3 = [(1, 16, 3), (16, 32, 3), (32, 64, 3), (128, 2)]
    SM_2H_K5 = [(1, 16, 5), (16, 32, 5), (32, 64, 5), (128, 2)]
    MM_2H_K3 = [(1, 32, 3), (32, 64, 3), (64, 128, 3), (256, 2)]
    MM_2H_K4 = [(1, 32, 4), (32, 64, 4), (64, 128, 4), (256, 2)]
    LM_2H_K3 = [(1, 64, 3), (64, 128, 3), (128, 256, 3), (512, 2)]
    LM_2H_K4 = [(1, 64, 4), (64, 128, 4), (128, 256, 4), (512, 2)]
    TOPOLOGIES = [SM_2H_K3, SM_2H_K5, MM_2H_K3, MM_2H_K4, LM_2H_K3, LM_2H_K4]
    PARAMETERS = {
        'beta': [0.9, 0.9, 0.9, 0.9],
        'slope': 100.0,
        'step': 100,
        'threshold': [1.0, 1.0, 1.0, 1.0]
    }
    for topology in TOPOLOGIES:
        model = Net_CSNN_3conv(
            topology=topology,
            params=PARAMETERS
        )
        print(model.get_num_params())
        print(model.get_architecture())