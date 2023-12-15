import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import execute
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit import Aer, IBMQ
import qiskit

from qiskit_ibm_provider import IBMProvider

from constant import *

from qiskit.providers.aer.noise import NoiseModel


IBMProvider.save_account(token="YOUR_TOKEN", overwrite=True)
provider = IBMQ.load_account()

class QuantumLayer(QuantumCircuit):
    def __init__(self, n_qubits, backend_name, shots):
        self.beta = Parameter("Beta")
        self.gamma = Parameter("Gamma")
        self.shots = shots

        self.backend = Aer.get_backend(backend_name)

        # Adding noise model to the backend
        # bknoise = provider.get_backend('ibmq_mumbai')
        # self.noise_model = NoiseModel.from_backend(bknoise)
        # self.basis_gates = self.noise_model.basis_gates

        self.circuit = self.create_circuit()

    def create_circuit(self):
        ckt = QuantumCircuit(2, 2)
        ckt.ry(self.beta, 0)
        ckt.ry(self.beta, 1)

        ckt.cx(0, 1)
        ckt.rz(-1 * self.gamma, 1)
        ckt.cx(0, 1)
        ckt.measure([0, 1], [0, 1])
        return ckt

    def energy_expectation(self, counts, shots, i, j, Cij=-1):
        expects = 0
        for key in counts.keys():
            perc = counts[key] / shots
            check = Cij * (float(key[i]) - 1 / 2) * (float(key[j]) - 1 / 2) * perc
            expects += check
        return [expects]

    def bind(self, parameters):
        [self.beta, self.gamma] = parameters
        self.circuit.data[0][0]._params = to_numbers(parameters)[0:1]
        self.circuit.data[1][0]._params = to_numbers(parameters)[0:1]
        self.circuit.data[3][0]._params = to_numbers(parameters)[1:2]
        return self.circuit

    def run(self, i):
        self.bind(i)
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            # noise_model=self.noise_model,
            # basis_gates=self.basis_gates, # Uncomment these 2 lines to add noise model
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        return self.energy_expectation(counts, self.shots, 0, 1)
