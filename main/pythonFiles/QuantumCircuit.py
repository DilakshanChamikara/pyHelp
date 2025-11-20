from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
# Import the transpile module
from qiskit import transpile
# Define the backend AerSimulator
from qiskit_aer import AerSimulator

# Circuit 1
circuit1 = QuantumCircuit(2, 2)
circuit1.h(0)
circuit1.cx(0, 1)
circuit1.measure([0, 1], [0, 1])
print(circuit1)
# circuit.draw("mpl")

# Defines the backend and runs the simulation
backend = BasicSimulator()
n_shots = 1024 # Default number of shots is 1024
result = backend.run(circuit1, shots = n_shots).result()
# Extract counts and probability distribution
counts = result.get_counts()
prob = { key : value/n_shots for key, value in counts.items()}
print(" BASIC SIMULATION")
print (" Counts: ", counts)
print (" Probabilities: ", prob, "\n")

# Using Aer
backend = AerSimulator()
# Transpile the circuit to a set of gates
# compatible with the backend
compiled_circuit = transpile(circuit1, backend)
# Execute the circuit on the qasm simulator .
n_shots = 1024 # default number of shots .
job_sim = backend.run(compiled_circuit, shots = n_shots)
# Extract Results
result_sim = job_sim.result()
counts = result_sim.get_counts(compiled_circuit)
probs = {key:value/n_shots for key, value in counts.items()}
print(" AER SIMULATION")
print(" Counts ", counts)
print(" Probabilities :", probs)

# Circuit 2
circuit2 = QuantumCircuit(3, 3)
circuit2.h(0)
circuit2.cx(0, 1)
circuit2.cx(1, 2)
circuit2.measure([0, 1, 2], [0, 1, 2])
print(circuit2)

# Circuit 3
def generalized_GHZ_state(n):
    circuit = QuantumCircuit(n, n)
    circuit.h(0)
    circuit.barrier() # Draw a barrier
    for i in range (n -1):
        circuit.cx(i, i +1)
    circuit.barrier()
    measured_qubits = [i for i in range(n)]
    classical_results = [i for i in range(n)]
    circuit.measure(measured_qubits, classical_results)
    return circuit
n = 4
circuit3 = generalized_GHZ_state(n)
print(circuit3)


