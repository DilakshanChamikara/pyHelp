from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Circuit 1
circuit1 = QuantumCircuit(2, 2)
circuit1.h(0)
circuit1.cx(0, 1)
circuit1.measure([0, 1], [0, 1])
print(circuit1)

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

# Using IBM Quantum Hardware
service = QiskitRuntimeService(
    channel = "ibm_quantum_platform",
    token = "YOUR_API_TOKEN" # paste your api token
)

# QPUs
backend_ibm_fez = service.backend("ibm_fez")
backend_ibm_torino = service . backend ("ibm_torino")

# Transpile the circuit
transpiled_circuit = transpile(circuit1, backend_ibm_fez)
print(transpiled_circuit)

# Sample
sampler = Sampler (mode = backend_ibm_fez)
n_shots = 1024
sampler.options.default_shots = n_shots
result = sampler.run([transpiled_circuit]).result()

# Extract Information ( counts and probability distribution )
counts = result[0].data.c.get_counts() # Use this line if you use the measure method passing register numbers .
# counts = result [0]. data . meas . get_counts () # Use this line if you use the method measure_all ()
probs = {key:value / n_shots for key, value in counts.items()}
print("Counts : ", counts)
print("Probs : ", probs)
