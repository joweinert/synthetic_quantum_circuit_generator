from typing import Union, Tuple, List, Dict
import time
import numpy as np
import random
from qiskit import QuantumCircuit
import math
import traceback

Gate = Union[int, Tuple[int, int]]   # Define the alias for gate, as it can be a tuple oir an int

class GeneradorCircuitsQuantics():
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _get_number_of_qubits(self, probabilities_for_gate: Dict[Gate, float] )-> None:
        """
        Returns the numebr of qubits of the circuit, given the probabilities
        """
        n_qubits = 0
        for q in probabilities_for_gate.keys():
            if type(q) == tuple:
                n_qubits = max(q[0], n_qubits)
                n_qubits = max(q[1], n_qubits)
            else:
                n_qubits = max(q, n_qubits)
        
        n_qubits += 1

        return n_qubits
    
    def _generate_initial_random_gates(self, probabilities_for_gate: Dict[Gate, float], n_gates: int) -> List[Gate]:
        """
        Returns a list of gates based on the probabilities for each gate
        """
        if n_gates <= 0:
            return []

        list_gates = list(probabilities_for_gate.keys())
        index_gates = range(len(list_gates))
        probabilities = np.array(list(probabilities_for_gate.values()))

        index_gates = list(np.random.choice(index_gates, size=n_gates, p=probabilities))
        gates = [list_gates[i] for i in index_gates]

        random.shuffle(gates)
        
        return list(gates)

    def _loss_function(self, gates : List[Gate]):
        """
        Returns the loss function from a given gate order
        """
        circuit_data = self._get_circuit_data(gates)

        #depht loss
        depht = circuit_data["depht"]
        depht_loss = abs(self._desired_depth - depht)

        #distribution loss

        dist_gates_per_slice = circuit_data["histo"]
        histogram_diferences = {k: dist_gates_per_slice[k] - self.desired_dist_gates_per_slice.get(k, 0) for k in dist_gates_per_slice.keys()}

        gates_loss = sum(abs(d) for d in histogram_diferences.values())
        gates_loss += sum(self.desired_dist_gates_per_slice[k] for k in self.desired_dist_gates_per_slice.keys() if k not in dist_gates_per_slice.keys() )



        #normailize
        if self._first_time_loss:
            self._first_time_loss = False
            self._initial_loss_depht = depht_loss
            self._initial_loss_gates = gates_loss
        
        self.normalized_depht_loss = depht_loss/self._initial_loss_depht if self._initial_loss_depht else 0
        self.normalized_gates_loss = gates_loss/self._initial_loss_gates if self._initial_loss_gates else 0

        return (self.normalized_depht_loss + self.normalized_gates_loss)/2
    
    def _get_circuit_data(self, gates : List[Gate]):

        last_used = [0 for _ in range(self.number_of_qubits)]

        gates_in_slice = [0 for i in range(len(gates))] # defineix a quina slice es troba el gate

        for i, gate in enumerate(gates):
            if type(gate) == tuple: # 2-qubit gate
                q1, q2 = gate

                sl = max(last_used[q1], last_used[q2])
                last_used[q1] = sl + 1
                last_used[q2] = sl + 1
                gates_in_slice[i] = sl + 1       


            else: # 1-qubit gate
                q = gate
                sl = last_used[q] 
                last_used[q]  = sl +1
                gates_in_slice[i] = sl + 1
                #print(last_used[q])

        #els primers gates es troben en l'slice 1

        depht = max(gates_in_slice)


        density_per_slice = [0]*(max(gates_in_slice) +1) # el index es la slice i el valor es el nombre de gates
        for sl in gates_in_slice:
            density_per_slice[sl] += 1

        gate_density = [ density_per_slice[sl] for sl in gates_in_slice] # el index es el gate i el value es la densitat de la slice en el que esta

        density_per_slice.pop(0) # ja podem eliminar la redundancia del slice 0 amb 0


        
        histo = {} # per cada density, el nombre de slices

        for n_gates in density_per_slice:
            histo[n_gates] = 1 + histo.get(n_gates, 0)



        return {"depht": depht, "histo": histo}
    
    def _get_gates_to_reorder(self, current_gates_to_reorder, ratio_percentage, min_gates_to_reorder, max_gates_to_reorder):

        if ratio_percentage < 20:
            current_gates_to_reorder /= self.ratio_descent_gates_to_reorder#1.5
        elif ratio_percentage > 80:
            current_gates_to_reorder *= 1.25

        return int(max(min_gates_to_reorder,min(current_gates_to_reorder, max_gates_to_reorder)))
      
    def _reorder_gates(self, gates:List[Gate], n:int) -> List[Gate]:
        """
        Returns a new list, reorders n gates from the gates list to a random position
        """
        old_indices = random.sample(range(len(gates)), n)
        new_indices = random.sample(range(len(gates)), n)

        new_gates = ["unset"]*len(gates)

        for new_ind, old_ind in zip(new_indices, old_indices):
            new_gates[new_ind] = gates[old_ind]

        
        new_i = 0
        old_i = 0
        while new_i < len(new_gates):
            if old_i in old_indices: # si el index es dels que hem escollit abans
                old_i += 1
                continue
            if new_gates[new_i] == "unset":
                new_gates[new_i] = gates[old_i]
                old_i += 1
            new_i += 1
        
        return new_gates

    def _generate_circuit_from_gates(self, gates: List[Gate]) -> QuantumCircuit:
        """
        Generates a circuit that has the gates on that order
        """
        qc = QuantumCircuit(self.number_of_qubits)

        for g in gates:
            if type(g) == tuple and len(g) == 2:
                q1, q2 = g
                qc.cz(q1, q2)
            elif type(g) == int:
                qc.h(g)
            else:
                raise TypeError("gate must be either a tuple or an int")
    
        return qc

    def generate_circuit(self, data):
        # temps d'execucio
        ini_time = time.time()

        # Definicio de parametres
        number_of_gates         = data["number_of_gates"]
        self._desired_depth     = data["desired_depth"]
        probabilities_for_gate  = data["probabilities"]
        self.number_of_qubits   = self._get_number_of_qubits(probabilities_for_gate)

        temperature             = data["temperature"]
        alpha                   = data.get("alpha", 1)
        val                     = data.get("val", 0.01)

        self.ratio_descent_gates_to_reorder = data.get("ratio_descent_gates_to_reorder", 1.5)
        self.desired_dist_gates_per_slice =   data.get("desired_dist_gates_per_slice")
        gates_to_reorder_min_percentatge = data.get("gates_to_reorder_min_percentatge", 0.5)

        #Definicio parametres bucle
        iterations_per_step = 500 
        max_steps           = 30000

        #Definicio de llistes per a visualitzar les execucions (les que porten v_ nomes son per a visualitzar)
        losses          = []
        loss_each_step  = []
        end_reason = f"MAX ITERATIONS achieved, steps = {max_steps}, iterations_per_step = {iterations_per_step}"

        # set up variables generacio cricuit
        gates = self._generate_initial_random_gates(probabilities_for_gate, number_of_gates)
        circuit_original = self._generate_circuit_from_gates(gates)
        self._first_time_loss = True # per a normalitzar
        loss  = self._loss_function(gates)
        gates_to_reorder = int(0.5*len(gates)) # initial value
        gates_to_reorder_max = int(0.5*len(gates))
        gates_to_reorder_min = int((gates_to_reorder_min_percentatge/100)*len(gates))

        n = 0
        m = 0
        a = 0
        end_bc_distribution = False

        iteration = 0
        #bucle
        try:
            for step in range(max_steps):
                iterations_accepted = 0
                for _ in range(iterations_per_step):
                    iteration += 1
                    
                    #generem el nou circuit
                    new_gates = self._reorder_gates(gates, gates_to_reorder)
                    new_loss  = self._loss_function(new_gates) 

                    # comparem els circuits
                    if temperature:
                        accept_new_state = new_loss < loss or random.random()< math.exp(-abs(new_loss - loss)/temperature)
                    else:
                        accept_new_state = new_loss < loss 

                    # si acceptem el nou estat
                    if accept_new_state:
                        gates = new_gates
                        loss = new_loss
                        iterations_accepted += 1


                    losses.append(loss)


                    n += 1
                    m += loss
                    a += loss*loss

                    S2 = a - (m*m/n)

                    
                    if n > 10000:#1500:
                        coef1 = math.sqrt(S2/(n*n - n))
                        coef2 = val * m/n
                        if coef1 < coef2:
                            end_bc_distribution = True
                            break


                #Han acabat 500 iteracions, calculem nous parametres
                
                #actualitzem el nombre de gates to reorder
                ratio = 100*iterations_accepted/iterations_per_step
                gates_to_reorder = self._get_gates_to_reorder(gates_to_reorder, ratio, gates_to_reorder_min, gates_to_reorder_max)

                #actualitzem temperatura
                temperature = temperature*alpha

                #comprovem si hem de parar
                if loss == 0:
                    end_reason = "LOSS = 0"
                    break 


                if end_bc_distribution:
                    end_reason = "distribution reached"
                    break


                loss_each_step.append(loss)


        #except Exception as e:  # per si hi ha alguna excepcio al bucle
        #    end_reason = "EXCEPTION"
        #    print(e)
        except Exception:
            traceback.print_exc()
        #end bucle
        circuit = self._generate_circuit_from_gates(gates)
        if self.verbose:
            print(f"Execution time: {time.time()- ini_time}")

        return [circuit_original, circuit], {
            "exec_time": time.time()- ini_time, 
            "final_loss": loss, 
            "end_reason": end_reason, 
            "final_iteration": iteration
            }