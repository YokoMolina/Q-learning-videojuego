
# funcion que ayuda en el decaimineto del epsilon
class LinearDecaySchedule(object):
    def __init__(self, initial_value, final_value, max_steps):
        assert initial_value > final_value, "el valor inicial debe ser estrictamente mayor que el valor final"
        # si se da la condicion sigue el algoritmo
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value-final_value)/max_steps

    # se llamará cuando se llame la funcion epsilon decay
    def __call__(self, step_num):
        current_value = self.initial_value - step_num * self.decay_factor
        if current_value < self.final_value:
            current_value = self.final_value
        return current_value

if __name__ == "__main__": # el script por terminal
    #import matplotlib.pyplot as plt
    epsilon_initial = 1.0
    epsilon_final = 0.005
    MAX_NUM_EPISODE = 100000
    STEPS_PER_EPISODE = 300
    total_steps = MAX_NUM_EPISODE * STEPS_PER_EPISODE
    linear_schedule = LinearDecaySchedule(initial_value = epsilon_initial,
                                                 final_value = epsilon_final,
                                                 max_steps = 0.5 * total_steps)
    epsilons = [linear_schedule(step) for step in range(total_steps)]
    #print("epsilon =", epsilons)
    #plt.plot(epsilons)
    #plt.show()