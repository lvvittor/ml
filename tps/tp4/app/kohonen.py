import numpy as np

class Kohonen:
    def __init__(self, k, inputs):
        """
        `inputs` MUST be already standardized, so it has mean=0, std=1
        """
        self.k = k                  # k^2 = amount of neurones (k x k map)
        self.p = inputs.shape[0]    # amount of inputs
        self.n = inputs.shape[1]    # dimensions of each input

        self.inputs = inputs

        # Initialize weights of each neurone with uniform distribution U(0,1).
        # self.weights = np.random.rand(self.k**2, self.n)

        # Initialize weights of each neurone with random samples from the inputs.
        self.weights = np.zeros((self.k**2, self.n))
        for i in range(self.k**2):
            self.weights[i] += self.inputs[np.random.randint(self.p)]

        self.R = self.k / 2   # initial radius of the neighbourhood
        self.eta = 1.0        # initial learning rate


    def train(self, max_epochs):
        for epoch in range(1, max_epochs):
            # Adjust learning rate and radius linearly with the epoch
            eta = self.eta * (1 - epoch/max_epochs)     # 1 to 0
            radius = self.R * (1 - epoch/max_epochs)
            radius = 1 if radius < 1 else radius        # k/2 to 1

            # Get a random input each epoch
            x = self.inputs[np.random.randint(self.p)]
            
            # Get the index of the minimum distance neurone (winner neurone)
            distances = np.linalg.norm(self.weights - x, axis=1) # euclidean distance between `x` and each neurone's weights
            winner_neuron_index = np.argmin(distances)

            # Get the indexes of all the neighbours of the winner neurone (inside the radius `R`)
            winner_neighbours = self.get_neighbours(winner_neuron_index, radius) # includes the winner neurone itself

            # self.log_epoch(epoch, x, eta, radius, distances, winner_neuron_index, winner_neighbours)

            # Update the weights of the winner neurone and its neighbours
            self.weights[winner_neighbours] += eta * (x - self.weights[winner_neighbours])

        return self.weights


    def get_neighbours(self, neuron_index, radius, include_self=True):
        """
        Returns the indexes of all the neighbours of the neuron with index `neuron_index`.
        The neuron with index `neuron_index` is also included by default.
        """

        neighbours = []

        # Think of the neurons' weights as a k x k matrix
        row_i, col_i = divmod(neuron_index, self.k)

        # Optimize the search by only looking at the neighbours inside a square of side length 2R+1
        min_row = max(0, row_i - int(radius))
        max_row = min(self.k - 1, row_i + int(radius))
        min_col = max(0, col_i - int(radius))
        max_col = min(self.k - 1, col_i + int(radius))

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                distance = np.linalg.norm([row - row_i, col - col_i])  # euclidean distance between (row, col) and (row_i, col_i)
                if distance <= radius and (include_self or distance != 0):
                    index = row * self.k + col
                    neighbours.append(index)

        return neighbours
    

    def log_epoch(self, epoch, x, eta, radius, distances, winner_neuron_index, winner_neighbours):
        print(f"\n\n---------------EPOCH {epoch}---------------")
        print("ETA: ", eta)
        print("RADIUS: ", radius)

        print("\nINPUT ; shape=", self.inputs.shape)
        print(x)

        print("\n\nWEIGHTS ; shape: ", self.weights.shape)
        print(self.weights)

        print("\n\nDistance between input and each neurone: ", distances)

        print("\n\nWinner neuron index: ", winner_neuron_index)
        print("Winner neuron: ", self.weights[winner_neuron_index])

        print("\n\nWinner neighbours: ", winner_neighbours)
        print("Winner neighbours weights: ", self.weights[winner_neighbours])

        print("\n\nDelta weights: ", eta * (x - self.weights[winner_neighbours]))

        print("\n\nUpdated weights: ", self.weights[winner_neighbours] + eta * (x - self.weights[winner_neighbours]))
    

    def map_inputs(self, inputs):
        """
        Map each input to its closest neurone, and return the indexes of the neurones for each input.
        """
        # Reshape the inputs and weights to have dimensions (p, 1, n)
        inputs = inputs[:, np.newaxis, :]
        weights = self.weights[np.newaxis, :, :]

        # Compute the euclidean distance between each input and each neurone's weights
        distances = np.linalg.norm(inputs - weights, axis=2)

        # Return the indices of the weights that are closest to each input.
        return np.argmin(distances, axis=1)
    

    def get_heatmap(self, inputs):
        """
        Get the heatmap of the Kohonen map (i.e. how many inputs are mapped to each neurone).
        """
        winner_neurons = self.map_inputs(inputs)

        # Count the occurrences of each neuron index
        counts = np.bincount(winner_neurons)

        # Create the new array with counts
        occurrences = np.zeros(self.weights.shape[0])
        occurrences[:len(counts)] = counts # fill the array with the counts, and leave the rest as 0

        # Reshape the array to have dimensions (k, k)
        return occurrences.reshape(self.k, self.k)


    def get_umatrix(self):
        """
        Get the (k, k) U-matrix of the Kohonen map.
        """
        umatrix = np.zeros((self.k, self.k))

        # Iterate through each neuron
        for i in range(self.k**2):
            # Get the neighbours of the neuron
            neighbours = self.get_neighbours(neuron_index=i, radius=1, include_self=False)
            # Calculate the average [euclidean] distance between the neuron and its neighbours
            distances = np.linalg.norm(self.weights[neighbours] - self.weights[i], axis=1)
            row, col = divmod(i, self.k)
            umatrix[row, col] = np.mean(distances)
        
        # Normalize the U-matrix
        # umatrix = (umatrix - np.min(umatrix)) / (np.max(umatrix) - np.min(umatrix))

        return umatrix
