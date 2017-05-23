import numpy as np


class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        # "+ 1" means Insert weights for the bias units into the first row and first column.
        self.weights = 0.1 * np.random.randn(self.num_visible + 1, self.num_hidden + 1)

    def fit(self, data, **kwargs):
        """
        Train the machine.
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.  
          array-like object. (n, n_visible)
        """
        flags = {
            "max_epoch": 5000,
            "display_frequency": None,
            "learning_rate": 0.01,
        }
        flags.update(kwargs)

        num_examples = np.size(data, 0)

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, np.ones(1), axis=1)

        error_list = []

        for epoch in range(1, flags["max_epoch"] + 1):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self.sigmoid(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self.sigmoid(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self.sigmoid(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += flags["learning_rate"] * ((pos_associations - neg_associations) / num_examples)

            error_list.append(np.sum((data - neg_visible_probs) ** 2))
            if flags["display_frequency"] is not None and epoch % flags["display_frequency"] is 0:
                print("Epoch %s: error is %s" % (epoch, np.asscalar(np.mean(error_list))))
                error_list = []

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.
    
        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.
    
        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = np.size(data, 0)

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, np.ones(1), axis=1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self.sigmoid(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    # TODO: Remove the code duplication between this method and `run_visible`?
    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.
        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.
        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, np.ones(1), axis=1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self.sigmoid(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1

        # Ignore the bias units.
        visible_states = visible_states[:, 1:]
        return visible_states

    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.
        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self.sigmoid(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self.sigmoid(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]

    def __str__(self):
        return 'num_visible: %d\nnum_hidden: %d\nweights: \n%s\n' % (self.num_visible, self.num_hidden, self.weights)

    def predict(self, data) -> np.ndarray:
        """
         array-like object (n, num_visible) or (n, num_hidden)
        :return probability of this configuration
        """
        if np.size(data, 1) == self.num_visible:
            configuration = np.expand_dims(np.insert(self.run_visible(np.asarray(data)), 0, np.ones(1), axis=1), 1)  # n * 1 * (h + 1)
            # print("configuration:", np.shape(configuration))
            prob_each_unit = np.sum(configuration * np.expand_dims(self.weights, 0), axis=-1)[:, 1:]  # n * v
        elif np.size(data, 1) == self.num_hidden:
            configuration = np.expand_dims(np.insert(self.run_hidden(np.asarray(data)), 0, np.ones(1), axis=1), 1)  # n * (v + 1) * 1
            prob_each_unit = np.sum(configuration * np.expand_dims(self.weights, 0), axis=-2)[:, 1:]  # n * v
        else:
            raise TypeError("data should be array-like object in shape of (n, num_visible) or (n, num_hidden)")
        # print("prob_each_unit:", np.shape(prob_each_unit))
        # print("prob_each_unit:", prob_each_unit)
        prob = np.prod(self.sigmoid(prob_each_unit), axis=1)  # n * 1
        # print("prob:", np.shape(prob))
        return prob

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return np.ones(1) / (1 + np.exp(-x))


if __name__ == '__main__':
    # example
    import time
    tic = time.time()

    r = RBM(num_visible=6, num_hidden=2)
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
    r.fit(training_data, max_epochs=10000, display_frequency=1000)
    print(r)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))

    toc = time.time()
    print("elapsed time: %s s" % (toc - tic))
