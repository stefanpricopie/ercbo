import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.distributions import uniform


class comm_erc:
    def __init__(self, p, epoch, order, seed=123, schema=None):
        """Wrap commitment constraint around Problem class

        Args:
            p (Problem): Problem class instance with objective function, bounds and true minimum
            epoch (int): fixed periods of time during which an ERC can be activated
            order (int): number of defining bits
            random_state (int): random state for reproducibility.
            schema (list, optional): Schema of the ERC. Defaults to None.
        """
        self.p = p
        self.epoch = epoch

        if order > p.n_dim:
            raise ValueError("The number of defining bits (order) cannot exceed the dimension of the problem.")
        self.order = order

        # Fix random state
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

        self.active = False     # constraint is inactive at initialization
        self.t = 0              # time step

        self.schema = self.generate_schema() if schema is None else schema

    @property
    def n_dim(self):
        return self.p.n_dim
    
    @property
    def scaler(self):
        s = MinMaxScaler()
        
        # self.p.bounds each row is a dimension range
        # transpose bounds to columns for MinMaxScaler
        s.fit(np.array(self.p.bounds).T)
        
        return s

    @property
    def unconstr_bounds(self):
        return [[0., 1.] for _ in range(self.n_dim)]

    @property
    def constr_bounds(self):
        bounds = []

        for i in range(self.n_dim):
            if self.schema[i] is None:
                bounds.append([0., 1.])
            elif self.schema[i] == 0:
                bounds.append([0., 0.5])
            elif self.schema[i] == 1:
                bounds.append([0.5, 1.])
            else:
                raise ValueError("Schema must be None, 0 or 1")

        return bounds

    @property
    def schema_bounds(self):
        """Returns the bounds of the schema

        Returns:
            list: list of tuples [min, max] for each dimension
        """
        if self.active is False:
            return self.unconstr_bounds

        return self.constr_bounds

    def generate_schema(self):
        """Generates a binary constraint schema of order o
        Returns:
            list: list of length l with None, 0 (and 1)
        """
        bits = np.random.randint(0, 2, self.order)
        none_bits = [None] * (self.n_dim - self.order)
        self.schema = list(bits) + none_bits
        np.random.shuffle(self.schema)
        return self.schema

    def in_schema(self, x):
        """Checks if x belongs to schema

        Args:
            x (list): Float vector of [0,1]^d

        Returns:
            bool: True if all values in schema are equal, False otherwise
        """
        x_std = self.scaler.transform([x])[0]

        # Check if x is between 0 and 1
        if np.any(x_std < 0) or np.any(x_std > 1):
            raise ValueError("X is not standardised between 0 and 1.")

        # check if all values in schema align
        return np.all([b_min <= b <= b_max for b, (b_min, b_max) in zip(x_std, self.constr_bounds)])


    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def sample(self, constrained=None, n_points=1):
        """Samples n_points from the constraint region.

        Args:
            n_points (int, optional): Number of points to sample. Defaults to 1.
            constrained (bool): If True, sample from constraint region, else sample from unconstrained region
                You can sample unconstrained when the constraint is active, as long as the final point is in the constraint region.
                If defaults to None, then it is replaced by self.active
            random_state (int): Random state for reproducibility

        Returns:
            list: List of sampled points
        """
        if constrained is None:
            constrained = self.active
            
        new_points = []
        for _ in range(n_points):
            if constrained:
                # Sample point with contrained bits
                std_x = []

                for bit in self.schema:
                    if bit is None:
                        # Sample from [0, 1]
                        std_x.append(uniform().rvs(random_state=self.rng))
                    elif bit == 0:
                        # Sample from [0, 0.5]
                        std_x.append(uniform(0, 0.5).rvs(random_state=self.rng))
                    elif bit == 1:
                        # Sample from [0.5, 1]
                        std_x.append(uniform(0.5, 0.5).rvs(random_state=self.rng))
                    else:
                        raise ValueError("Schema can only contain 0, 1 or None.")

            else:
                # Sample random point
                std_x = uniform().rvs(size=self.n_dim, random_state=self.rng)
            
            new_points.append(std_x)

        # Inverse transform to original scale
        new_points = self.scaler.inverse_transform(new_points)

        if n_points == 1:
            # If only one point is sampled, return as list
            return new_points[0].tolist()
        else:
            # If multiple points are sampled, return as np.array
            return new_points

    def __str__(self):
        np_bounds = np.array(self.constr_bounds).T
        original_bounds = self.scaler.inverse_transform(np_bounds).T
        constraint_space = "{" + ", ".join([f"{low:g}<=x{i+1}<={high:g}" for i, (low, high) in enumerate(original_bounds)]) + "}"
        return fr"ERC(epoch={self.epoch}, order={self.order}): {constraint_space}"

    def __repr__(self):
        return fr"ERC({self.epoch}, {self.order}): {self.schema}"
