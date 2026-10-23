import numpy as np
import igraph as ig
import pyvis.network as net


class factor:
    def __init__(self, variables=None, distribution=None):
        if (distribution is None) and (variables is not None):
            self.__set_data(np.array(variables), None, None)
        elif (variables is None) or (len(variables) != len(distribution.shape)):
            raise Exception('Data is incorrect')
        else:
            self.__set_data(np.array(variables),
                            np.array(distribution),
                            np.array(distribution.shape))

    def __set_data(self, variables, distribution, shape):
        self.__variables = variables
        self.__distribution = distribution
        self.__shape = shape

    # ------------------------ Info ------------------------
    def is_none(self):
        return True if self.__distribution is None else False

    # ----------------------- Getters -----------------------
    def get_variables(self):
        return self.__variables

    def get_distribution(self):
        return self.__distribution

    def get_shape(self):
        return self.__shape


def factor_product(x, y):
    if x.is_none() or y.is_none():
        raise Exception('One of the factors is None')

    xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d(x.get_variables(), y.get_variables(), return_indices=True)

    if xy.size == 0:
        raise Exception('Factors do not have common variables')

    if not np.all(x.get_shape()[xy_in_x_ind] == y.get_shape()[xy_in_y_ind]):
        raise Exception('Common variables have different order')

    # Get the variables that are in x but not in y, ordered according to x
    x_not_in_y = np.setdiff1d(x.get_variables(), y.get_variables(), assume_unique=True)
    y_not_in_x = np.setdiff1d(y.get_variables(), x.get_variables(), assume_unique=True)

    # Generate masks for variables that are in x and y but not in their intersection
    # When invert=True, returns True if the element in a does not appear in b
    x_mask = np.isin(x.get_variables(), xy, invert=True)
    y_mask = np.isin(y.get_variables(), xy, invert=True)

    x_ind = np.array([-1] * len(x.get_variables()), dtype=int)
    y_ind = np.array([-1] * len(y.get_variables()), dtype=int)

    x_ind[x_mask] = np.arange(np.sum(x_mask))
    y_ind[y_mask] = np.arange(np.sum(y_mask)) + np.sum(np.invert(y_mask))

    x_ind[xy_in_x_ind] = np.arange(len(xy)) + np.sum(x_mask)
    y_ind[xy_in_y_ind] = np.arange(len(xy))

    x_distribution = np.moveaxis(x.get_distribution(), range(len(x_ind)), x_ind)
    y_distribution = np.moveaxis(y.get_distribution(), range(len(y_ind)), y_ind)

    res_distribution = x_distribution[tuple([slice(None)] * len(x.get_variables()) + [None] * len(y_not_in_x))] \
                       * y_distribution[tuple([None] * len(x_not_in_y) + [slice(None)])]

    return factor(list(x_not_in_y) + list(xy) + list(y_not_in_x), res_distribution)


def factor_marginalization(x, variables):
    variables = np.array(variables)

    if x.is_none():
        raise Exception('Factor is None')

    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')

    res_variables = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.sum(x.get_distribution(), tuple(np.where(np.isin(x.get_variables(), variables))[0]))

    return factor(res_variables, res_distribution)


def factor_reduction(x, variable, value):
    if x.is_none() or (variable is None) or (value is None):
        raise Exception('Input is None')

    if not np.any(variable == x.get_variables()):
        raise Exception('Factor do not contain given variable')

    if value >= x.get_shape()[np.where(variable == x.get_variables())[0]]:
        raise Exception('Incorrect value of given variable')

    res_variables = np.setdiff1d(x.get_variables(), variable, assume_unique=True)
    res_distribution = np.take(x.get_distribution(), value, int(np.where(variable == x.get_variables())[0]))

    return factor(res_variables, res_distribution)


def joint_distribution(ar):
    for element in ar:
        if element.is_none():
            raise Exception('Factor is None')

    res = ar[0]
    for element in ar[1:]:
        res = factor_product(res, element)

    return res


class factor_graph:
    def __init__(self):
        self._graph = ig.Graph()

    # ----------------------- Factor node functions -----------------------
    def add_factor_node(self, f_name, factor_):
        if (self.get_node_status(f_name) != False) or (f_name in factor_.get_variables()):
            raise Exception('Invalid factor name')
        if type(factor_) is not factor:
            raise Exception('Invalid factor_')
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == 'factor':
                raise Exception('Invalid factor')

        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 1)
        # Create variables
        for v_name in factor_.get_variables():
            if not self.get_node_status(v_name):
                self.__create_variable_node(v_name)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Add node and corresponding edges
        self.__create_factor_node(f_name, factor_)

    def change_factor_distribution(self, f_name, factor_):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        if set(factor_.get_variables()) != set(self._graph.vs[self._graph.neighbors(f_name)]['name']):
            raise Exception('invalid factor distribution')

        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 0)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Set data
        self._graph.vs.find(name=f_name)['factor_'] = factor_

    def remove_factor(self, f_name, remove_zero_degree=False):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')

        neighbors = self._graph.neighbors(f_name, mode="out")
        self._graph.delete_vertices(f_name)

        if remove_zero_degree:
            for v_name in neighbors:
                if self._graph.vs.find(v_name).degree() == 0:
                    self.remove_variable(v_name)

    def __create_factor_node(self, f_name, factor_):
        # Create node
        self._graph.add_vertex(f_name)
        self._graph.vs.find(name=f_name)['is_factor'] = True
        self._graph.vs.find(name=f_name)['factor_'] = factor_

        # Create corresponding edges
        start = self._graph.vs.find(name=f_name).index
        edge_list = [tuple([start, self._graph.vs.find(name=i).index]) for i in factor_.get_variables()]
        self._graph.add_edges(edge_list)

    # ----------------------- Rank functions -----------------------
    def __check_variable_ranks(self, f_name, factor_, allowded_v_degree):
        for counter, v_name in enumerate(factor_.get_variables()):
            if (self.get_node_status(v_name) == 'variable') and (not factor_.is_none()):
                if (self._graph.vs.find(name=v_name)['rank'] != factor_.get_shape()[counter]) \
                        and (self._graph.vs.find(name=v_name)['rank'] is not None) \
                        and (self._graph.vs.find(v_name).degree() > allowded_v_degree):
                    raise Exception('Invalid shape of factor_')

    def __set_variable_ranks(self, f_name, factor_):
        for counter, v_name in enumerate(factor_.get_variables()):
            if factor_.is_none():
                self._graph.vs.find(name=v_name)['rank'] = None
            else:
                self._graph.vs.find(name=v_name)['rank'] = factor_.get_shape()[counter]

    # ----------------------- Variable node functions -----------------------
    def add_variable_node(self, v_name):
        if self.get_node_status(v_name):
            raise Exception('Node already exists')
        self.__create_variable_node(v_name)

    def remove_variable(self, v_name):
        if self.get_node_status(v_name) != 'variable':
            raise Exception('Invalid variable name')
        if self._graph.vs.find(v_name).degree() != 0:
            raise Exception('Can not delete variables with degree >0')
        self._graph.delete_vertices(self._graph.vs.find(v_name).index)

    def __create_variable_node(self, v_name, rank=None):
        self._graph.add_vertex(v_name)
        self._graph.vs.find(name=v_name)['is_factor'] = False
        self._graph.vs.find(name=v_name)['rank'] = rank

    # ------------------------ Info ------------------------
    def get_node_status(self, name):
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            if self._graph.vs.find(name=name)['is_factor']:
                return 'factor'
            else:
                return 'variable'

    # ------------------ Graph structure ------------------
    def get_graph(self):
        return self._graph

    def is_connected(self):
        return self._graph.is_connected()

    def is_loop(self):
        return any(self._graph.is_loop())


def string2factor_graph(str_):
    res_factor_graph = factor_graph()

    str_ = [i.split('(') for i in str_.split(')') if i != '']
    for i in range(len(str_)):
        str_[i][1] = str_[i][1].split(',')

    for i in str_:
        res_factor_graph.add_factor_node(i[0], factor(i[1]))

    return res_factor_graph


def plot_factor_graph(x):
    graph = net.Network(notebook=True, width="100%")
    graph.toggle_physics(False)

    # Vertices
    label = x.get_graph().vs['name']
    color = ['#2E2E2E' if i is True else '#F2F2F2' for i in x.get_graph().vs['is_factor']]
    graph.add_nodes(range(len(x.get_graph().vs)), label=label, color=color)

    # Edges
    graph.add_edges(x.get_graph().get_edgelist())

    return graph.show("./img/3/graph.html")


class llr_belief_propagation:
    def __init__(self, pgm, use_min_sum=False, alpha=0.7):
        """
        Full LLR belief propagation
        pgm: Factor graph object, containing two types of factors:
            1. Parity-check factors
            2. Unary factors that will inject the initial LLR
        """
        self.__t = 0
        self.__pgm = pgm
        self.use_min_sum = use_min_sum
        self.alpha = alpha

        # Message dictionary: (Node A, Node B) -> LLR value
        self.__msg = {}

        graph = self.__pgm.get_graph()
        for edge in graph.es:
            u, v = edge.tuple
            u_name, v_name = graph.vs[u]['name'], graph.vs[v]['name']
            self.__msg[(u_name, v_name)] = 0.0
            self.__msg[(v_name, u_name)] = 0.0

        self.__initial_llrs = self.__extract_initial_llrs()

        self.__var_neighbors = {}
        for var in [v for v in graph.vs if not v['is_factor']]:
            var_name = var['name']
            self.__var_neighbors[var_name] = [
                graph.vs[neighbor]['name']
                for neighbor in graph.neighbors(var)
            ]

        self.__factor_neighbors = {}
        for factor in [v for v in graph.vs if v['is_factor']]:
            factor_name = factor['name']
            self.__factor_neighbors[factor_name] = [
                graph.vs[neighbor]['name']
                for neighbor in graph.neighbors(factor)
            ]

    def __extract_initial_llrs(self):
        """Extract initial LLRs from unary factors"""
        initial_llrs = {}
        graph = self.__pgm.get_graph()

        for node in graph.vs:
            if node['is_factor']:
                factor_obj = node['factor_']
                vars_in_factor = factor_obj.get_variables()

                if len(vars_in_factor) == 1:
                    var_name = vars_in_factor[0]
                    prob_dist = factor_obj.get_distribution()

                    if len(prob_dist) == 2:
                        p0, p1 = prob_dist[0], prob_dist[1]
                        # Add epsilon to prevent division by zero
                        epsilon = 1e-10
                        p0 = max(p0, epsilon)
                        p1 = max(p1, epsilon)
                        p0, p1 = p0 / (p0 + p1), p1 / (p0 + p1)

                        # Convert to LLR: log(P(1)/P(0))
                        if p0 > 0:
                            llr = np.log(p1 / p0)
                        else:
                            llr = 10.0 if p1 > 0 else 0.0

                        initial_llrs[var_name] = llr

        return initial_llrs

    def __compute_variable_to_factor_msg(self, v_name, f_name):
        """Compute message from variable -> factor: LLR_init(v) + Σ(messages from other factors)"""
        msg_sum = self.__initial_llrs.get(v_name, 0.0)

        for neighbor in self.__var_neighbors[v_name]:
            if neighbor != f_name:
                msg_sum += self.__msg.get((neighbor, v_name), 0.0)

        return msg_sum

    def __compute_factor_to_variable_msg(self, f_name, v_name):
        if self.use_min_sum:
            return self.__compute_factor_to_variable_msg_min_sum(f_name, v_name, self.alpha)

        """Compute message from factor -> variable"""
        graph = self.__pgm.get_graph()
        factor_node = graph.vs.find(name=f_name)
        factor_obj = factor_node['factor_']

        factor_vars = factor_obj.get_variables()

        if len(factor_vars) == 1:
            var_name = factor_vars[0]
            return self.__initial_llrs.get(var_name, 0.0)

        incoming_llrs = []
        for var in factor_vars:
            if var != v_name:
                llr = self.__msg.get((var, f_name), 0.0)
                incoming_llrs.append(llr)

        if len(factor_vars) == 2:
            # Binary parity-check factor A⊕B=0
            dist_matrix = factor_obj.get_distribution().reshape(2, 2)
            if np.array_equal(dist_matrix, np.array([[1, 0], [0, 1]])):
                return incoming_llrs[0] if incoming_llrs else 0.0
            elif np.array_equal(dist_matrix, np.array([[0, 1], [1, 0]])):
                return -incoming_llrs[0] if incoming_llrs else 0.0

        elif len(factor_vars) == 3:
            # Ternary parity-check factor A⊕B⊕C=0
            if len(incoming_llrs) == 2:
                L1, L2 = incoming_llrs
                product = np.tanh(L1 / 2.0) * np.tanh(L2 / 2.0)
                product = np.clip(product, -0.999999, 0.999999)
                return 2.0 * np.arctanh(product)

        return 0.0

    def __compute_factor_to_variable_msg_min_sum(self, f_name, v_name, alpha):
        """Implementation of the min-sum algorithm"""
        graph = self.__pgm.get_graph()
        factor_node = graph.vs.find(name=f_name)
        factor_obj = factor_node['factor_']
        factor_vars = factor_obj.get_variables()

        if len(factor_vars) == 1:
            var_name = factor_vars[0]
            return self.__initial_llrs.get(var_name, 0.0)

        incoming_llrs = []
        for var in factor_vars:
            if var != v_name:
                llr = self.__msg.get((var, f_name), 0.0)
                incoming_llrs.append(llr)

        if not incoming_llrs:
            return 0.0

        # ===== Core min-sum calculation (with normalization) =====
        # 1. Compute the sign product
        sign_product = 1.0
        for llr in incoming_llrs:
            if llr >= 0:
                sign_product *= 1.0
            else:
                sign_product *= -1.0

        # 2. Compute the minimum magnitude
        magnitudes = [abs(llr) for llr in incoming_llrs]
        min_magnitude = min(magnitudes)

        return alpha * sign_product * min_magnitude

    def belief(self, v_name, num_iter):
        """Compute the posterior probability distribution of a variable"""
        if self.__t < num_iter:
            self.__loop(num_iter)

        posterior_llr = self.__initial_llrs.get(v_name, 0.0)
        for neighbor in self.__var_neighbors[v_name]:
            posterior_llr += self.__msg.get((neighbor, v_name), 0.0)

        p1 = 1.0 / (1.0 + np.exp(-posterior_llr))
        p0 = 1.0 - p1

        class ResultWrapper:
            def __init__(self, p0, p1):
                self.distribution = np.array([p0, p1])

            def get_distribution(self):
                return self.distribution

            def reshape(self, shape):
                return self.distribution.reshape(shape)

        return ResultWrapper(p0, p1)

    def __loop(self, num_iter):
        graph = self.__pgm.get_graph()

        while self.__t < num_iter:
            new_msg = {}

            for edge in graph.es:
                u, v = edge.tuple
                u_name = graph.vs[u]['name']
                v_name = graph.vs[v]['name']
                u_is_factor = graph.vs[u]['is_factor']

                if u_is_factor:
                    new_msg[(u_name, v_name)] = self.__compute_factor_to_variable_msg(u_name, v_name)
                    new_msg[(v_name, u_name)] = self.__compute_variable_to_factor_msg(v_name, u_name)
                else:
                    new_msg[(u_name, v_name)] = self.__compute_variable_to_factor_msg(u_name, v_name)
                    new_msg[(v_name, u_name)] = self.__compute_factor_to_variable_msg(v_name, u_name)

            self.__msg.update(new_msg)
            self.__t += 1

        return self.__t
