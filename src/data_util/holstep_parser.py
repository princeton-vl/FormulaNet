'''This utility converts HolStep database into graph representation.
'''
from __future__ import division
from __future__ import print_function

import sys
import re

from formula import NodeType
from formula import Node

DEBUG = True

TOKEN_RE = re.compile(r'[(),]|[^\s(),]+')
QUANTIFIER_RE = re.compile(r"^([!?\\@]|\?!|lambda)([a-zA-Z0-9$%_'<>]+)\.$")
STRIP_TOKEN_RE = re.compile(r'\s(\*|part|\/)')
INFIX_OP = {
    "=", "/\\", "==>", "\\/", "o", ",", "+", "*", "EXP", "<=", "<", ">=", ">", "-",
    "DIV", "MOD", "treal_add", "treal_mul", "treal_le", "treal_eq", "/", "|-", "pow",
    "div", "rem", "==", "divides", "IN", "INSERT", "UNION", "INTER", "DIFF", "DELETE",
    "SUBSET", "PSUBSET", "HAS_SIZE", "CROSS", "<=_c", "<_c", "=_c", ">=_c", ">_c", "..",
    "$", "PCROSS"
}
# lambda and \ are semantically the same, but different in syntax (token). Thus
# cannot simply substitute lambda with \ in parsing lambda is changed to !lambda to solve name conflict with function
# Same for !, ?, ?!, @
FO_QUANT = {'!!', '!?', '!?!'}
HOL_QUANT = {'!@', '!\\', '!lambda'}
QUANTIFIER = FO_QUANT | HOL_QUANT
OPS = QUANTIFIER | INFIX_OP

# Only |- can take variable number of inputs


class Tokenization(object):
    '''A helper class to read tokenization.

    Parameters
    ----------
    tokenization : str
        Tokenization string
    '''

    def __init__(self, tokenization):
        self.i = -1  # index
        self.tokens = tokenization.split()

    def next(self):
        '''Read next token in tokenization'''
        self.i += 1
        return self.tokens[self.i]


class Scope(object):
    '''Recursive scope manager

    Use with statement to manage scope automatically.
    '''

    def __init__(self):
        self.dict = {}
        self.bounded_no = 0  # key in the dict for next bounded var.
        self.bounded_stack = []

    def __enter__(self):
        self.bounded_stack.append(0)

    def __exit__(self, exc_type, exc_value, traceback):
        self.bounded_no -= self.bounded_stack[-1]
        self.bounded_stack.pop()

    def query(self, name):
        '''Query if value of name is in current scope.

        Parameters
        ----------
        name : str
            Name being queried.

        Returns
        -------
        The corresponding object saved by the name in the scope. Return None if
        not found.
        '''
        if isinstance(name, int) and name >= self.bounded_no:
            return None
        elif name in self.dict:
            return self.dict[name]
        else:
            return None

    def declare_var(self, quant):
        '''Declare a name assoicated with a given quant

        Parameters
        ----------
        quant : Quant
            Quantifier that a name is linked with
        '''
        # No other constant has just digit name
        self.dict[self.bounded_no] = quant
        self.bounded_stack[-1] += 1
        self.bounded_no += 1

    def declare_const(self, name, node):
        '''Declare a value constant in current scope with given name.

        Parameters
        ----------
        name : str
            Name of the new value variable
        node
            Variable the name associated with.
        '''
        assert name not in self.dict, "Replicated name: {}".format(name)
        self.dict[name] = node

    @property
    def depth(self):
        '''Length of the program stack'''
        return len(self.stack)


class FormulaToGraph(object):
    '''A handy converter that converts formula to graph.'''

    def __init__(self):
        self.scope = Scope()
        self.graph = []
        self.root = None
        self.tail = None  # Tail for newly added forall quantifiers
        self.debug = False  # Switch for debug output
        self.rename = False

    def _create_var(self, quant, name):
        '''Create and return a Var node
        This function won't declare the variable!

        Parameters
        ----------
        quant : Quant
            Quantifier that quantifies this variable.

        Returns
        -------
        Node
            A newly created variable node
        '''
        assert quant != None
        if not self.debug and self.rename:
            name = 'VAR'
        else:
            name = name.split(':')[0]
        node = Node(NodeType.VAR, name)
        node.quant = quant
        quant.vvalue = node
        self.graph.append(node)
        return node

    def _create_varfunc(self, quant, name):
        assert quant != None
        if not self.debug and self.rename:
            name = 'VARFUNC'
        else:
            name = name.split(':')[0]
        node = Node(NodeType.VARFUNC, name)
        node.quant = quant
        quant.vfunc.append(node)
        self.graph.append(node)
        return node

    def _create_const(self, name):
        '''Create and return a Const node

        Parameters
        ----------
        name : str
            Name of the constant.

        Returns
        -------
        Node
            A newly created constant node
        '''
        node = Node(NodeType.CONST, name)
        self.graph.append(node)
        self.scope.declare_const(name, node)
        return node

    def _create_constfunc(self, name):
        node = Node(NodeType.CONSTFUNC, name)
        self.graph.append(node)
        return node

    def _create_quant(self, name, free_var=False):
        '''Create and return a Quant node

        Parameters
        ----------
        name : str
            Name of the quantifier.
        free_var : bool
            True if quant is associated with a free variable

        Returns
        -------
        Node
            A newly created quantifier node
        '''
        node = Node(NodeType.QUANT, name)
        self.graph.append(node)
        if not free_var:
            self.scope.declare_var(node)
        return node

    def _create_global_free(self, name, real_name, is_func=False):
        '''Create global free variable by name'''
        assert name[0] == 'f', name
        name = name + ':f'
        if not is_func:
            quant = self.scope.query(name)
            if quant is not None:
                if quant.vvalue is not None:
                    return quant.vvalue
                else:
                    return self._create_var(quant, real_name)
            else:
                quant = self._create_quant('!!', free_var=True)
                var = self._create_var(quant, real_name)
                self.scope.declare_const(name, quant)
        else:
            quant = self.scope.query(name)
            if quant is not None:
                return self._create_varfunc(quant, real_name)
            else:
                quant = self._create_quant('!!', free_var=True)
                var = self._create_varfunc(quant, real_name)
                self.scope.declare_const(name, quant)

        if self.tail is None:
            self.tail = quant
        else:
            self.root.incoming.append(quant)
            quant.outgoing.append(self.root)
        self.root = quant
        return var

    def _create_var_const_by_name(self, name, is_func=False):
        '''Create a const or node by name.

        Parameters
        ----------
        name : str
            Name of the new node.
        is_func : bool
            A flag indicates whether this node is a function.

        Returns
        -------
        Node
            A node created or retrieved
        '''
        if not is_func:
            r_name, type_info = name.split(':')
            if type_info[0] == 'c':
                node = self.scope.query(name)
                if node is None:
                    node = self._create_const(name)
                return node
            elif type_info[0] == 'b':
                quant = self.scope.query(int(type_info[1:]))
                if quant is not None:
                    if quant.vvalue is not None:
                        return quant.vvalue
                    else:
                        return self._create_var(quant, name)
                else:
                    assert quant is None
            elif type_info[0] == 'f':
                return self._create_global_free(type_info, r_name, is_func=False)
            else:
                assert False, "Invalid type: %s" % type_info
        else:
            r_name, type_info = name.split(':')
            if type_info[0] == 'c':
                node = self._create_constfunc(name)
                return node
            elif type_info[0] == 'b':
                quant = self.scope.query(int(type_info[1:]))
                if quant is not None:
                    return self._create_varfunc(quant, name)
                else:
                    assert quant is None
            elif type_info[0] == 'f':
                return self._create_global_free(type_info, r_name, is_func=True)
            else:
                assert False, 'Invalid type: %s' % name

    def _formula_to_graph(self, formula, parent=None, is_func=False):
        '''Helper func that converts tupled formula parse tree into graph

        Parameters
        ----------
        formula : tuple
            Tupled tree representation of a formula
        parent : list of Vertex or None
            The vertex current formula tuple is connected with
        is_func : bool
            If this formula acts like a function.

        Returns
        -------
        Node
            Root node of the converted graph from the given formula.
        '''
        if isinstance(formula, str):
            node = self._create_var_const_by_name(formula, is_func)
            if parent is not None:
                node.incoming.append(parent)
            return node
        # Now formula is a tuple
        with self.scope:
            if formula[0] in QUANTIFIER:
                assert isinstance(formula[1], str), formula[1]
                quant_name = formula[0]
                if quant_name == '!lambda':
                    quant_name = '!\\'
                quant = self._create_quant(quant_name)
                if parent is not None:
                    quant.incoming.append(parent)
                node = self._formula_to_graph(formula[2], quant, is_func=False)
                quant.outgoing.append(node)
                return quant
            else:
                # assert isinstance(formula[0], str) or formula[0][0] in HOL_QUANT
                node = self._formula_to_graph(formula[0], parent, is_func=True)
                for arg in formula[1:]:
                    arg_node = self._formula_to_graph(arg, node)
                    node.outgoing.append(arg_node)
                return node

    def _finalize_graph(self):
        '''Finalize nodes in the graph by adding nodes in
        self.vfunc/self.vvalue/self.quant to corresponding self.incoming and
        self.outgoing
        '''
        for node in self.graph:
            if node.type in (NodeType.VAR, NodeType.VARFUNC):
                node.incoming.append(node.quant)
            elif node.type == NodeType.QUANT:
                for x in node.vfunc:
                    node.outgoing.append(x)
                if node.vvalue is not None:
                    node.outgoing.append(node.vvalue)

    def convert(self, formula):
        '''Convert tupled formula parse tree into graph

        Parameters
        ----------
        formula : tuple
            Tupled tree representation of a formula

        Returns
        -------
        Graph
            DAG representation of a formula
        '''
        self.scope = Scope()
        self.graph = []
        self.root = None
        self.tail = None
        Node.reset_id()
        append_infer_node = False

        if isinstance(formula, tuple) and len(formula) == 2 and formula[0] == '|-:c':
            formula = formula[1]
            append_infer_node = True
            while isinstance(formula, tuple) and formula[0] == '!!':
                formula = formula[2]

        node = self._formula_to_graph(formula)
        if self.tail is not None:
            self.tail.outgoing.append(node)
            node.incoming.append(self.tail)
        else:
            self.root = node

        if append_infer_node:
            node = Node(NodeType.CONSTFUNC, '|-:c')
            self.graph.append(node)
            node.outgoing.append(self.root)
            self.root.incoming.append(node)
            self.root = node

        if not self.debug:
            self._finalize_graph()

        return self.graph


class FormulaToTree(object):
    '''A handy converter that converts formula to tree.'''

    def __init__(self):
        self.graph = []
        self.debug = False  # Switch for debug output

    def _name_type(self, typed_name):
        '''Get name, type tuple based on typed name

        Parameters
        ----------
        typed_name : str
            Name with type info followed by :

        Returns
        -------
        str, int
            Name, type
        '''
        if typed_name.find(':') == -1:
            return typed_name, NodeType.QUANT
        else:
            r_name, type_info = typed_name.split(':')
            if type_info[0] == 'c':
                return r_name, NodeType.CONST
            elif type_info[0] in ('b', 'f'):
                return r_name, NodeType.VAR
            else:
                assert False, 'Wrong typed name.'

    def _create_node(self, name):
        '''Create and return a node

        Parameters
        ----------
        name : str
            Name of the node

        Returns
        -------
        Node
            A newly created node
        '''
        nname, nodetype = self._name_type(name)
        node = Node(nodetype, nname)
        self.graph.append(node)
        return node

    def _formula_to_tree(self, formula, parent=None):
        '''Helper func that converts tupled formula parse tree into tree

        Parameters
        ----------
        formula : tuple
            Tupled tree representation of a formula
        parent : list of Vertex or None
            The vertex current formula tuple is connected with

        Returns
        -------
        Node
            Root node of the converted tree from the given formula.
        '''
        if isinstance(formula, str):
            if formula == '!lambda':
                formula = '!\\'
            node = self._create_node(formula)
            if parent is not None:
                node.incoming.append(parent)
            return node
        else:  # Now formula is a tuple
            node = self._formula_to_tree(formula[0], parent)
            for arg in formula[1:]:
                arg_node = self._formula_to_tree(arg, node)
                node.outgoing.append(arg_node)
            return node

    def convert(self, formula):
        '''Convert tupled formula parse tree into tree

        Parameters
        ----------
        formula : tuple
            Tupled tree representation of a formula

        Returns
        -------
        Graph
            tree representation of a formula
        '''
        self.graph = []
        Node.reset_id()

        node = self._formula_to_tree(formula)

        return self.graph


def _pack_term(l):
    '''Pack a list of tokens into a term'''
    if len(l) == 2:
        if isinstance(l[0], str):
            match = QUANTIFIER_RE.match(l[0])
            if match:  # if has quantifier
                quantifier, var = match.groups()
                quantifier = '!' + quantifier
                term = (quantifier, var, l[1])
            else:  # if is unary operator, or curried function application
                term = tuple(l)
        else:  # if is curried function.
            assert isinstance(l[0], tuple)
            # If the first is not quantifier \ or @ or infix op, merge curried
            if l[0][0] not in OPS:
                term = l[0] + tuple(l[1:])
            else:
                term = tuple(l)
    elif len(l) == 3:  # Handle infix operation
        assert l[1] in INFIX_OP
        term = (l[1], l[0], l[2])
    elif len(l) > 3 and l[-2] == '|-':
        assert l[1:-2:2] == [',' for _ in range((len(l) - 3) // 2)]
        term = tuple([l[-2]] + l[::2])
    else:
        raise ValueError('Unexpected syntax: {}'.format(l))
    return term


def _get_typed_name(term, token):
    '''Mark type info based on the corresponding single token

    Parameters
    ----------
    token : str
        Token extracted from tokenization.
    '''
    assert isinstance(term, str)
    if token[0] == 'c':
        assert term == token[1:], 'Term: {} ||| token: {}'.format(term, token)
        return term + ':c'
    elif token[0] == 'f':
        return term + ':' + token
    elif token[0] == 'b':
        return term + ':' + token
    else:
        raise ValueError('Unknown token! Term: {} ||| token: {}'.format(term, token))


def _check_quant(term, token):
    '''Check if quantifier is consistant in term and token'''
    if term == '!!':
        return term[1] == token
    elif term in QUANTIFIER:
        return term[1:] == token[1:]
    else:
        return 'c' + term == token


def _repack(formula, token):
    '''Use token to add type info and repack the Formulas

    Parameters
    ----------
    token : Tokenization
        The tokenization object
    '''
    if isinstance(formula, tuple):
        # Ensure every tupled formula has more than one elements.
        assert len(formula) > 1, formula
        if formula[0] in QUANTIFIER:
            assert len(formula) == 3, formula
            # Corresponding token is deleted in clean due to ambiguity
            if formula[0] != '!\\':
                t_quant = token.next()
                assert _check_quant(formula[0],
                                    t_quant), 'Term: {} ||| token: {}'.format(
                                        formula[0], t_quant)
            return (formula[0], formula[1] + ':b', _repack(formula[2], token))
        else:  # Function
            result = []
            for x in formula:
                if isinstance(x, tuple):
                    result.append(_repack(x, token))
                else:
                    result.append(_get_typed_name(x, token.next()))
            return tuple(result)
    else:  # single token
        return _get_typed_name(formula, token.next())


def _fix_omitted_forall(formula, token):
    '''Handle omitted forall in token'''
    if formula[0] == '!!':
        assert len(formula) == 3
        return (formula[0], formula[1] + ':b', _fix_omitted_forall(formula[2], token))
    else:
        return _repack(formula, token)


def _add_type(processed_formula, tokenization):
    '''Add type information right after every constant & variable, separated by #.'''
    tokenization = Tokenization(tokenization)
    if len(processed_formula) == 2:
        assert processed_formula[0] == '|-'
        return ('|-:c', _fix_omitted_forall(processed_formula[1], tokenization))
    else:
        # Case: A, B, C |- D
        # Note in tokenization, it's ==> A ==> B ==> C D
        formula = ['|-:c']
        for i, t in enumerate(processed_formula[1:]):
            if i < len(processed_formula) - 2:
                token = tokenization.next()
            assert token == 'c==>', 'Token is {}'.format(token)
            formula.append(_repack(t, tokenization))
        return tuple(formula)


def parse_formula(formula, tokenization):
    '''Parse HolStep formula to a tupled representation

    Parameters
    ----------
    formula : str
        A single HolStep formula
    tokenization : str
        A corresponding tokenization.

    Returns
    -------
    tuple
        Tupled tree representation of a formula
    '''
    tokens = TOKEN_RE.findall(formula)
    stack = [[]]
    for t in tokens:
        if t == '(':
            stack.append([])
        else:
            if t == ')':
                t = _pack_term(stack.pop())
            stack[-1].append(t)
    assert len(stack) == 1
    processed_formula = _pack_term(stack[0])
    if __name__ == '__main__':
        print(processed_formula)

    tokenization = STRIP_TOKEN_RE.sub('', ' ' + tokenization)  # Pad a space
    typed_formula = _add_type(processed_formula, tokenization)
    return typed_formula


def graph_from_hol_stmt(formula, tokenization):
    '''Parse HolStep formula to a graph.

    Formulas in HolStep are using a dialects of HOL Light. The formula
    provides all necessary parentheses, so that the knowledge of precedence
    is no more needed. Besides formulas, a tokenization is also provided
    for crucial type information.

    Parameters
    ----------
    formula : str
        A single HolStep formula
    tokenization : str
        A corresponding tokenization.

    Returns
    -------
    list of Node
        Graph representation of a formula
    '''
    converter = FormulaToGraph()
    return converter.convert(parse_formula(formula, tokenization))


def tree_from_hol_stmt(formula, tokenization, rename=False, charpool=None, unseen=False):
    '''Parse HolStep formula to a tree

    Formulas in HolStep are using a dialects of HOL Light. The formula
    provides all necessary parentheses, so that the knowledge of precedence
    is no more needed. Besides formulas, a tokenization is also provided
    for crucial type information.

    Parameters
    ----------
    formula : str
        A single HolStep formula
    tokenization : str
        A corresponding tokenization.
    rename : bool
        Randomly shuffle name among all available names is True
    charpool : list
        Set used for picking up random names

    Returns
    -------
    list of Node
        Tree representation of a formula
    '''
    if rename:
        converter = FormulaToGraph()
        stmt =  converter.convert(parse_formula(formula, tokenization))
        return shuffle_tree_name.convert(stmt, charpool, unseen)
    else:
        converter = FormulaToTree()
        return converter.convert(parse_formula(formula, tokenization))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test HolStep dataset parsing.')
    parser.add_argument('files', type=str, help='File to convert', nargs='+')
    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree of graph (default).')
    parser.add_argument('--latex', action='store_true', help='Output latex code')
    args = parser.parse_args()
    if args.format == 'graph':
        converter = FormulaToGraph()
    elif args.format == 'tree':
        converter = FormulaToTree()
    else:
        assert False
    converter.debug = False

    for path in args.files:
        with open(path, 'r') as f:
            for line in f:
                print(line, end='')
                token_line = next(f)[2:]
                print(token_line, end='')
                if line and line[0] in '+-CA':
                    parsed = parse_formula(line[2:], token_line)
                    print(parsed, end='\n\n')
                    graph = converter.convert(parsed)
                    for x in graph:
                        if args.latex:
                            print(x.to_latex())
                        else:
                            print(x)
                    print('\n\n')
