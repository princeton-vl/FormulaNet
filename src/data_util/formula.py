'''Python 3. This file contains a graph data structure.'''
import collections
FO_QUANT = {'!', '?', '?!'}
HOL_QUANT = {'@', '\\'}
QUANTIFIER = FO_QUANT | HOL_QUANT


class NodeType(object):
    'Enumerate class'
    VAR = 0
    VARFUNC = 1
    CONST = 2
    CONSTFUNC = 3
    QUANT = 4
    NULL = 5


def type_printer(node_type):
    'Print type string'
    d = {
        NodeType.VAR: 'VAR',
        NodeType.VARFUNC: 'VFUNC',
        NodeType.CONST: 'CONST',
        NodeType.CONSTFUNC: 'CFUNC',
        NodeType.QUANT: 'QUANT',
        NodeType.NULL: 'NODE'
    }
    return d[node_type]


class Node(object):
    '''Unified node representation.

    Parameters
    ----------
    ntype : NodeType
        Type of the node
    name : str
        Name of the node.
    '''

    id = 0

    def __init__(self, ntype, name):
        self.id = self.__class__.id
        self.__class__.id += 1
        self.name = name
        self.type = ntype
        self.incoming = []
        self.outgoing = []
        # The following is included in incoming / outgoing
        # To have them just for convenience
        self.quant = None  # For var/varfuncs
        self.vfunc = []  # For var functions
        self.vvalue = None  # For var value

    @classmethod
    def reset_id(cls):
        """Reset id count to 0"""
        cls.id = 0

    def __str__(self):
        for x in self.incoming:
            if x is None:
                assert False
        for x in self.outgoing:
            if x is None:
                assert False
        extra = ''
        if self.type == NodeType.VAR or self.type == NodeType.VARFUNC:
            extra = ' | quant: %s' % self.quant.id if self.quant is not None else ''
        elif self.type == NodeType.QUANT:
            extra = ' | var: value: {} | functions: [{}]'.format(
                self.vvalue.id if self.vvalue is not None else ' ',
                ' '.join([str(x.id) for x in self.vfunc]))
        return '<{}> {} {}: i: {} | o: {} | {}'.format(
            self.id,
            type_printer(self.type),
            self.name,
            ' '.join([str(x.id) for x in self.incoming]),
            ' '.join([str(x.id) for x in self.outgoing]),
            extra)  # yapf: disable

    def __repr__(self):
        return self.__str__()

    @property
    def latex_name(self):
        '''Name used in latex node'''
        if self.type == NodeType.VARFUNC:
            extra = '#F'
        else:
            extra = ''
        node_name = self.name
        if node_name == '==>:c':
            node_name = '==$>$'
        node_name = '"' + node_name + str(self.id) + extra + '"'
        if self.type == NodeType.QUANT:
            node_name += '[diamond]'
        elif self.type == NodeType.VARFUNC or self.type == NodeType.VAR:
            node_name += '[rectangle]'
        return node_name

    def to_latex(self):
        '''Generate latex code'''
        edges = self.outgoing + self.vfunc
        if self.vvalue:
            edges += [self.vvalue]
        latex = self.latex_name + ' -> {' + ','.join([x.latex_name
                                                      for x in edges]) + '},'
        escaped = latex.translate(
            str.maketrans({
                "%": r"\%",
                "#": r"\#",
                "|": r"$\mid$",
                "~": r"$\sim$",
                "\\": r"\textbackslash",
                "<": r"$<$",
                "_": r"\_",
            }))
        return escaped
