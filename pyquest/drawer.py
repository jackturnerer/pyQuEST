'''
A simple circuit drawer, rendering in matplotlib.

The bottom qubit is index 0, and all qubits from 0
to the maximum targeted/controlled upon in the 
circuit are rendered. Circuits are rendered as
compactly as possible without commutation. The main
function draw_circuit() accepts a pyQuEST Circuit 
or a list of pyQuEST operators, and spawns a new
matplotlib window.

Quirks:
    - multi-target gates acting upon non-contiguous
      qubits are drawn as a column of one-target 
      gates connected by vertical lines (similar to
      how control qubits are rendered).
    - operators without explicit target qubits are
      assumed to apply to the entire state and are
      drawn as N-target gates, where the number of
      state qubits N is inferred from the other
      operators in the circuit. This might differ
      from the actual dimension of operators like
      MixDensityMatrix.
    - decoherence channels are drawn as gates with
      dashed borders.

The algorithm is basic; the circuit canvas is 
partitioned into a (#qubits x #depth) grid and
each gate is assigned a column index therein. This
is chosen as the leftmost (smallest index) column
which has empty grid squares at every qubit between
the min and max qubits operated upon by the gate.
Note that vertical connectors of a gate (e.g. the
line between target and control qubits) occupy
grid squares, but do not prevent subsequent gates
from being placed left of them. Implementing this
is easy; we track the rightmost targeted column 
of each qubit (we can never place new gates left of
this), and also the columns to the right of this
which are occluded (but not targeted) by vertical
connectors.

@author Tyson Jones
@date June 2024
'''


from operator import itemgetter
from itertools import groupby
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import colormaps

# all concrete classes herein are drawable
from pyquest.gates import *
from pyquest.unitaries import *
from pyquest.operators import *
from pyquest.decoherence import *
from pyquest.initialisations import *



'''
TODO:
    - custom labels for CompactU, SqrtSwap, RotateAroundAxis
    - bespoke target graphic for CX as bullseye
'''



# visual order of graphic constituents (lower is occluded)
class layer:

    # horizontal qubit lines are drawn at the very bottom
    QUBIT_STAVE = 0

    # vertical lines connecting control and target qubits come next
    VERTICAL_CONNECTOR = 1

    # control qubit circles appear above the connectors
    CONTROL_CIRCLE = 2

    # gate bodies appear on top
    GATE_BODY = 2



# size constants relative to the 1x1 gate grid
class size:

    # minimum padding between circuit graphic and maptlotlib window
    PLOT_PADDING = .1

    # padding between gate's grid space and its gate body
    GATE_RECTANGLE_NEG_PADDING = .1

    # radius of circle upon control qubits, and phase gate targets
    CONTROL_CIRCLE_RADIUS = .1



'''
Logic for deciding gate placement, which...
    - positions all gates within an integer grid by deciding each gate's column
    - assigns gates as far left as is possible without commuting existing gates
    - does not allow gates to coincide with vertical connectors of other gates
'''


def has_explicit_targets(gate):

    # duck-check whether 'targets' was overwritten by operator subclass
    try:
        gate.targets
        return True
    except:
        return False


def has_controls(gate):

    # duck-check whether 'controls' was overwritten by operator subclass
    try:
        gate.controls
        return True
    except:
        return False


def get_operated_qubits(gate, num_qubits):

    # generic gates "operate" upon all their control and target qubits
    if has_explicit_targets(gate):
        return gate.controls + gate.targets

    # un-targeted gates are assumed to operate upon all qubits
    return list(range(0, num_qubits))


def get_num_qubits(gates):

    # find the biggest indexed target/control qubit among explicitly-targeted gates
    return 1 + max(
        max([*g.targets, *g.controls]) 
        for g in gates if has_explicit_targets(g))


def get_gate_column(gate_qubits, columns_of_last_target, columns_occluded_by_connectors):

    # qubits spanned by connectors between targets & controls
    gate_range = list(range(min(gate_qubits), max(gate_qubits)+1))

    # initial choice is the leftmost un-targeted column
    column = 1 + max(columns_of_last_target[q] for q in gate_range)

    # but this column may be occluded by control lines
    while any(column in columns_occluded_by_connectors[q] for q in gate_range):
        column += 1

    return column


def get_circuit_columns(gates):

    # choose one column index for each gate
    gate_columns = []

    # the grid height as informed by the highest index targeted qubit
    num_qubits = get_num_qubits(gates)

    # {qubit index: column}
    columns_of_last_target = {i:-1 for i in range(num_qubits)}

    # {qubit index: [columns]}
    columns_occluded_by_connectors = {i:[] for i in range(num_qubits)}

    for gate in gates:

        # all qubits controlled or targeted by gate (global operators return all)
        gate_qubits = get_operated_qubits(gate, num_qubits)

        # there must be room for all qubits between those explicitly targeted, for connectors
        gate_range = (min(gate_qubits), max(gate_qubits)+1)

        # find the leftmost column which fits the gate
        gate_column = get_gate_column(gate_qubits, columns_of_last_target, columns_occluded_by_connectors)
        gate_columns.append(gate_column)

        # prevent subsequent gates from commuting left of this gate
        for q in gate_qubits:
            columns_of_last_target[q] = gate_column

        # prevent subsequent gates from occupying the vertical connectors
        for q in range(*gate_range):
            columns_occluded_by_connectors[q].append(gate_column)

        # unnecessary memory cleanup; delete redundant vertical connectors left of targets
        for q in range(*gate_range):
            columns_occluded_by_connectors[q] = [c 
                for c in columns_occluded_by_connectors[q] 
                if c > columns_of_last_target[q]]

    # return one column per gate
    return gate_columns



'''
Visual gate styling
    - in matplotlib 3.8+, colours can be in a tuple with an alpha value,
      e.g. ('green', 0.3). We don't make use of this below
'''


def get_gate_label(gate):

    # channels get abbreviated
    if isinstance(gate, Damping):
        return 'γ'
    if isinstance(gate, Dephasing):
        return 'φ'
    if isinstance(gate, Depolarising):
        return 'Δ'
    if isinstance(gate, KrausMap):
        return 'K'
    if isinstance(gate, PauliNoise):
        return 'σ'
    if isinstance(gate, MixDensityMatrix):
        return 'ρ'

    # initialisations get abbreviated
    if isinstance(gate, ZeroState):
        return '0'
    if isinstance(gate, BlankState):
        return '∅'
    if isinstance(gate, ClassicalState):
        return 'i'
    if isinstance(gate, PlusState):
        return '+'
    if isinstance(gate, PureState):
        return 'ψ'

    # measurements get abbreviated
    if isinstance(gate, M):
        return '↗'

    # some gates have no labels
    if isinstance(gate, Swap):
        raise RuntimeError()
    if isinstance(gate, Phase):
        raise RuntimeError()

    # generic gates use their class name
    return type(gate).__name__


def get_gate_rect_style(gate):

    # decoherence channels have dashed rectangles
    if hasattr(pyquest.decoherence, type(gate).__name__):
        return "dashed"

    # initialisations have dotted rectangles
    if hasattr(pyquest.initialisations, type(gate).__name__):
        return "dotted"

    # all other operators have solid lines
    return "solid"


def get_gate_rect_color(gate):

    # decoherence channels are red
    if hasattr(pyquest.decoherence, type(gate).__name__):
        return "red"

    # initialisations are blue
    if hasattr(pyquest.initialisations, type(gate).__name__):
        return "blue"

    # measurements are green
    if isinstance(gate, M):
        return "green"

    # all other operators are black
    return "black"


def get_gate_face_color(gate):

    # some specific operators have their own symbols and ergo colours
    if isinstance(gate, Swap):
        return "orange"
    if isinstance(gate, Phase):
        return "purple"

    # generic gates have rectangle bodies with white faces
    return "white"


def get_vertical_connector_color(gate):

    # some specific operators have their own connector colors
    if isinstance(gate, Swap):
        return "yellow"
    if isinstance(gate, Phase):
        return "pink"

    # default
    return "gray"


def get_control_qubit_color(gate):

    # phase gate controls should be indistinguishable from targets
    if isinstance(gate, Phase):
        return get_gate_face_color(gate)

    return "black"


def get_qubit_stave_color(qubit, num_qubits):

    # qubit-specific stave colours... for some reason...
    a, b = .05, .2
    return colormaps['binary'](b + (a-b) * qubit/float(num_qubits))

    # but I won't blame you for doing a boring fixed colour, like...
    return "lightgray"



'''
Logic for producing graphics, which...
    - draws phase and control qubits as circles
    - makes decoherence channel borders dashed
    - draws swap gates with X symbols
    - labels gates with concise strings
    - merges gate bodies which target adjacent qubits
'''


def get_grouped_consecutive_items(nums):

    # [1,2,4,5,6] -> [(1,2), (4,5,6)]
    indAndNums = enumerate(sorted(nums))
    for _, group in groupby(indAndNums, lambda x:x[0]-x[1]):
        yield list(map(itemgetter(1), group))


def get_gate_graphic_components(gate, column, num_qubits):

    # graphics consist of vertical connector lines, control circles, and gate body rectangles
    lines = []        # item = [(x0,y0), (x1,y1)]
    circles = []      # item = (x0,y0)
    rectangles = []   # item = [(x0,y0), (x0,y1), (x1,y1), (x1,y0)]

    # clarifying (in principle...) constants relative to 1x1 grid
    qubits  = get_operated_qubits(gate, num_qubits)
    pad     = size.GATE_RECTANGLE_NEG_PADDING
    halfcol = .5
    nextcol = column      + 1
    midcol  = column      + halfcol
    midtop  = max(qubits) + halfcol
    midbot  = min(qubits) + halfcol
    padcol  = column + pad
    padnextcol = nextcol - pad
    
    # note connector lines may be superfluous and occluded by rectangles
    lines.append([ (midcol, midbot), (midcol, midtop) ]) # only one line needed

    # only attempt drawing controls if any exist (else .controls throws)
    if has_controls(gate):
        circles += [(midcol, q+halfcol) for q in gate.controls]

    # explicitly targeted gates have adjacent targets merged into rectangles
    if has_explicit_targets(gate):
        for group in get_grouped_consecutive_items(gate.targets):
            x0, y0 = padcol,     min(group)   + pad
            x1, y1 = padnextcol, max(group)+1 - pad
            rectangles.append([ (x0,y0), (x0,y1), (x1,y1), (x1,y0) ])

    # whereas untargeted gates are assumed global and act on every qubit
    else:
        x0, y0 = padcol,     0          + pad
        x1, y1 = padnextcol, num_qubits - pad
        rectangles.append([ (x0,y0), (x0,y1), (x1,y1), (x1,y0) ])

    # returned in order of increasing z-order
    return lines, circles, rectangles


def draw_gate_body(gate, column, rectangles, plt, ax):

    # gate-specific styling for special operators (which aren't drawn as rectangles)
    special_opts = {
        'color':  get_gate_face_color(gate),
        'zorder': layer.GATE_BODY}

    # SWAP gates ignore rectangles and draw X at every target
    if isinstance(gate, Swap):
        for q in gate.targets:
            plt.scatter(column+.5, q+.5, marker='x', **special_opts)
        return

    # Phase gates ignore rectangles and draw circle at every target
    if isinstance(gate, Phase):
        radius = size.CONTROL_CIRCLE_RADIUS
        for q in gate.targets:
            ax.add_patch(plt.Circle((column+.5, q+.5), radius, **special_opts))
        return

    # styling for gates drawn as rectangles
    rect_ops = {
        'linestyle': get_gate_rect_style(gate),
        'edgecolor': get_gate_rect_color(gate),
        'facecolor': get_gate_face_color(gate),
        'zorder':    layer.GATE_BODY}

    for rect in rectangles:
        ax.add_patch(plt.Polygon(rect, **rect_ops))

    # each rectangle is labelled
    label = get_gate_label(gate)
    for rect in rectangles:
        pos = (mean(x[i] for x in rect) for i in [0,1])
        plt.text(*pos, s=label, va='center', ha='center')


def draw_gate(gate, column, num_qubits, plt, ax):

    lines, dots, rectangles = get_gate_graphic_components(gate, column, num_qubits)

    # draw vertical connector lines (at back)
    for line in lines:

        # avoid drawing zero-length lines (eles matplotlib throws)
        if line[0] == line[1]:
            continue

        (a,b),(c,d) = line
        plt.plot(
            (a,c), (b,d), 
            color=get_vertical_connector_color(gate), 
            zorder=layer.VERTICAL_CONNECTOR)

    # draw control dots
    for dot in dots:
        ax.add_patch(plt.Circle(
            dot, size.CONTROL_CIRCLE_RADIUS, 
            color=get_control_qubit_color(gate), 
            zorder=layer.CONTROL_CIRCLE))

    # draw the main body of the gate; possibly labelled rectangles, or bespoke symbols
    draw_gate_body(gate, column, rectangles, plt, ax)


def draw_circuit(gates):

    # get matplotlib handles
    ax = plt.gca()

    # determine circuit layout
    gate_columns = get_circuit_columns(gates)
    num_columns = 1 + max(gate_columns)
    num_qubits = get_num_qubits(gates)

    # draw horizontal qubit stave
    for q in range(num_qubits):
        plt.plot(
            [-.5, num_columns+.5], [q+.5, q+.5], 
            color=get_qubit_stave_color(q, num_qubits), 
            zorder=layer.QUBIT_STAVE)

    # draw each gate above stave
    for gate, column in zip(gates, gate_columns):
        draw_gate(gate, column, num_qubits, plt, ax)

    # set plot range
    pad = size.PLOT_PADDING
    ax.set_xlim(-.5 - pad, num_columns + pad +.5)
    ax.set_ylim(-.5 - pad, num_qubits  + pad +.5)

    # hide frame
    ax.axis('off')

    # force 1:1 aspect ratio (not crucial; fun to relax)
    ax.set_aspect('equal')

    # render circuit immediately
    plt.show()
