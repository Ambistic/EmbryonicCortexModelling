import sys
import os
sys.path.append(os.path.dirname(__file__))  # patch

import bistate1
import tristate1
import cellbasic1
import bistate_LI
import tristate_LI
import tri_ambi_mutant


factories = {
    "bistate1": bistate1.BiStateModelFactory,
    "tristate1": tristate1.TriStateModelFactory,
    "basic": cellbasic1.CellBasicModelFactory,
    "bistateLI": bistate_LI.BiStateLIModelFactory,
    "tristateLI": tristate_LI.TriStateLIModelFactory,
    "triambimutant": tri_ambi_mutant.TriStateAmbiLIMutantModelFactory,
}