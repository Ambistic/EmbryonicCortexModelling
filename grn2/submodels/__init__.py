import submodels.bistate1
import submodels.tristate1
import submodels.cellbasic1
import submodels.bistate_LI
import submodels.tristate_LI
import submodels.tri_ambi_mutant
from submodels import grn_auto_v1

factories = {
    "bistate1": bistate1.BiStateModelFactory,
    "tristate1": tristate1.TriStateModelFactory,
    "basic": cellbasic1.CellBasicModelFactory,
    "bistateLI": bistate_LI.BiStateLIModelFactory,
    "tristateLI": tristate_LI.TriStateLIModelFactory,
    "triambimutant": tri_ambi_mutant.TriStateAmbiLIMutantModelFactory,
    "grn_auto1": grn_auto_v1.GRNModelFactory,
}