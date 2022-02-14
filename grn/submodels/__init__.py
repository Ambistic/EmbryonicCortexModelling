import submodels.bistate1
import submodels.tristate1
import submodels.cellbasic1
import submodels.bistate_LI
import submodels.tristate_LI
import submodels.tri_ambi_mutant
import submodels.grn_v1
import submodels.grn_v2
import submodels.grn_v2_opti
from submodels import grn_v3
from submodels import grn_v4

factories = {
    "bistate1": bistate1.BiStateModelFactory,
    "tristate1": tristate1.TriStateModelFactory,
    "basic": cellbasic1.CellBasicModelFactory,
    "bistateLI": bistate_LI.BiStateLIModelFactory,
    "tristateLI": tristate_LI.TriStateLIModelFactory,
    "triambimutant": tri_ambi_mutant.TriStateAmbiLIMutantModelFactory,
    "grn1": grn_v1.GRNModelFactory,
    "grn2": grn_v2.GRNModelFactory,
    "grn2_opti": grn_v2_opti.GRNModelFactory,
    "grn3": grn_v3.GRNModelFactory,
    "grn4": grn_v4.GRNModelFactory,
}