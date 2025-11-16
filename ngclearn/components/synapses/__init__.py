from .denseSynapse import DenseSynapse
from .staticSynapse import StaticSynapse

## short-term plasticity components
from .STPDenseSynapse import STPDenseSynapse
from .exponentialSynapse import ExponentialSynapse
from .doubleExpSynapse import DoubleExpSynapse
from .alphaSynapse import AlphaSynapse

## dense synaptic components
# from .hebbian.hebbianSynapse import HebbianSynapse
from .hebbian.traceSTDPSynapse import TraceSTDPSynapse
from .hebbian.expSTDPSynapse import ExpSTDPSynapse
from .hebbian.eventSTDPSynapse import EventSTDPSynapse
from .hebbian.BCMSynapse import BCMSynapse

## conv/deconv synaptic components
from .convolution.convSynapse import ConvSynapse
from .convolution.staticConvSynapse import StaticConvSynapse
from .convolution.hebbianConvSynapse import HebbianConvSynapse
from .convolution.traceSTDPConvSynapse import TraceSTDPConvSynapse
from .convolution.deconvSynapse import DeconvSynapse
from .convolution.staticDeconvSynapse import StaticDeconvSynapse
from .convolution.hebbianDeconvSynapse import HebbianDeconvSynapse
from .convolution.traceSTDPDeconvSynapse import TraceSTDPDeconvSynapse

## modulated synaptic components
from .modulated.MSTDPETSynapse import MSTDPETSynapse
# from .modulated.REINFORCESynapse import REINFORCESynapse

## patched synaptic components
from .patched.patchedSynapse import PatchedSynapse
from .patched.staticPatchedSynapse import StaticPatchedSynapse
from .patched.hebbianPatchedSynapse import HebbianPatchedSynapse

