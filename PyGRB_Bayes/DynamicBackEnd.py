#
# import numpy as np
# import scipy.special as special
# from scipy.special import gammaln
# import bilby
# from bilby.core.prior import PriorDict        as bilbyPriorDict
# from bilby.core.prior import Uniform          as bilbyUniform
# from bilby.core.prior import Constraint       as bilbyConstraint
# from bilby.core.prior import LogUniform       as bilbyLogUniform
#
# # MIN_FLOAT = sys.float_info[3]
#
# from PyGRB_Bayes.backend.makekeys import MakeKeys
#
#
# # @dataclass
# # class PriorRanges:
# #     priors_pulse_start: float
# #     priors_pulse_end:   float
# #     priors_td_lo:       float = None
# #     priors_td_hi:       float = None
# #     priors_bg_lo:       float = 1e-1  ## SCALING IS COUNTS / BIN
# #     priors_bg_hi:       float = 1e3   ## SCALING IS COUNTS / BIN
# #     priors_mr_lo:       float = 0.2   ## which means that it is
# #     priors_mr_hi:       float = 1.4     # 1 / 0.064 times smaller
# #     priors_tau_lo:      float = 1e-3  # than you think it is
# #     priors_tau_hi:      float = 1e3   # going to be !!!!!!!!!!!!
# #     priors_xi_lo:       float = 1e-3
# #     priors_xi_hi:       float = 1e3
# #     priors_gamma_min:   float = 1e-1
# #     priors_gamma_max:   float = 1e1
# #     priors_nu_min:      float = 1e-1
# #     priors_nu_max:      float = 1e1
# #     priors_scale_min:   float = 1e0  ## SCALING IS COUNTS / BIN
# #     priors_scale_max:   float = 1e4
