from deepview.DeepView import DeepView

import warnings
import sys

if not sys.warnoptions:
	from matplotlib import MatplotlibDeprecationWarning
	from numba import NumbaWarning
	# Disable UserWarning from umap saying that when using precomputed distances,
	# the inverse transformation is disabled.
	warnings.filterwarnings('ignore', module='umap')
	# Disable matplotlib Deprication Warning. In this case, this is a warning
	# from one of their own methods Axes.plot
	warnings.filterwarnings('ignore', module='deepview',
		category=MatplotlibDeprecationWarning)
	# Disable numba Warnings.
	warnings.filterwarnings('ignore', module='numba',
		category=NumbaWarning)
