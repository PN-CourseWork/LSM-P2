Domain Decomposition
====================

Compare 1D sliced decomposition vs 3D cubic decomposition strategies. The sliced approach splits the domain along the Z-axis with each rank owning horizontal slices, exchanging 2 ghost planes. The cubic approach uses 3D Cartesian grid decomposition across all spatial dimensions, exchanging 6 ghost faces. These experiments measure communication overhead, verify load balance, and analyze scaling behavior.
