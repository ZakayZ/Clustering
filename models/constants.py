import enum

MEV_PER_GEV = 1000.0
EDGE_PHYS_DIM = 23
GAT_NODE_IN_DIM = 8

class GraphKind(enum.StrEnum):
    KNN = enum.auto()
    RADIUS = enum.auto()
    FULL = enum.auto()
