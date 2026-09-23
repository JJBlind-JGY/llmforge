from llmforge.distributed.topology import (
    classify_placement,
    parse_gpu_inventory_csv,
    parse_nvidia_smi_topology,
)

INVENTORY = """
0, NVIDIA GeForce RTX 4090, GPU-a, 00000000:01:00.0, 24564
1, NVIDIA GeForce RTX 4090, GPU-b, 00000000:02:00.0, 24564
2, NVIDIA GeForce RTX 4090, GPU-c, 00000000:81:00.0, 24564
3, NVIDIA GeForce RTX 4090, GPU-d, 00000000:82:00.0, 24564
"""


TOPOLOGY = """
        GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0     X   NODE SYS  SYS  0-11,24-35 0 N/A
GPU1    NODE  X   SYS  SYS  0-11,24-35 0 N/A
GPU2    SYS  SYS   X   NODE 12-23,36-47 1 N/A
GPU3    SYS  SYS  NODE  X   12-23,36-47 1 N/A
"""


def test_parse_realistic_four_gpu_topology() -> None:
    inventory = parse_gpu_inventory_csv(INVENTORY)

    snapshot = parse_nvidia_smi_topology(
        TOPOLOGY,
        inventory=inventory,
    )

    assert len(snapshot.devices) == 4
    assert snapshot.device(0).numa_node == 0
    assert snapshot.device(2).numa_node == 1
    assert snapshot.relation(0, 1) == "NODE"
    assert snapshot.relation(0, 2) == "SYS"


def test_classify_same_numa_pair() -> None:
    snapshot = parse_nvidia_smi_topology(
        TOPOLOGY,
        inventory=parse_gpu_inventory_csv(INVENTORY),
    )

    placement = classify_placement(
        snapshot,
        (0, 1),
    )

    assert placement.same_numa is True
    assert placement.crosses_system_interconnect is False
    assert placement.pair_relations == ("NODE",)


def test_classify_cross_numa_pair() -> None:
    snapshot = parse_nvidia_smi_topology(
        TOPOLOGY,
        inventory=parse_gpu_inventory_csv(INVENTORY),
    )

    placement = classify_placement(
        snapshot,
        (0, 2),
    )

    assert placement.same_numa is False
    assert placement.crosses_system_interconnect is True
    assert placement.pair_relations == ("SYS",)
