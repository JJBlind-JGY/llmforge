"""NVIDIA GPU inventory/topology parsing and placement classification."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass
from itertools import combinations


_GPU_RE = re.compile(r"GPU\d+")


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_mib: int
    numa_node: int | None = None
    cpu_affinity: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GpuLink:
    source: int
    destination: int
    relation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Placement:
    gpu_indices: tuple[int, ...]
    pair_relations: tuple[str, ...]
    numa_nodes: tuple[int, ...]
    same_numa: bool | None
    crosses_system_interconnect: bool

    def to_dict(self) -> dict:
        return {
            "gpu_indices": list(self.gpu_indices),
            "pair_relations": list(self.pair_relations),
            "numa_nodes": list(self.numa_nodes),
            "same_numa": self.same_numa,
            "crosses_system_interconnect": self.crosses_system_interconnect,
        }


@dataclass(frozen=True)
class TopologySnapshot:
    devices: tuple[GpuDevice, ...]
    links: tuple[GpuLink, ...]
    raw_topology: str

    def device(self, index: int) -> GpuDevice:
        for device in self.devices:
            if device.index == index:
                return device
        raise KeyError(f"GPU{index} not found.")

    def relation(self, source: int, destination: int) -> str:
        if source == destination:
            return "X"

        for link in self.links:
            if link.source == source and link.destination == destination:
                return link.relation

        raise KeyError(f"No topology relation for GPU{source} -> GPU{destination}.")

    def to_dict(self) -> dict:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "links": [link.to_dict() for link in self.links],
            "raw_topology": self.raw_topology,
        }


def _parse_int(value: str) -> int:
    match = re.search(r"-?\d+", value)
    if match is None:
        raise ValueError(f"Expected integer in {value!r}.")
    return int(match.group())


def parse_gpu_inventory_csv(text: str) -> list[GpuDevice]:
    """Parse `nvidia-smi --format=csv,noheader,nounits` inventory output.

    Expected columns:
    index,name,uuid,pci.bus_id,memory.total
    """

    devices = []

    reader = csv.reader(io.StringIO(text))

    for row in reader:
        if not row:
            continue

        if len(row) != 5:
            raise ValueError(
                "GPU inventory must have exactly five columns: "
                "index,name,uuid,pci.bus_id,memory.total."
            )

        devices.append(
            GpuDevice(
                index=_parse_int(row[0].strip()),
                name=row[1].strip(),
                uuid=row[2].strip(),
                pci_bus_id=row[3].strip(),
                memory_total_mib=_parse_int(row[4].strip()),
            )
        )

    if not devices:
        raise ValueError("GPU inventory is empty.")

    return devices


def parse_nvidia_smi_topology(
    text: str,
    *,
    inventory: list[GpuDevice],
) -> TopologySnapshot:
    """Parse the matrix portion of `nvidia-smi topo -m`.

    The parser intentionally uses the number of `GPU<N>` columns rather than
    relying on the multi-word header labels after the connectivity matrix.
    """

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    header_index = None
    gpu_labels: tuple[str, ...] = ()

    for index, line in enumerate(lines):
        tokens = line.split()
        labels = tuple(token for token in tokens if _GPU_RE.fullmatch(token))

        if labels and "CPU" in line and "Affinity" in line:
            header_index = index
            gpu_labels = labels
            break

    if header_index is None:
        raise ValueError("Could not locate `nvidia-smi topo -m` header.")

    expected_indices = tuple(int(label[3:]) for label in gpu_labels)

    device_by_index = {device.index: device for device in inventory}

    missing = set(expected_indices) - set(device_by_index)
    if missing:
        raise ValueError(
            f"Topology references GPUs missing from inventory: {sorted(missing)}"
        )

    links: list[GpuLink] = []
    updated_devices: dict[int, GpuDevice] = {}

    rows_seen = 0

    for line in lines[header_index + 1 :]:
        columns = line.split()

        if not columns or not _GPU_RE.fullmatch(columns[0]):
            if rows_seen >= len(gpu_labels):
                break
            continue

        source = int(columns[0][3:])

        if source not in expected_indices:
            continue

        if len(columns) < 1 + len(gpu_labels):
            raise ValueError(f"Malformed topology row: {line!r}")

        matrix_values = columns[1 : 1 + len(gpu_labels)]

        for destination, relation in zip(expected_indices, matrix_values):
            if source == destination:
                continue

            links.append(
                GpuLink(
                    source=source,
                    destination=destination,
                    relation=relation,
                )
            )

        # `nvidia-smi topo -m` places CPU affinity and NUMA immediately after
        # the N GPU matrix columns. GPU NUMA ID may follow and is not needed.
        tail = columns[1 + len(gpu_labels) :]

        cpu_affinity = tail[0] if len(tail) >= 1 else None
        numa_node = None

        if len(tail) >= 2 and tail[1] not in {"N/A", "-"}:
            try:
                numa_node = int(tail[1])
            except ValueError:
                numa_node = None

        base = device_by_index[source]

        updated_devices[source] = GpuDevice(
            index=base.index,
            name=base.name,
            uuid=base.uuid,
            pci_bus_id=base.pci_bus_id,
            memory_total_mib=base.memory_total_mib,
            numa_node=numa_node,
            cpu_affinity=cpu_affinity,
        )

        rows_seen += 1

        if rows_seen == len(gpu_labels):
            break

    if rows_seen != len(gpu_labels):
        raise ValueError("Did not parse one topology row per GPU.")

    devices = tuple(updated_devices[index] for index in expected_indices)

    return TopologySnapshot(
        devices=devices,
        links=tuple(links),
        raw_topology=text,
    )


def classify_placement(
    snapshot: TopologySnapshot,
    gpu_indices: tuple[int, ...],
) -> Placement:
    if not gpu_indices:
        raise ValueError("gpu_indices must not be empty.")

    if len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError("gpu_indices must be unique.")

    devices = [snapshot.device(index) for index in gpu_indices]

    relations = tuple(
        snapshot.relation(left, right) for left, right in combinations(gpu_indices, 2)
    )

    known_numa = tuple(
        device.numa_node for device in devices if device.numa_node is not None
    )

    same_numa: bool | None

    if len(known_numa) != len(devices):
        same_numa = None
    else:
        same_numa = len(set(known_numa)) == 1

    return Placement(
        gpu_indices=gpu_indices,
        pair_relations=relations,
        numa_nodes=tuple(known_numa),
        same_numa=same_numa,
        crosses_system_interconnect=any(relation == "SYS" for relation in relations),
    )
