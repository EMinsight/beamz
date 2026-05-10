from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PortDirection = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
PortPolarization = Literal["tm", "te"]
WaveSelector = Literal["plus", "minus"]


def _normalize_direction(direction: str) -> PortDirection:
    value = str(direction).lower()
    if value not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(f"Unsupported port direction {direction!r}.")
    return value  # type: ignore[return-value]


def _normalize_polarization(polarization: str) -> PortPolarization:
    value = str(polarization).lower()
    if value not in {"tm", "te"}:
        raise ValueError(f"Unsupported port polarization {polarization!r}.")
    return value  # type: ignore[return-value]


def positive_axis_direction(direction: str) -> PortDirection:
    direction = _normalize_direction(direction)
    return ("+" + direction[1])  # type: ignore[return-value]


def opposite_direction(direction: str) -> PortDirection:
    direction = _normalize_direction(direction)
    return (("-" if direction.startswith("+") else "+") + direction[1])  # type: ignore[return-value]


def _wave_for_direction(direction: str, projection_direction: str) -> WaveSelector:
    direction = _normalize_direction(direction)
    projection_direction = _normalize_direction(projection_direction)
    if direction[1] != projection_direction[1]:
        raise ValueError(
            "Port direction and projection_direction must use the same axis: "
            f"{direction!r} vs {projection_direction!r}."
        )
    return "plus" if direction[0] == projection_direction[0] else "minus"


def _opposite_wave(selector: str) -> WaveSelector:
    selector = str(selector).lower()
    if selector == "plus":
        return "minus"
    if selector == "minus":
        return "plus"
    raise ValueError(f"Unsupported wave selector {selector!r}.")


def _object_name(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} cannot be None.")
    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "name", None)
    if not name:
        raise ValueError(f"{field} must be a monitor name or object with a name.")
    return str(name)


def _optional_object_name(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _object_name(value, field=field)


@dataclass(frozen=True)
class Port:
    """First-class modal port metadata.

    `direction` is the direction of the wave entering the simulated device at
    this port. The outgoing/scattered wave is the opposite modal wave. This
    avoids manually spelling out `incident_wave` and `scattered_wave`.
    """

    name: str
    direction: PortDirection
    polarization: PortPolarization
    monitor: str | Any | None = None
    mode_index: int = 0
    reference_monitor: str | Any | None = None
    projection_direction: PortDirection | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Port":
        return cls(
            name=str(data["name"]),
            monitor=data.get("monitor", data.get("monitor_name")),
            direction=_normalize_direction(data["direction"]),
            polarization=_normalize_polarization(data["polarization"]),
            mode_index=int(data.get("mode_index", 0)),
            reference_monitor=data.get("reference_monitor"),
            projection_direction=(
                None
                if data.get("projection_direction") is None
                else _normalize_direction(data["projection_direction"])
            ),
        )

    @property
    def monitor_name(self) -> str:
        return _object_name(
            self.name if self.monitor is None else self.monitor,
            field="monitor",
        )

    @property
    def reference_monitor_name(self) -> str | None:
        return _optional_object_name(self.reference_monitor, field="reference_monitor")

    @property
    def solver_direction(self) -> PortDirection:
        if self.projection_direction is not None:
            direction = _normalize_direction(self.projection_direction)
            incoming = _normalize_direction(self.direction)
            if direction[1] != incoming[1]:
                raise ValueError(
                    "projection_direction must use the same axis as direction: "
                    f"{direction!r} vs {incoming!r}."
                )
            return direction
        return positive_axis_direction(self.direction)

    @property
    def incident_wave(self) -> WaveSelector:
        return _wave_for_direction(self.direction, self.solver_direction)

    @property
    def scattered_wave(self) -> WaveSelector:
        return _opposite_wave(self.incident_wave)

    def to_portspec_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "monitor_name": self.monitor_name,
            "direction": self.solver_direction,
            "polarization": _normalize_polarization(self.polarization),
            "mode_index": int(self.mode_index),
            "reference_monitor": self.reference_monitor_name,
            "incident_wave": self.incident_wave,
            "scattered_wave": self.scattered_wave,
        }
