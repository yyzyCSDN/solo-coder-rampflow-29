from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from threading import RLock
from typing import Iterable, Mapping

from .ops_store import VersionConflict
from .timeutil import utc


Q = Decimal("0.001")
MONEY = Decimal("0.01")


@dataclass(frozen=True)
class FuelTank:
    tank_id: str
    capacity_l: Decimal
    usable_l: Decimal
    density_kg_l: Decimal


@dataclass(frozen=True)
class FuelDelivery:
    delivery_id: str
    flight_id: str
    tank_id: str
    meter_start_l: Decimal
    meter_end_l: Decimal
    temperature_c: Decimal
    density_kg_l: Decimal
    delivered_at: datetime
    corrected_l: Decimal
    mass_kg: Decimal
    price_per_l: Decimal
    amount: Decimal
    reversed_by: str | None = None
    measurement_basis: "FuelMeasurementBasis | None" = None


@dataclass(frozen=True)
class FuelMeasurementBasis:
    """Correction rules frozen for one historical backfill batch."""

    basis_id: str
    reference_temp_c: Decimal
    thermal_coefficient: Decimal = Decimal("0.00095")
    volume_unit: str = "L"
    volume_quantity: Decimal = Decimal("0.001")
    mass_unit: str = "kg"
    mass_quantity: Decimal = Decimal("0.001")
    money_unit: str = "amount-major"
    money_quantity: Decimal = MONEY
    method: str = "temperature-corrected-meter-volume"

    def __post_init__(self) -> None:
        if not self.basis_id:
            raise FuelAccountingError("measurement basis id required")
        if self.thermal_coefficient <= 0:
            raise FuelAccountingError("thermal coefficient must be positive")
        if self.volume_quantity <= 0 or self.mass_quantity <= 0 or self.money_quantity <= 0:
            raise FuelAccountingError("measurement quantum must be positive")
        if self.volume_unit != "L" or self.mass_unit != "kg":
            raise FuelAccountingError("unsupported measurement unit")


@dataclass(frozen=True)
class FuelBatchEntry:
    delivery_id: str
    flight_id: str
    tank_id: str
    meter_start_l: Decimal
    meter_end_l: Decimal
    temperature_c: Decimal
    measured_density_kg_l: Decimal
    price_per_l: Decimal
    delivered_at: datetime


FuelBatchItem = FuelBatchEntry


@dataclass(frozen=True)
class FuelBatchReceipt:
    batch_id: str
    basis: FuelMeasurementBasis
    deliveries: tuple[FuelDelivery, ...]
    committed_at: datetime
    version: int


class FuelAccountingError(RuntimeError):
    pass



def quant(value: Decimal, unit: Decimal = Q) -> Decimal:
    return value.quantize(unit, rounding=ROUND_HALF_UP)


class FuelLedger:
    def __init__(self, tanks: Iterable[FuelTank], reference_temp_c: Decimal = Decimal("15")):
        self.tanks = {t.tank_id: t for t in tanks}
        self.reference_temp_c = reference_temp_c
        self.deliveries: dict[str, FuelDelivery] = {}
        self.inventory_l: dict[str, Decimal] = {
            t.tank_id: quant(t.usable_l) for t in tanks
        }
        self.flight_totals_l: dict[str, Decimal] = {}
        self.flight_totals_kg: dict[str, Decimal] = {}
        self.batches: dict[str, FuelBatchReceipt] = {}
        self.version = 0
        self._lock = RLock()

    @property
    def generation(self) -> int:
        return self.version

    def _temperature_corrected(
        self,
        volume_l: Decimal,
        temp_c: Decimal,
        coefficient: Decimal | None = None,
        reference_temp_c: Decimal | None = None,
        quantum: Decimal | None = None,
    ) -> Decimal:
        coefficient = Decimal("0.00095") if coefficient is None else coefficient
        reference = self.reference_temp_c if reference_temp_c is None else reference_temp_c
        factor = Decimal("1") + coefficient * (temp_c - reference)
        if factor <= 0:
            raise FuelAccountingError("invalid thermal correction")
        corrected = volume_l / factor
        return corrected if quantum is None else quant(corrected, quantum)

    @staticmethod
    def _basis(
        basis: FuelMeasurementBasis | str | None,
        reference_temp_c: Decimal = Decimal("15"),
    ) -> FuelMeasurementBasis | None:
        if basis is None or isinstance(basis, FuelMeasurementBasis):
            return basis
        return FuelMeasurementBasis(basis, reference_temp_c)

    def _correct_for(
        self,
        volume_l: Decimal,
        temp_c: Decimal,
        basis: FuelMeasurementBasis | None,
    ) -> Decimal:
        quantum = Q if basis is None else basis.volume_quantity
        if basis is None:
            return self._temperature_corrected(volume_l, temp_c, quantum=quantum)
        return self._temperature_corrected(
            volume_l,
            temp_c,
            basis.thermal_coefficient,
            basis.reference_temp_c,
            quantum,
        )

    def _build_delivery(
        self,
        delivery_id: str,
        flight_id: str,
        tank_id: str,
        meter_start_l: Decimal,
        meter_end_l: Decimal,
        temperature_c: Decimal,
        measured_density_kg_l: Decimal,
        price_per_l: Decimal,
        delivered_at: datetime,
        basis: FuelMeasurementBasis | None,
    ) -> FuelDelivery:
        if not delivery_id or not flight_id:
            raise FuelAccountingError("delivery and flight ids are required")
        if tank_id not in self.tanks:
            raise KeyError(tank_id)
        raw = meter_end_l - meter_start_l
        if raw <= 0:
            raise FuelAccountingError("fuel meter must advance")
        if measured_density_kg_l <= 0 or price_per_l < 0:
            raise FuelAccountingError("invalid density or price")
        corrected = self._correct_for(raw, temperature_c, basis)
        mass_quantum = Q if basis is None else basis.mass_quantity
        money_quantum = MONEY if basis is None else basis.money_quantity
        mass = quant(corrected * measured_density_kg_l, mass_quantum)
        amount = quant(corrected * price_per_l, money_quantum)
        return FuelDelivery(
            delivery_id,
            flight_id,
            tank_id,
            meter_start_l,
            meter_end_l,
            temperature_c,
            measured_density_kg_l,
            utc(delivered_at),
            corrected,
            mass,
            price_per_l,
            amount,
            None,
            basis,
        )

    def _apply_delivery(self, delivery: FuelDelivery) -> None:
        current = self.inventory_l[delivery.tank_id]
        if delivery.corrected_l > current:
            raise FuelAccountingError("insufficient tank inventory")
        self.inventory_l[delivery.tank_id] = quant(current - delivery.corrected_l)
        self.flight_totals_l[delivery.flight_id] = quant(
            self.flight_totals_l.get(delivery.flight_id, Decimal("0")) + delivery.corrected_l
        )
        self.flight_totals_kg[delivery.flight_id] = quant(
            self.flight_totals_kg.get(delivery.flight_id, Decimal("0")) + delivery.mass_kg
        )
        self.deliveries[delivery.delivery_id] = delivery

    @staticmethod
    def _same_entries(receipt: FuelBatchReceipt, entries: tuple[FuelBatchEntry, ...]) -> bool:
        if len(receipt.deliveries) != len(entries):
            return False
        fields = (
            "delivery_id",
            "flight_id",
            "tank_id",
            "meter_start_l",
            "meter_end_l",
            "temperature_c",
            "measured_density_kg_l",
            "price_per_l",
        )
        for delivery, entry in zip(receipt.deliveries, entries):
            for entry_name in fields:
                delivery_name = "density_kg_l" if entry_name == "measured_density_kg_l" else entry_name
                if getattr(delivery, delivery_name) != getattr(entry, entry_name):
                    return False
            if delivery.delivered_at != utc(entry.delivered_at):
                return False
        return True

    def record(
        self,
        delivery_id: str,
        flight_id: str,
        tank_id: str,
        meter_start_l: Decimal,
        meter_end_l: Decimal,
        temperature_c: Decimal,
        measured_density_kg_l: Decimal,
        price_per_l: Decimal,
        delivered_at: datetime,
        measurement_basis: FuelMeasurementBasis | str | None = None,
    ) -> FuelDelivery:
        basis = self._basis(measurement_basis, self.reference_temp_c)
        with self._lock:
            if delivery_id in self.deliveries:
                return self.deliveries[delivery_id]
            delivery = self._build_delivery(
                delivery_id,
                flight_id,
                tank_id,
                meter_start_l,
                meter_end_l,
                temperature_c,
                measured_density_kg_l,
                price_per_l,
                delivered_at,
                basis,
            )
            self._apply_delivery(delivery)
            self.version += 1
            return delivery

    def preview_batch(
        self,
        batch_id: str,
        entries: Iterable[FuelBatchEntry],
        measurement_basis: FuelMeasurementBasis,
    ) -> FuelBatchReceipt:
        if not batch_id:
            raise FuelAccountingError("batch id required")
        if not isinstance(measurement_basis, FuelMeasurementBasis):
            raise FuelAccountingError("measurement basis must be fixed before preview")
        ordered = tuple(entries)
        if not ordered:
            raise FuelAccountingError("batch must not be empty")
        with self._lock:
            existing = self.batches.get(batch_id)
            if existing is not None:
                if existing.basis != measurement_basis:
                    raise VersionConflict("measurement basis for batch changed")
                if not self._same_entries(existing, ordered):
                    raise VersionConflict("batch entries changed")
                return existing
            projected = dict(self.inventory_l)
            seen: set[str] = set()
            deliveries = []
            for entry in ordered:
                if entry.delivery_id in seen:
                    raise FuelAccountingError(f"duplicate delivery in batch: {entry.delivery_id}")
                seen.add(entry.delivery_id)
                if entry.delivery_id in self.deliveries:
                    raise FuelAccountingError(f"delivery already recorded: {entry.delivery_id}")
                delivery = self._build_delivery(
                    entry.delivery_id,
                    entry.flight_id,
                    entry.tank_id,
                    entry.meter_start_l,
                    entry.meter_end_l,
                    entry.temperature_c,
                    entry.measured_density_kg_l,
                    entry.price_per_l,
                    entry.delivered_at,
                    measurement_basis,
                )
                if delivery.corrected_l > projected[delivery.tank_id]:
                    raise FuelAccountingError("insufficient tank inventory")
                projected[delivery.tank_id] = quant(projected[delivery.tank_id] - delivery.corrected_l)
                deliveries.append(delivery)
            return FuelBatchReceipt(
                batch_id,
                measurement_basis,
                tuple(deliveries),
                max(d.delivered_at for d in deliveries),
                self.version + 1,
            )

    def commit_batch(
        self,
        batch_id: str,
        entries: Iterable[FuelBatchEntry],
        measurement_basis: FuelMeasurementBasis,
        expected_version: int | None = None,
        expected_generation: int | None = None,
    ) -> FuelBatchReceipt:
        expected = expected_version if expected_version is not None else expected_generation
        ordered = tuple(entries)
        with self._lock:
            previous = self.batches.get(batch_id)
            if previous is not None:
                if previous.basis != measurement_basis:
                    raise VersionConflict("measurement basis for batch changed")
                if not self._same_entries(previous, ordered):
                    raise VersionConflict("batch entries changed")
                return previous
            if expected is not None and expected != self.version:
                raise VersionConflict(f"expected {expected}, found {self.version}")
            staged = self.preview_batch(batch_id, ordered, measurement_basis)
            for delivery in staged.deliveries:
                self._apply_delivery(delivery)
            receipt = replace(staged, version=self.version + 1)
            self.batches[batch_id] = receipt
            self.version += 1
            return receipt

    record_batch = commit_batch
    prevalidate_batch = preview_batch

    def reverse(self, delivery_id: str, reversal_id: str, reversed_at: datetime) -> FuelDelivery:
        with self._lock:
            original = self.deliveries[delivery_id]
            if original.corrected_l < 0:
                raise FuelAccountingError("reversal entries cannot be reversed again")
            if original.reversed_by is not None:
                if reversal_id == original.reversed_by:
                    return self.deliveries[reversal_id]
                raise FuelAccountingError("delivery already reversed")
            if reversal_id in self.deliveries:
                raise FuelAccountingError("duplicate reversal id")
            corrected = original.corrected_l
            corrected_mass = original.mass_kg
            reversal = FuelDelivery(
                reversal_id,
                original.flight_id,
                original.tank_id,
                original.meter_end_l,
                original.meter_start_l,
                original.temperature_c,
                original.density_kg_l,
                utc(reversed_at),
                -corrected,
                -corrected_mass,
                original.price_per_l,
                -original.amount,
                None,
                original.measurement_basis,
            )
            self.deliveries[delivery_id] = replace(original, reversed_by=reversal_id)
            self.deliveries[reversal_id] = reversal
            self.inventory_l[original.tank_id] = quant(self.inventory_l[original.tank_id] + corrected)
            self.flight_totals_l[original.flight_id] = quant(self.flight_totals_l[original.flight_id] - corrected)
            self.flight_totals_kg[original.flight_id] = quant(self.flight_totals_kg[original.flight_id] - corrected_mass)
            self.version += 1
            return reversal

    def replenish(self, tank_id: str, amount_l: Decimal) -> Decimal:
        if amount_l <= 0:
            raise FuelAccountingError("replenishment must be positive")
        with self._lock:
            tank = self.tanks[tank_id]
            next_level = quant(self.inventory_l[tank_id] + amount_l)
            if next_level > tank.capacity_l:
                raise FuelAccountingError("tank capacity exceeded")
            self.inventory_l[tank_id] = next_level
            self.version += 1
            return next_level

    def flight_summary(self, flight_id: str) -> Mapping[str, Decimal]:
        with self._lock:
            amount = sum(
                d.amount
                for d in self.deliveries.values()
                if d.flight_id == flight_id
            )
            return {
                "volume_l": self.flight_totals_l.get(flight_id, Decimal("0")),
                "mass_kg": self.flight_totals_kg.get(flight_id, Decimal("0")),
                "amount": quant(amount, MONEY),
            }

    def inventory_mass_kg(self, tank_id: str) -> Decimal:
        with self._lock:
            tank = self.tanks[tank_id]
            return quant(self.inventory_l[tank_id] * tank.density_kg_l)
