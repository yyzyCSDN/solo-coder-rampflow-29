from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Mapping

from .timeutil import utc


Q = Decimal("0.001")
MONEY = Decimal("0.01")


class FuelAccountingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MeasurementBasis:
    """The metering convention a delivery is booked under.

    Pinned per ledger and snapshotted onto every delivery so a later reversal
    is anchored to the same basis instead of whatever the ledger uses then.
    """

    reference_temp_c: Decimal = Decimal("15")
    coefficient: Decimal = Decimal("0.00095")

    def corrected(self, volume_l: Decimal, temp_c: Decimal) -> Decimal:
        factor = Decimal("1") + self.coefficient * (temp_c - self.reference_temp_c)
        if factor <= 0:
            raise FuelAccountingError("invalid thermal correction")
        return quant(volume_l / factor)


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
    basis: MeasurementBasis = MeasurementBasis()


@dataclass(frozen=True)
class FuelDeliveryRequest:
    delivery_id: str
    flight_id: str
    tank_id: str
    meter_start_l: Decimal
    meter_end_l: Decimal
    temperature_c: Decimal
    measured_density_kg_l: Decimal
    price_per_l: Decimal
    delivered_at: datetime


def quant(value: Decimal, unit: Decimal = Q) -> Decimal:
    return value.quantize(unit, rounding=ROUND_HALF_UP)


class FuelLedger:
    def __init__(
        self,
        tanks: Iterable[FuelTank],
        reference_temp_c: Decimal = Decimal("15"),
        coefficient: Decimal = Decimal("0.00095"),
    ):
        self.tanks = {t.tank_id: t for t in tanks}
        self.basis = MeasurementBasis(reference_temp_c, coefficient)
        self.deliveries: dict[str, FuelDelivery] = {}
        self.inventory_l: dict[str, Decimal] = {
            t.tank_id: quant(t.usable_l) for t in tanks
        }
        self.flight_totals_l: dict[str, Decimal] = {}
        self.flight_totals_kg: dict[str, Decimal] = {}

    @property
    def reference_temp_c(self) -> Decimal:
        return self.basis.reference_temp_c

    def _matches(self, existing: FuelDelivery, req: FuelDeliveryRequest) -> bool:
        return (
            existing.flight_id == req.flight_id
            and existing.tank_id == req.tank_id
            and existing.meter_start_l == req.meter_start_l
            and existing.meter_end_l == req.meter_end_l
            and existing.temperature_c == req.temperature_c
            and existing.density_kg_l == req.measured_density_kg_l
            and existing.price_per_l == req.price_per_l
        )

    def _plan(self, req: FuelDeliveryRequest) -> FuelDelivery:
        """Build the booked delivery under the pinned basis, or raise. No state changes."""
        if req.tank_id not in self.tanks:
            raise KeyError(req.tank_id)
        raw = req.meter_end_l - req.meter_start_l
        if raw <= 0:
            raise FuelAccountingError("fuel meter must advance")
        corrected = self.basis.corrected(raw, req.temperature_c)
        mass = quant(corrected * req.measured_density_kg_l)
        amount = quant(corrected * req.price_per_l, MONEY)
        return FuelDelivery(
            delivery_id=req.delivery_id,
            flight_id=req.flight_id,
            tank_id=req.tank_id,
            meter_start_l=req.meter_start_l,
            meter_end_l=req.meter_end_l,
            temperature_c=req.temperature_c,
            density_kg_l=req.measured_density_kg_l,
            delivered_at=utc(req.delivered_at),
            corrected_l=corrected,
            mass_kg=mass,
            price_per_l=req.price_per_l,
            amount=amount,
            basis=self.basis,
        )

    def _apply(self, delivery: FuelDelivery) -> None:
        self.inventory_l[delivery.tank_id] = quant(
            self.inventory_l[delivery.tank_id] - delivery.corrected_l
        )
        self.flight_totals_l[delivery.flight_id] = quant(
            self.flight_totals_l.get(delivery.flight_id, Decimal("0")) + delivery.corrected_l
        )
        self.flight_totals_kg[delivery.flight_id] = quant(
            self.flight_totals_kg.get(delivery.flight_id, Decimal("0")) + delivery.mass_kg
        )
        self.deliveries[delivery.delivery_id] = delivery

    def _record_one(self, req: FuelDeliveryRequest) -> FuelDelivery:
        existing = self.deliveries.get(req.delivery_id)
        if existing is not None:
            if self._matches(existing, req):
                return existing
            raise FuelAccountingError(
                f"delivery id reused with different measurements: {req.delivery_id}"
            )
        delivery = self._plan(req)
        if delivery.corrected_l > self.inventory_l[req.tank_id]:
            raise FuelAccountingError("insufficient tank inventory")
        self._apply(delivery)
        return delivery

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
    ) -> FuelDelivery:
        return self._record_one(
            FuelDeliveryRequest(
                delivery_id,
                flight_id,
                tank_id,
                meter_start_l,
                meter_end_l,
                temperature_c,
                measured_density_kg_l,
                price_per_l,
                delivered_at,
            )
        )

    def record_batch(
        self, requests: Iterable[FuelDeliveryRequest]
    ) -> tuple[FuelDelivery, ...]:
        """All-or-nothing backfill with whole-batch pre-validation.

        Every request is validated up front — duplicate ids, unknown tanks,
        meter advance, thermal factor, and a running inventory simulation in
        batch order — before any delivery is committed, so a conflict late in
        the batch cannot leave earlier records written. Replaying an identical
        batch returns the already-booked deliveries without drawing inventory
        again; reusing an id with different measurements is a conflict.
        """
        results: list[FuelDelivery] = []
        new: list[FuelDelivery] = []
        seen: set[str] = set()
        scratch = dict(self.inventory_l)
        for req in requests:
            if req.delivery_id in seen:
                raise FuelAccountingError(
                    f"duplicate delivery id in batch: {req.delivery_id}"
                )
            seen.add(req.delivery_id)
            existing = self.deliveries.get(req.delivery_id)
            if existing is not None:
                if not self._matches(existing, req):
                    raise FuelAccountingError(
                        f"delivery id reused with different measurements: {req.delivery_id}"
                    )
                results.append(existing)
                continue
            delivery = self._plan(req)
            if delivery.corrected_l > scratch[req.tank_id]:
                raise FuelAccountingError(
                    f"insufficient tank inventory: {req.delivery_id}"
                )
            scratch[req.tank_id] = quant(scratch[req.tank_id] - delivery.corrected_l)
            new.append(delivery)
            results.append(delivery)
        for delivery in new:
            self._apply(delivery)
        return tuple(results)

    def reverse(self, delivery_id: str, reversal_id: str, reversed_at: datetime) -> FuelDelivery:
        """Offset a delivery by exactly the figures that were booked.

        The reversal negates the booked corrected volume, mass, and amount —
        never the raw metered volume — so inventory returns to the pre-delivery
        level at any temperature. Repeating the same reversal_id is an
        idempotent replay; a different reversal_id for the same delivery, or a
        reversal_id that collides with any existing record, is rejected.
        """
        original = self.deliveries[delivery_id]
        if original.reversed_by is not None:
            if original.reversed_by == reversal_id:
                return self.deliveries[reversal_id]
            raise FuelAccountingError("delivery already reversed")
        if reversal_id in self.deliveries:
            raise FuelAccountingError("duplicate reversal id")
        reversal = FuelDelivery(
            delivery_id=reversal_id,
            flight_id=original.flight_id,
            tank_id=original.tank_id,
            meter_start_l=original.meter_end_l,
            meter_end_l=original.meter_start_l,
            temperature_c=original.temperature_c,
            density_kg_l=original.density_kg_l,
            delivered_at=utc(reversed_at),
            corrected_l=-original.corrected_l,
            mass_kg=-original.mass_kg,
            price_per_l=original.price_per_l,
            amount=-original.amount,
            basis=original.basis,
        )
        self.deliveries[delivery_id] = replace(original, reversed_by=reversal_id)
        self.deliveries[reversal_id] = reversal
        self.inventory_l[original.tank_id] = quant(
            self.inventory_l[original.tank_id] + original.corrected_l
        )
        self.flight_totals_l[original.flight_id] = quant(
            self.flight_totals_l[original.flight_id] - original.corrected_l
        )
        self.flight_totals_kg[original.flight_id] = quant(
            self.flight_totals_kg[original.flight_id] - original.mass_kg
        )
        return reversal

    def replenish(self, tank_id: str, amount_l: Decimal) -> Decimal:
        if amount_l <= 0:
            raise FuelAccountingError("replenishment must be positive")
        tank = self.tanks[tank_id]
        next_level = quant(self.inventory_l[tank_id] + amount_l)
        if next_level > tank.capacity_l:
            raise FuelAccountingError("tank capacity exceeded")
        self.inventory_l[tank_id] = next_level
        return next_level

    def flight_summary(self, flight_id: str) -> Mapping[str, Decimal]:
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
        tank = self.tanks[tank_id]
        return quant(self.inventory_l[tank_id] * tank.density_kg_l)
