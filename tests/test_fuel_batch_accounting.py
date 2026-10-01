import unittest
from datetime import datetime, timezone

from decimal import Decimal

from rampflow.fuel_accounting import (
    FuelBatchEntry,
    FuelLedger,
    FuelMeasurementBasis,
    FuelTank,
)
from rampflow.ops_store import VersionConflict
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc
AT = datetime(2026, 1, 1, 10, tzinfo=UTC)


def ledger():
    return FuelLedger(
        [FuelTank("T1", Decimal("10000"), Decimal("1050"), Decimal("0.8"))]
    )


def entries():
    return [
        FuelBatchEntry(
            "D1", "F1", "T1", Decimal("0"), Decimal("1000"),
            Decimal("25"), Decimal("0.80"), Decimal("1.20"), AT,
        ),
        FuelBatchEntry(
            "D2", "F1", "T1", Decimal("1000"), Decimal("1050"),
            Decimal("25"), Decimal("0.80"), Decimal("1.20"), AT,
        ),
    ]


class FuelBatchAccountingTests(unittest.TestCase):
    def test_batch_uses_fixed_basis_and_commits_atomically(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        receipt = fuel.commit_batch("BATCH-1", entries(), basis, expected_version=0)
        self.assertEqual([d.delivery_id for d in receipt.deliveries], ["D1", "D2"])
        self.assertAlmostEqual(receipt.deliveries[0].corrected_l, Decimal("990.589"), places=3)
        self.assertEqual(fuel.inventory_l["T1"], Decimal("9.882"))
        self.assertEqual(fuel.version, 1)
        for delivery in receipt.deliveries:
            self.assertIs(delivery.measurement_basis, basis)

    def test_failed_prevalidation_leaves_no_partial_records(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        bad_entries = [
            FuelBatchEntry(
                "D1", "F1", "T1", Decimal("0"), Decimal("2000"),
                Decimal("25"), Decimal("0.80"), Decimal("1.20"), AT,
            ),
            entries()[1],
        ]
        with self.assertRaises(Exception):
            fuel.commit_batch("BATCH-1", bad_entries, basis, expected_version=0)
        self.assertEqual(fuel.deliveries, {})
        self.assertEqual(fuel.batches, {})
        self.assertEqual(fuel.inventory_l["T1"], Decimal("1050.000"))
        self.assertEqual(fuel.version, 0)

    def test_stale_version_rejects_replayed_batch(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        fuel.replenish("T1", Decimal("100"))
        with self.assertRaises(VersionConflict):
            fuel.commit_batch("BATCH-1", entries(), basis, expected_generation=0)
        self.assertEqual(fuel.generation, 1)

    def test_same_batch_replay_is_idempotent_even_with_stale_expected_version(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        first = fuel.commit_batch("BATCH-1", entries(), basis, expected_version=0)
        second = fuel.commit_batch("BATCH-1", entries(), basis, expected_version=0)
        self.assertIs(first, second)
        self.assertEqual(len(fuel.deliveries), 2)
        self.assertEqual(fuel.inventory_l["T1"], Decimal("9.882"))

    def test_reverse_uses_corrected_volume_and_is_idempotent(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        receipt = fuel.commit_batch("BATCH-1", entries(), basis)
        before = fuel.inventory_l["T1"]
        first = fuel.reverse("D1", "R1", AT)
        second = fuel.reverse("D1", "R1", AT)
        self.assertIs(first, second)
        self.assertEqual(first.corrected_l, receipt.deliveries[0].corrected_l.copy_negate())
        self.assertEqual(fuel.inventory_l["T1"], before + receipt.deliveries[0].corrected_l)
        self.assertEqual(fuel.flight_totals_l["F1"], receipt.deliveries[1].corrected_l)
        self.assertEqual(fuel.version, 2)

    def test_workflow_does_not_publish_failed_or_duplicate_batch(self):
        fuel = ledger()
        basis = FuelMeasurementBasis("BASIS-1", Decimal("15"))
        control = RampWorkflowControl()
        calls = []

        def commit(valid=True):
            def action():
                calls.append(1)
                batch_entries = entries()
                if not valid:
                    batch_entries[0] = FuelBatchEntry(
                        "D1", "F1", "T1", Decimal("0"), Decimal("2000"),
                        Decimal("25"), Decimal("0.80"), Decimal("1.20"), AT,
                    )
                return fuel.commit_batch("BATCH-1", batch_entries, basis, expected_version=0)
            return action

        with self.assertRaises(Exception):
            control.commit_fuel_ledger_batch("OP-1", "BATCH-1", AT, commit(False))
        self.assertEqual(control.stream.head, 0)
        first = control.commit_fuel_ledger_batch("OP-1", "BATCH-1", AT, commit(True))
        second = control.commit_fuel_ledger_batch("OP-1", "BATCH-1", AT, commit(True))
        self.assertIs(first, second)
        self.assertEqual(calls, [1, 1])
        self.assertEqual(control.stream.head, 1)


if __name__ == "__main__":
    unittest.main()
