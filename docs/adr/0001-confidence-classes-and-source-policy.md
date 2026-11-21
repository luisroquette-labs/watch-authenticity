# Confidence, model classes, and source policy

## Context

This prototype must avoid absolute authenticity claims and must not treat a seller's description as proof. Angle heads, rejection, calibration, and downstream reports share one class vocabulary.

## Decisions

- Use `ModelClass` for the number and positions of model outputs. Genuine is at index zero; rejection is the final class. All per-angle heads and rejection/calibration readers use those positions.
- Bound reported confidence between 0.1% and 99.9%, so a score never reads as a certain verdict.
- Cap a grade-4 replica's genuine probability at 60%. This is the owner's provisional policy, not a measured detection rate; it needs held-out factory evaluation before recalibration.
- Treat OLX and Mercado Livre seller claims as untrusted. Owner forums identify candidates only; named replica forums and vendors are discovery sources, not authenticity evidence.
- Recognize Clean and VSF as named superclone factories. Unknown factories receive the conservative high-grade replica default. With no named factory, a listing price below 50 maps to the low-grade category.

## Consequences

The output vocabulary and safety limits remain consistent across stages. Source claims cannot alone establish authenticity. The 60% ceiling and 50-price threshold remain policy assumptions until evaluation data supports a change.
