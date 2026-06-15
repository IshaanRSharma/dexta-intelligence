# Medical Disclaimer

**Read this before using dexta-intelligence with real health data.**

dexta-intelligence is software for **observation and discussion only**. It surfaces patterns,
statistics, and hypotheses from your own diabetes data so that you and your care team have
better evidence to talk about. It is a tool for understanding, not a tool for treatment.

## Not a medical device

dexta-intelligence is **not a medical device** and has not been reviewed, cleared, or approved
by the FDA or any other regulatory body. It is not certified, validated, or intended for
clinical diagnosis, monitoring, or treatment. The evals shipped with this project (E1, E4, E5)
are calibration and robustness checks on *synthetic* data — they are **not clinical
validation**.

## Not medical advice — and never dosing advice

Nothing dexta-intelligence produces — findings, hypotheses, briefs, chat answers, goal
progress, wiki pages — is medical advice. In particular:

- **Dexta never gives insulin dosing or treatment recommendations**, and is designed so that it
  cannot. The no-dosing rule and the faithfulness guard are not optional and not routable-out.
- **Do not make any treatment decision** — dosing, basal/bolus changes, correction, carb
  ratios, device settings, or anything else — based on dexta-intelligence output.
- All findings are **hypotheses, not prescriptions.** Statistical patterns over your history
  can be wrong, confounded, or not apply to your future.

## Consult your care team

Always consult your physician, diabetes care team, or other qualified healthcare professional
before acting on anything you see here, and before changing any aspect of your diabetes
management. If you think you may be having a medical emergency (severe hypo- or hyperglycemia,
DKA, or any urgent symptom), **contact your local emergency services or your care team
immediately** — do not consult this software.

## Data, connectors, and accuracy

- Dexta is **self-hosted**: your data is stored in a database you control and does not leave
  your infrastructure unless *you* configure a path that sends it elsewhere.
- If you configure a **hosted LLM provider**, your prompts (which include computed evidence
  numbers from your data) are sent to that provider under its terms and privacy policy.
- Several connectors use **unofficial / reverse-engineered APIs**. They may break without
  notice, may return incorrect or incomplete data, and are subject to each vendor's terms of
  service. Use them at your own risk. See `docs/CONNECTORS.md` for the per-source tiers.
- Source data (CGM, pump, wearable, manual entries) can be inaccurate, delayed, or missing.
  Dexta's output is only as good as the data it reads.

## No warranty

dexta-intelligence is provided "as is", without warranty of any kind, as stated in the
[LICENSE](LICENSE). The authors and contributors accept no liability for any decision made, or
harm arising, from use of this software. By using it you accept these terms and the medical
disclaimer above.
