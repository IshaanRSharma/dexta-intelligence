"""Synthetic data generators with planted ground-truth effects for evaluation."""

from dexta_intelligence.testing.synthetic import (
    BaselineConfig,
    EventsByType,
    PlantedEffect,
    PostWorkoutHypo,
    ScenarioManifest,
    SensitivityRegimeShift,
    SleepQualityAssociation,
    WeekdayBreakfastSpike,
    generate_baseline,
    generate_dataset,
    generate_null,
    scenario_all,
    scenario_post_workout_hypo,
    scenario_sensitivity_shift,
    scenario_sleep_quality,
    scenario_weekday_breakfast,
)

__all__ = [
    "BaselineConfig",
    "EventsByType",
    "PlantedEffect",
    "PostWorkoutHypo",
    "ScenarioManifest",
    "SensitivityRegimeShift",
    "SleepQualityAssociation",
    "WeekdayBreakfastSpike",
    "generate_baseline",
    "generate_dataset",
    "generate_null",
    "scenario_all",
    "scenario_post_workout_hypo",
    "scenario_sensitivity_shift",
    "scenario_sleep_quality",
    "scenario_weekday_breakfast",
]
