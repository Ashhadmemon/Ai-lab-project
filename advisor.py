from __future__ import annotations

from dataclasses import dataclass, asdict

NON_ACTIONABLE = {"Age", "ParentalEducation", "ParentalSupport"}

_PHRASING = {
    "StudyTimeWeekly": "studying about {delta:.1f} more hour(s) a week",
    "Absences": "cutting absences by about {delta:.0f} day(s)",
    "Tutoring": "starting tutoring",
    "Extracurricular": "joining an extracurricular activity",
    "Sports": "getting involved in a sport",
    "Music": "getting involved in music",
    "Volunteering": "starting volunteering",
}

_MIN_GAIN = 0.01


@dataclass
class Suggestion:
    feature: str 
    action: str
    estimated_gain: float

    def as_dict(self) -> dict:
        return asdict(self)


class ImprovementAdvisor:

    def advise(self, clean_input: dict, current_gpa: float, rules: dict,
               top_n: int = 3) -> list[Suggestion]:
        raise NotImplementedError


class LinearCoefficientAdvisor(ImprovementAdvisor):

    def __init__(self, coefficients: dict):
        self._coefficients = dict(coefficients)

    def advise(self, clean_input: dict, current_gpa: float, rules: dict,
               top_n: int = 3) -> list[Suggestion]:
        candidates: list[Suggestion] = []
        headroom = 4.0 - current_gpa

        for name, coef in self._coefficients.items():
            if name in NON_ACTIONABLE or name not in rules:
                continue

            lo, hi, _is_int = rules[name]
            current = clean_input[name]
            target = hi if coef > 0 else lo
            delta = target - current
            if abs(delta) < 1e-9:
                continue  # already at the best achievable value for this input

            gain = coef * delta
            if gain < _MIN_GAIN:
                continue

            gain = min(gain, headroom) if headroom > 0 else 0.0
            if gain < _MIN_GAIN:
                continue

            phrase = _PHRASING.get(name, name + " (change toward {delta:.1f})")
            candidates.append(Suggestion(
                feature=name,
                action=phrase.format(delta=abs(delta)),
                estimated_gain=round(gain, 2),
            ))

        candidates.sort(key=lambda s: s.estimated_gain, reverse=True)
        return candidates[:top_n]
