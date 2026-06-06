"""Smoke test for dynamic specialist pool generation."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from GDesigner.prompt.dynamic_prompt import generate_specialist_panel

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-32B")

SAMPLE_QUESTIONS = [
    "A 4-year-old boy is brought to the emergency department with intense crying and pain in both hands after playing with ice cubes. His mother denies any preceding trauma. The temperature is 37.0°C (98.6°F), the blood pressure is 90/55 mm Hg, and the pulse is 100/min. The physical examination shows swollen dorsa of the hands and scleral icterus. The laboratory tests show hemoglobin of 10.1 g/dL and unconjugated hyperbilirubinemia. The cellulose acetate electrophoresis shows 60% HbS and absence of HbA. Which of the following can reduce the recurrence of the patient's current condition?\nOption A: Avoidance of sulfa drugs\nOption B: Vaccinations\nOption C: Hydroxyurea\nOption D: Folic acid\nOption E: Allopurinol",

    "A 65-year-old male presents to his pulmonologist for a follow-up visit. He has a history of chronic progressive dyspnea over the past five years. He uses oxygen at home and was seen in the emergency room two months prior for an exacerbation of his dyspnea. His past medical history is notable for hyperlipidemia and hypertension. He drinks alcohol socially and has a 45 pack-year smoking history. Which of the following will most likely appear in his PFT report?\nOption A: Residual volume increased, total lung capacity decreased\nOption B: Residual volume increased, total lung capacity increased\nOption C: Residual volume decreased, total lung capacity increased\nOption D: Residual volume normal, total lung capacity normal\nOption E: Residual volume normal, total lung capacity decreased",
]


async def main():
    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"Question {i+1}: {q[:100]}...")
        panel = await generate_specialist_panel(
            judge_model=JUDGE_MODEL,
            question=q,
        )
        if panel is None:
            print("  FAILED — returned None")
            continue
        print(f"  Generated {len(panel)} specialists:")
        for j, entry in enumerate(panel):
            print(f"    {j+1}. {entry['role']}")
            print(f"       {entry['description'][:120]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
