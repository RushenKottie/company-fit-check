"""Schema checks for non-deterministic regression case files."""

from evals.nondeterministic.case_loader import load_nondeterministic_cases


def test_non_deterministic_regression_cases_load_and_match_expected_shape():
    cases = load_nondeterministic_cases()

    assert len(cases) == 10
    assert {case.id for case in cases} == set(range(1, 11))
    assert {case.name for case in cases} == {
        "cardiologist_relocation_hospital_network",
        "illustrator_apac_stable_creative_role",
        "ml_engineer_product_focused_ai_company",
        "civil_engineer_international_infrastructure_projects",
        "electrical_engineer_apac_industrial_companies",
        "marketing_strategist_north_america_brands",
        "history_lecturer_academic_relocation_funding",
        "clinical_psychologist_ethical_healthcare_balance",
        "mechanical_engineer_climate_tech_transition",
        "devops_engineer_remote_latam_companies",
    }
    for case in cases:
        assert case.filter_criteria
        assert case.axes
        assert case.communication_style.description
        assert case.communication_style.behavioral_traits
        assert case.first_prompt
