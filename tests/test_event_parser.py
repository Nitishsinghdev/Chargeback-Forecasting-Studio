"""Natural-language event parsing."""



from __future__ import annotations



from services.event_parser import parse_event_batch, parse_event_text





def test_percent_date_bu_product_extraction():

    text = (

        'Holiday surge: +12% volume for Enterprise Cloud Services from Jan 2026 to Mar 2026'

    )

    event = parse_event_text(

        text,

        segment_candidates=["Enterprise", "Cloud Services"],

        business_units=["Enterprise", "Consumer"],

        products=["Cloud Services", "Mobile Plans"],

    )

    assert event.effect_type == "percent"

    assert event.effect_value == 12.0

    assert event.start_period is not None

    assert event.start_period.year == 2026

    assert event.start_period.month == 1

    assert event.end_period is not None and event.end_period.month == 3

    assert event.business_unit == "Enterprise"

    assert event.product == "Cloud Services"

    assert event.metric in {"volume", "both", "unknown"}





def test_no_number_yields_unknown_effect():

    event = parse_event_text("Regulatory review expected next quarter for Consumer")

    assert event.effect_type == "unknown"

    assert event.effect_value is None

    assert event.confidence <= 0.5





def test_decrease_percent_sign_from_words():

    event = parse_event_text("Fraud reduction: decrease volume by 8% in 2025-06")

    assert event.effect_type == "percent"

    assert event.effect_value == -8.0

    assert event.direction == "decrease"





def test_liability_absolute_amount():

    event = parse_event_text("Settlement: -$25000 liability for Wholesale in 2025-04")

    assert event.effect_type == "absolute"

    assert event.effect_value == -25000.0

    assert event.business_unit == "Wholesale"





def test_batch_preserves_order():

    lines = ["Event A: +5% in 2025-01", "Event B: +3% in 2025-02"]

    events = parse_event_batch(lines)

    assert len(events) == 2

    assert events[0].raw_text.startswith("Event A")

    assert events[1].effect_value == 3.0

