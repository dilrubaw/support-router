HUMAN_TEAM = "Human agent"
SMALLTALK_TEAM = "Small-talk bot (rule-based)"


CUSTOM_TEAMS = {
    "appliance_fault": "Technical Service",
    "appliance_installation": "Installation Team",
    "appliance_warranty": "Warranty & Returns",
    "appliance_spare_parts": "Spare Parts",
}


CLINC_DOMAINS = {
    "banking": "transfer transactions balance freeze_account pay_bill bill_balance bill_due interest_rate routing min_payment order_checks pin_change report_fraud account_blocked spending_history",
    "credit_cards": "credit_score report_lost_card credit_limit rewards_balance new_card application_status card_declined international_fees apr redeem_rewards credit_limit_change damaged_card replacement_card_duration improve_credit_score expiration_date",
    "kitchen_dining": "recipe restaurant_reviews calories nutrition_info restaurant_suggestion ingredients_list ingredient_substitution cook_time food_last meal_suggestion restaurant_reservation confirm_reservation how_busy cancel_reservation accept_reservations",
    "home": "shopping_list shopping_list_update next_song play_music update_playlist todo_list todo_list_update calendar calendar_update what_song order order_status reminder reminder_update smart_home",
    "auto_commute": "traffic directions gas gas_type distance current_location mpg oil_change_when oil_change_how jump_start uber schedule_maintenance last_maintenance tire_pressure tire_change",
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert travel_notification carry_on timezone vaccines translate flight_status international_visa lost_luggage plug_type exchange_rate",
    "utility": "time alarm share_location find_phone weather text spelling make_call timer date calculator measurement_conversion flip_coin roll_dice definition",
    "work": "direct_deposit pto_request taxes payday w2 pto_balance pto_request_status next_holiday insurance insurance_change schedule_meeting pto_used meeting_schedule rollover_401k income",
    "small_talk": "greeting goodbye tell_joke where_are_you_from how_old_are_you what_is_your_name who_made_you thank_you what_can_i_ask_you what_are_your_hobbies do_you_have_pets are_you_a_bot meaning_of_life who_do_you_work_for fun_fact",
    "meta": "change_ai_name change_user_name cancel user_name reset_settings whisper_mode repeat no yes maybe change_language change_accent change_volume change_speed sync_device",
}

DOMAIN_TEAMS = {
    "banking": "Billing & Payments",
    "credit_cards": "Billing & Payments",
    "kitchen_dining": "Kitchen & Recipe Assistant",
    "home": "Smart Home & App",
    "auto_commute": "General Support",
    "travel": "General Support",
    "utility": "Smart Home & App",
    "work": "Corporate / HR",
    "small_talk": SMALLTALK_TEAM,
    "meta": "Smart Home & App",
}

INTENT_TO_DOMAIN = {i: d for d, s in CLINC_DOMAINS.items() for i in s.split()}


def team_for_intent(intent):
    if intent in CUSTOM_TEAMS:
        return CUSTOM_TEAMS[intent]
    if intent == "oos":
        return HUMAN_TEAM
    return DOMAIN_TEAMS.get(INTENT_TO_DOMAIN.get(intent, ""), "General Support")


def route(intent, confidence, threshold):
    """Düşük güven veya out-of-scope -> insan temsilci."""
    if intent == "oos" or confidence < threshold:
        return HUMAN_TEAM
    return team_for_intent(intent)
