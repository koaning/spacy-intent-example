import typer
import srsly
from pathlib import Path
from spacy.tokens import DocBin
import spacy

CATEGORIES = ['oos', 'bill_due', 'pin_change', 'restaurant_reservation', 'current_location', 'uber', 'change_user_name', 'pto_request', 'find_phone', 'w2', 'routing', 'schedule_meeting', 'measurement_conversion', 'rollover_401k', 'confirm_reservation', 'account_blocked', 'change_accent', 'international_fees', 'thank_you', 'shopping_list', 'pto_used', 'cancel_reservation', 'insurance', 'todo_list', 'card_declined', 'timezone', 'order', 'income', 'plug_type', 'translate', 'alarm', 'pto_balance', 'application_status', 'last_maintenance', 'text', 'bill_balance', 'meaning_of_life', 'who_made_you', 'jump_start', 'apr', 'definition', 'meeting_schedule', 'calendar', 'reminder', 'lost_luggage', 'book_hotel', 'ingredient_substitution', 'roll_dice', 'international_visa', 'transfer', 'share_location', 'book_flight', 'nutrition_info', 'schedule_maintenance', 'payday', 'restaurant_suggestion', 'todo_list_update', 'reminder_update', 'damaged_card', 'direct_deposit', 'carry_on', 'change_volume', 'reset_settings', 'travel_suggestion', 'credit_limit', 'redeem_rewards', 'change_speed', 'pto_request_status', 'smart_home', 'flip_coin', 'order_checks', 'weather', 'traffic', 'date', 'how_busy', 'no', 'make_call', 'improve_credit_score', 'flight_status', 'accept_reservations', 'user_name', 'expiration_date', 'recipe', 'insurance_change', 'next_holiday', 'restaurant_reviews', 'order_status', 'exchange_rate', 'cancel', 'yes', 'replacement_card_duration', 'report_lost_card', 'food_last', 'credit_limit_change', 'credit_score', 'oil_change_when', 'where_are_you_from', 'spending_history', 'new_card', 'travel_notification', 'play_music', 'are_you_a_bot', 'goodbye', 'travel_alert', 'interest_rate', 'time', 'gas', 'who_do_you_work_for', 'update_playlist', 'report_fraud', 'do_you_have_pets', 'tire_change', 'fun_fact', 'what_are_your_hobbies', 'tire_pressure', 'what_song', 'how_old_are_you', 'next_song', 'vaccines', 'tell_joke', 'distance', 'calendar_update', 'shopping_list_update', 'what_can_i_ask_you', 'timer', 'ingredients_list', 'greeting', 'taxes', 'what_is_your_name', 'gas_type', 'meal_suggestion', 'pay_bill', 'balance', 'car_rental', 'oil_change_how', 'calories', 'transactions', 'directions', 'mpg', 'maybe', 'min_payment', 'change_language', 'rewards_balance', 'cook_time', 'change_ai_name', 'whisper_mode', 'repeat', 'calculator', 'sync_device', 'spelling', 'freeze_account']

def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    data_tuples = ((eg["text"], eg) for eg in srsly.read_jsonl(input_path))
    for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
        # doc.cats = {category: 0 for category in CATEGORIES}
        doc.cats[eg["label"]] = 1
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Processed {len(doc_bin)} documents: {output_path.name}")


if __name__ == "__main__":
    typer.run(main)