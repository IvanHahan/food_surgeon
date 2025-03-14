import firebase_admin
from firebase_admin import credentials, db
from googletrans import Translator

from food_surgeon.config import FIREBASE_URL, SIA_FIREBASE_URL


def migrate_realtime_data(
    source_credential_path,
    target_credential_path,
    source_db_url,
    target_db_url,
    path
):
    # Initialize source Firebase app
    source_cred = credentials.Certificate(source_credential_path)
    firebase_admin.initialize_app(source_cred, {'databaseURL': source_db_url}, name='source')
    source_db = db.reference('/', firebase_admin.get_app('source'))

    # Initialize target Firebase app
    target_cred = credentials.Certificate(target_credential_path)
    firebase_admin.initialize_app(target_cred, {'databaseURL': target_db_url}, name='target')
    target_db = db.reference('/', firebase_admin.get_app('target'))

    # Initialize translator
    translator = Translator()

    # Fetch data from source
    data = source_db.child(path).get()

    # Translate and write to target
    def translate_data(data):
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                if key in ["description", "ingredients", "name"] and isinstance(value, str) and not value.startswith("https://"):
                    new_data[key] = translator.translate(value, src='ru', dest='uk').text
                else:
                    new_data[key] = translate_data(value)
            return new_data
        elif isinstance(data, list):
            return [translate_data(item) for item in data]
        else:
            return data

    translated_data = translate_data(data)
    flattened = {
        k: v
        for d in translated_data.values()
        if isinstance(d, dict)
        for k, v in d.items()
    }
    translated_data = flattened
    target_db.child(path).set(translated_data)

if __name__ == "__main__":
    # Example usage
    migrate_realtime_data(
        ".creds/sia_firebase.json",
        ".creds/ivan_firebase.json",
        SIA_FIREBASE_URL,
        FIREBASE_URL,
        "dishes"
    )
