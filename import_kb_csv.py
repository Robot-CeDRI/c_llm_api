import csv
from datetime import datetime

from API.Utils.Databases.SQL_Database import DATABASE, Knowledge

CSV_PATH = "KnowledgeBase/IPB_data.csv"
DELIMITER = ";"


def import_csv_to_db():
    added = 0

    with DATABASE.session() as session:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=DELIMITER)

            for row in reader:
                user_q = (row.get("user") or "").strip()
                assistant_a = (row.get("assistant") or "").strip()

                if not user_q or not assistant_a:
                    continue

                # Documento que vai para a KB
                doc = f"Q: {user_q}\nA: {assistant_a}"

                session.add(
                    Knowledge(
                        tmstmp=datetime.now(),
                        document=doc
                    )
                )
                added += 1

        session.commit()

    print(f"[KB] Import concluído. Registos adicionados: {added}")


if __name__ == "__main__":
    import_csv_to_db()
