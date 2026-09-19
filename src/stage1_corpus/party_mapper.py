"""
Political Entity Mapper

Loads, normalizes, and maps political party affiliations to debate speakers 
across multiple legislative terms.
"""


import json
import re
from pathlib import Path
import pandas as pd
import Levenshtein


TERMS = {"11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV"}
ROMAN_TO_NUM = {"XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15}
NUM_TO_ROMAN = {v: k for k, v in ROMAN_TO_NUM.items()}

class PartyLoader:
    """Manages speaker-to-party mappings across legislative terms (XI-XV)."""
    def __init__(self):
        self.parties = {}

    @staticmethod
    def normalize_name(name: str) -> str:
        return name.lower().strip().replace(" ", "").replace(",", "")

    @staticmethod
    def extract_term_from_filename(filename: str) -> str:
        match = re.search(r'(\d+)(?=\.json$)', filename)
        return TERMS.get(match.group(), "No disponible") if match else "No disponible"

    def load_from_json_folder(self, folder_path: str) -> None:
        folder = Path(folder_path)
        for json_file in folder.rglob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                term = self.extract_term_from_filename(json_file.name)
                for entry in data:
                    if "Nombre" in entry and entry["Nombre"]:
                        name = self.normalize_name(entry["Nombre"])
                        party = entry.get("Grupo", "No disponible")
                        if party:
                            self.parties[(name, term)] = party
            except Exception as e:
                print(f"Error leyendo {json_file.name}: {e}")

    def load_from_csvs(self, csv_files: list[str]) -> None:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["speaker"] = df["surname"].fillna("") + "," + df["name"].fillna("")
                for _, row in df.iterrows():
                    name = self.normalize_name(row.get("speaker", ""))
                    term = str(row.get("term", "No disponible"))
                    party = str(row.get("party", "No disponible"))
                    if party:
                        self.parties[(name, term)] = party
            except Exception as e:
                print(f"Error leyendo {csv_file}: {e}")

    def normalize_name(self, name: str) -> str:
        return name.lower().strip().replace(" ", "").replace(",", "")

    def assign_party(self, entry: dict, party: str) -> tuple[str,bool]:
        """Resolves party affiliation via exact match, Levenshtein distance, or term fallback."""
        term = entry.get("LEGISLATURA", "No disponible")
        orator = entry.get("ORADOR", "No disponible")
        key = (self.normalize_name(orator), term)
        entry["PARTIDO"] = party
        party_assigned = False

        while party == "No disponible":

            if key in self.parties:
                party = self.parties[key]

            elif party == "No disponible":
                similar_keys = [
                    k for k in self.parties.keys()
                    if (Levenshtein.ratio(str(key[0]), str(k[0])) >= 0.9 and key[1] == k[1])
                ]
                print(similar_keys)
                if similar_keys:
                    party = self.parties[similar_keys[-1]]

            if term in ROMAN_TO_NUM and term != "XI":
                term = ROMAN_TO_NUM[term] - 1
                term = NUM_TO_ROMAN[term]
                key = (self.normalize_name(orator), term)
            else:
                break

            if party != "No disponible":
                party_assigned = True

        entry["PARTIDO"] = party
        return entry.get("PARTIDO", "No disponible"), party_assigned

    def get_party_dict(self):
        return self.parties
