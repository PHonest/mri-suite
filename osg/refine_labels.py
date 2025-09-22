from pathlib import Path
import json
import datetime
import os

import instructor
import openai

from osg.model import SprunggelenkMRTBericht

API_KEY = os.getenv("API_KEY_HPC")
API_URL = os.getenv("API_URL_HPC")

EXTRACT_PATH = Path(
    "/home/homesOnMaster/pehrlich/dataset_preparation/osg/extracted_labels/results"
)

client = instructor.from_openai(
    openai.AzureOpenAI(
        api_key=API_KEY, azure_endpoint=API_URL, api_version="2024-08-01-preview"
    )
)
# client = instructor.from_openai(
#     openai.AzureOpenAI(
#         api_key=API_KEY, azure_endpoint=API_URL, api_version="2024-08-01-preview"
#     )
# )


def extract_data(report: str) -> SprunggelenkMRTBericht:
    return client.chat.completions.create_with_completion(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        response_model=SprunggelenkMRTBericht,
        messages=[
            {
                "role": "system",
                "content": """Extrahiere Daten gemäß dem Sprunggelenk-MRT-Schema.

  ## LABEL-SCHEMATA UND REFERENCES - KERNPRINZIP:

  ### LabelWithReferences:
  - Für definitive pathologische Befunde mit VERPFLICHTENDEN Referenzen
  - `references` MUSS wortwörtliche Zitate aus dem Befundtext enthalten
  - Jede Pathologie braucht ihre Textquelle als Beleg

  ### LabelWithOptionalReferences:
  - Für Befunde mit OPTIONALEN Referenzen, besonders Negativbefunde
  - Bei Negativbefunden (KEIN, NEUTRAL, STABIL): `references = None` ist ausreichend
  - Bei Positivbefunden: `references` mit wortwörtlichen Zitaten füllen
  - Negativbefunde entstehen oft durch ABWESENHEIT pathologischer Beschreibungen

  ### REFERENCES - WORTWÖRTLICHE ZITATE:
  - Exakte Textpassagen aus dem MRT-Befund kopieren
  - KEINE Paraphrasierung oder Zusammenfassung
  - KEINE eigenen Formulierungen
  - KEINE Referenzen für Negativ Befunde ["Nicht abgrenzbar", "Kein"]
  - Beispiel: "mäßiger Gelenkerguss im OSG" → references: ["mäßiger Gelenkerguss im OSG"]

  ## SUCHSTRATEGIE:
  - Nutze 620+ spezifische Suchbegriffe in Enum-Kommentaren
  - Kombiniere anatomische mit pathologischen Begriffen
  - Beachte Synonyme: "Fibularis" = "Peroneus", "Tendo Achillis" = "Achillessehne"
  - Erkenne Abkürzungen: LFTA, LFC, LFTP, SPR, IPR, OSG, USG

  ## ANATOMISCHE PRÄZISION:
  - OSG (oberes Sprunggelenk) vs USG (unteres Sprunggelenk)
  - Lokalisationen: medial/lateral/zentral am Talus/Tibia
  - Sehnendifferenzierung: Tibialis anterior vs posterior
  - Bandstrukturen: LFTA vs LFC vs LFTP

  ## UNBEKANNTE_DIAGNOSEN - Schema-Lücken-Erfassung:

  ### Zweck:
  Forschungstool zur Identifikation fehlender Pathologien im Schema. Dient der Schema-Verbesserung und -Vollständigkeit.

  ### Erfassungskriterien (alle müssen erfüllt sein):
  1. Gesicherte, definitiv beschriebene Pathologie
  2. Keine passende Option in 620+ verfügbaren Enum-Begriffen
  3. Medizinisch relevante Pathologie
  4. Sprunggelenkbezogen

  ### NICHT erfassen:
  - Vermutungen: "möglich", "verdächtig", "DD:", "eventuell"
  - Detailbeschreibungen bereits erfasster Pathologien
  - Synonyme vorhandener Enum-Optionen
  - Lokalisierungsdetails bereits erfasster Befunde
  - Ausschlussdiagnosen: "kein Hinweis auf"

  ### Beispiele NICHT als unbekannt (bereits im Schema):
  - "Dorsales Ganglion" → ganglion_oder_kapselaussackung: true
  - "Fibularis brevis Tendinopathie" → PERONEUS_BREVIS: [TENDINOPATHIE]
  - "Anteromediale osteochondrale Läsion" → lokalisation_talus: ANTEROMEDIAL
  - "Tenosynovitis der Peronealsehnen" → tenosynovitis_grad: MODERAT

  ### Beispiele für echte Schema-Lücken:
  - "Morton Neurom zwischen 3. und 4. Mittelfußknochen"
  - "Lisfranc-Verletzung"
  - "Metatarsale Stressfraktur"

  ### Bei Unsicherheit:
  1. Alle 620+ Enum-Optionen geprüft?
  2. Wirklich schema-fremde Pathologie?
  3. Im Zweifel: NICHT in unbekannte_diagnosen

  WICHTIG: References müssen exakte, wortwörtliche Zitate sein - keine Interpretationen! Bitte zitiere nur relevante Passagen. Wenn zwischen zwei relevanten Passagen (für das Label) unwichtiger Text liegt sollst du die Liste nutzen und zwei (oder mehrere) Textabschnitte angeben. Deswegen ist references als list[str] angegeben""",
            },
            {"role": "user", "content": report},
        ],
    )


def save_extraction_results(results: SprunggelenkMRTBericht, filename: str = None):
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extraction_results_{timestamp}.json"

    # Convert Pydantic model to dict, then save as JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    return filename


def save_processing_index(filename: str, content: list[str]):
    """Save a list of strings to a JSON file. Overwrites if file exists."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def refine_model(reports: list[Path], index_file: list[str]) -> None:
    processing_index_file = f"/home/homesOnMaster/pehrlich/dataset_preparation/osg/extracted_labels/indices/index_{datetime.datetime.now()}.json"
    processed_studies = []

    for report_path in reports:
        study_uid = report_path.name.split(".txt")[0]

        if study_uid in index_file:
            processed_studies.append(study_uid)
            save_processing_index(processing_index_file, processed_studies)
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        extracted_labels, completion = extract_data(report)

        if not extracted_labels.unbekannte_diagnosen:
            save_extraction_results(
                extracted_labels, filename=EXTRACT_PATH / f"{study_uid}.json"
            )

            processed_studies.append(study_uid)

            save_processing_index(processing_index_file, processed_studies)

        else:
            print("Found a new label")


if __name__ == "__main__":
    dir_path = Path(
        "/home/homesOnMaster/pehrlich/dataset_preparation/.export/osg/reports"
    )
    # index_files = "/home/homesOnMaster/pehrlich/dataset_preparation/osg/extracted_labels/indices/index_2025-09-17 15:33:41.263418.json"

    # with open(index_files, "r", encoding="utf-8") as f:
    #     index_file = json.load(f) or []
    index_file = []

    refine_model(list(dir_path.glob("*.txt")), index_file=index_file)
