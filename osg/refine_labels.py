from pathlib import Path
import json
import datetime
import os

import instructor
import openai

from osg.model import SprunggelenkMRTBericht

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

EXTRACT_PATH = Path(
    "/Users/admin/Projekte/Datenaufbereitung/Label Engineering/osg/extracted_labels/results"
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
        # model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        model="gpt-5",
        # model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        response_model=SprunggelenkMRTBericht,
        messages=[
            {
                "role": "system",
                "content": """# Sprunggelenk-MRT Datenextraktion

Extrahiere Daten gemäß dem Sprunggelenk-MRT-Schema.

## KRITISCHE REGEL - REFERENZ-PFLICHT FÜR POSITIVE DIAGNOSEN:

**JEDE POSITIVE PATHOLOGIE MUSS MIT ORIGINALTEXT BELEGT WERDEN**

- Positive Diagnose → `references` MUSS wortwörtliches Zitat enthalten
- Positive Diagnose ohne Referenz → UNGÜLTIG
- Negative Diagnose → `references = None` ist ausreichend

## LABEL-SCHEMATA UND REFERENCES:

### LabelWithReferences:
**Für definitive pathologische Befunde mit VERPFLICHTENDEN Referenzen**

- `references` MUSS wortwörtliche Zitate aus dem Befundtext enthalten
- **KEINE AUSNAHMEN:** Jede Pathologie braucht ihre Textquelle als Beleg
- Bei fehlender Referenz: Diagnose ist UNGÜLTIG

### LabelWithOptionalReferences:
**Für Befunde mit konditionalen Referenzen**

#### POSITIVE Befunde:
- **ZWINGEND:** `references` mit wortwörtlichen Zitaten füllen
- Gleiche Regel wie LabelWithReferences

#### NEGATIVE Befunde (KEIN, NEUTRAL, STABIL):
- `references = None` ist ausreichend
- Entstehen oft durch ABWESENHEIT pathologischer Beschreibungen

## REFERENCES - ORIGINALTEXT-REGEL:

### RICHTIG:
- **Exakte Textpassagen** aus dem MRT-Befund kopieren
- Wortwörtliche Übernahme ohne Änderungen
- Beispiel: Befundtext enthält "mäßiger Gelenkerguss im OSG" 
  → `references: ["mäßiger Gelenkerguss im OSG"]`

### FALSCH:
- Paraphrasierung oder Zusammenfassung
- Eigene Formulierungen
- Leere References bei positiven Diagnosen
- Interpretationen oder Schlussfolgerungen

### AUSNAHME - Keine Referenzen nötig:
- Negativbefunde: "Nicht abgrenzbar", "Kein Nachweis", etc.

## QUALITÄTSKONTROLLE - CHECKLISTE:

Vor Abgabe prüfen:
1. Alle positiven Diagnosen haben Originaltext-Referenzen?
2. Keine leeren References bei pathologischen Befunden?
3. Wortwörtliche Zitate ohne Eigeninterpretation?
4. Negative Befunde korrekt als solche klassifiziert?

## UNBEKANNTE_DIAGNOSEN - Schema-Lücken-Erfassung:

### Zweck:
Forschungstool zur Identifikation fehlender Pathologien im Schema. Dient der Schema-Verbesserung und -Vollständigkeit.

### Erfassungskriterien (alle müssen erfüllt sein):
1. **Gesicherte, definitiv beschriebene Pathologie**
2. **Keine passende Option** in 620+ verfügbaren Enum-Begriffen
3. **Medizinisch relevante Pathologie**
4. **Sprunggelenkbezogen**

### NICHT erfassen:
- Vermutungen: "möglich", "verdächtig", "DD:", "eventuell"
- Detailbeschreibungen bereits erfasster Pathologien
- Synonyme vorhandener Enum-Optionen
- Lokalisierungsdetails bereits erfasster Befunde
- Ausschlussdiagnosen: "kein Hinweis auf"

### Beispiele für echte Schema-Lücken:
- "Morton Neurom zwischen 3. und 4. Mittelfußknochen"
- "Lisfranc-Verletzung"  
- "Metatarsale Stressfraktur"

### Bei Unsicherheit:
1. Alle 620+ Enum-Optionen geprüft?
2. Wirklich schema-fremde Pathologie?
3. **Im Zweifel: NICHT in unbekannte_diagnosen**

---

## ZUSAMMENFASSUNG - KERNREGEL:
**POSITIVE DIAGNOSE = ORIGINALTEXT-REFERENZ PFLICHT**
**KEINE REFERENZ = UNGÜLTIGE DIAGNOSE**""",
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
    processing_index_file = f"/Users/admin/Projekte/Datenaufbereitung/Label Engineering/osg/extracted_labels/indices/index_{datetime.datetime.now()}.json"
    processed_studies = []

    for report_path in [
        r for r in reports if "1.2.840.113619.6.95.31.0.3.4.1.3096.13.263561" in str(r)
    ]:
        study_uid = report_path.name.split(".txt")[0]

        if study_uid in index_file:
            processed_studies.append(study_uid)
            save_processing_index(processing_index_file, processed_studies)
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        try:
            extracted_labels, completion = extract_data(report)
        except Exception as e:
            print(e)

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
        "/Users/admin/Projekte/Datenaufbereitung/Label Engineering/osg/reports"
    )
    # index_files = "/home/homesOnMaster/pehrlich/dataset_preparation/osg/extracted_labels/indices/index_2025-09-17 15:33:41.263418.json"

    # with open(index_files, "r", encoding="utf-8") as f:
    #     index_file = json.load(f) or []
    index_file = []

    refine_model(list(dir_path.glob("*.txt")), index_file=index_file)
