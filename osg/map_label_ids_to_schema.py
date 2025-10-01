from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path

from mongoengine import Document, fields

from osg.model import SprunggelenkMRTBericht


class Label(Document):
    meta = {"collection": "labels", "strict": False}

    parentID: str = fields.ObjectIdField(null=True)
    text: str = fields.StringField(required=True)
    schemaName: str = fields.StringField(required=True)


class Edit(Document):
    meta = {"collection": "edits", "strict": False}

    reportID: Report = fields.ReferenceField("Report", required=True)
    labelID: Label = fields.ReferenceField("Label", required=True)
    commentHeadID: str = fields.StringField()
    highlightedText: str = fields.StringField(required=True)
    startOffset: int = fields.IntField(required=True)
    endOffset: int = fields.IntField(required=True)


class Report(Document):
    meta = {"collection": "reports", "strict": False}

    isStarred: bool = fields.BooleanField()
    text: str = fields.StringField()
    externalID: str = fields.StringField()


def extract_child_labels(
    labels: list[Label], return_leaf_ids: bool = True
) -> dict[str, list[str]]:
    """Returns a dict of all child labels with their string path as key (values on path a seperated by "_") and their paths as List of ObjectIDs as values

        i.e.
            {
                "Gelenkflüssigkeiten_OSG Erguss_Kein": ["68c2b5031e20419d5a283c62", "68c2b5031e20419d5a283c64", "68c2b5031e20419d5a283c66"]
            }
    My labels are grouped as a tree
    """
    # Find root labels (those with parentID = None)
    root_labels = [label for label in labels if label.parentID is None]

    result = {}

    def build_paths(label: Label, current_path_texts, current_path_ids):
        # Add current label to the paths
        new_path_texts = current_path_texts + [label.schemaName]
        new_path_ids = current_path_ids + [str(label.id)]

        # Find children of current label
        children = [l for l in labels if l.parentID == label.id]

        if not children:
            # This is a leaf node, add it to results
            path_key = "_".join(new_path_texts)
            result[path_key] = new_path_ids
        else:
            # Recursively process children
            for child in children:
                build_paths(child, new_path_texts, new_path_ids)

    # Start building paths from each root label
    for root in root_labels:
        build_paths(root, [], [])

    if return_leaf_ids:
        return {k: v[-1] for k, v in result.items()}

    return result


def match_extraction_on_label_tree(
    extraction: SprunggelenkMRTBericht,
    parsed_label_tree: dict[str, str],
    report_id: str,
    report_text: str,
) -> list[Edit]:
    """Create an edit instance for each parsed label in extractions and each reference in it (so there can be multiple Edit instance per label). Additionally for each selected a label a dummy Edit is created:

    {
        "_id" : ObjectId("xxx"),
        "reportID" : ObjectId("yyy"),
        "labelID" : ObjectId("zzz"),
        "commentHeadID" : null,
        "highlightedText" : "[LABEL_SELECTION]",
        "startOffset" : NumberInt(-1),
        "endOffset" : NumberInt(-1),
        "__v" : NumberInt(0)
    }

    Args:
        extraction: the actual parsed labels from the report
        parsed_labeL_tree: the path of label names for each child label and its LabelObjectID (important for the Edit reference)

    Returns:
        list[Edit]: The parsed labels from the extraction embedded in the edits

    """
    edits = []

    def process_label_with_references(label_path: str, label_obj):
        """Process a label that has references"""
        if label_path not in parsed_label_tree:
            return

        # print(label_path)
        label_id = parsed_label_tree[label_path]

        # Create dummy edit for label selection
        selection_edit = Edit(
            reportID=report_id,
            labelID=label_id,
            commentHeadID=None,
            highlightedText="[LABEL_SELECTION]",
            startOffset=-1,
            endOffset=-1,
        )
        edits.append(selection_edit)

        # Create edits for each reference
        if hasattr(label_obj, "references") and label_obj.references:
            for reference in label_obj.references:
                start_offset, end_offset = calculate_offsets(reference, report_text)
                reference_edit = Edit(
                    reportID=report_id,
                    labelID=label_id,
                    commentHeadID=None,
                    highlightedText=reference,
                    startOffset=start_offset,
                    endOffset=end_offset,
                )
                edits.append(reference_edit)

    def process_model_recursively(obj, path_prefix=""):
        """Recursively process the extraction model"""
        if hasattr(obj, "__dict__"):
            for field_name, field_value in obj.__dict__.items():
                if "__" in field_name or field_name[0] == "_":
                    continue

                current_path = (
                    f"{path_prefix}_{field_name}" if path_prefix else field_name
                )

                if field_value is None:
                    continue

                # Handle lists
                if isinstance(field_value, list):
                    for item in field_value:
                        if hasattr(item, "value"):
                            # This is a LabelWithReferences or LabelWithOptionalReferences
                            value_path = f"{current_path}_{item.value.name if hasattr(item.value, 'name') else str(item.value)}"
                            process_label_with_references(value_path, item)
                        else:
                            process_model_recursively(item, current_path)

                # Handle single objects with value field
                elif hasattr(field_value, "value"):
                    if isinstance(field_value.value, Enum):
                        value_path = f"{current_path}_{field_value.value.name if hasattr(field_value.value, 'name') else str(field_value.value)}"
                        process_label_with_references(value_path, field_value)
                    if isinstance(field_value.value, bool):
                        if not field_value.value:
                            continue
                        process_label_with_references(current_path, field_value)
                    if hasattr(field_value.value, "__dict__"):
                        process_model_recursively(field_value.value, current_path)

                # Handle nested models
                elif hasattr(field_value, "__dict__"):
                    process_model_recursively(field_value, current_path)

    # Process the extraction
    process_model_recursively(deepcopy(extraction))

    return edits


def calculate_offsets(reference: str, report_text: str) -> tuple[int, int]:
    """Calculates start and end offset for the reference text"""
    start_offset = report_text.find(reference)
    if start_offset == -1:
        print(f"Could not find {reference} in report text")
        return (-1, -1)
    end_offset = start_offset + len(reference)
    return (start_offset, end_offset)


def read_jsonified_schema(json_path: str) -> SprunggelenkMRTBericht:
    """Read a JSON file created by save_extraction_results and return a SprunggelenkMRTBericht model

    Args:
        json_path: Path to the JSON file created by save_extraction_results

    Returns:
        SprunggelenkMRTBericht: Parsed Pydantic model instance
    """
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SprunggelenkMRTBericht.model_validate(data)


if __name__ == "__main__":
    from mongoengine import connect

    connect(host="mongodb://localhost:27017/report-label-editor")

    all_labels = list(Label.objects().all())
    ch_labels = extract_child_labels(all_labels)

    extractions = [
        # (
        #     Path(
        #         "osg/extracted_labels/results/1.2.840.113619.6.95.31.0.3.4.1.3096.13.252199.json"
        #     ),
        #     Path("osg/reports/1.2.840.113619.6.95.31.0.3.4.1.3096.13.252199.txt"),
        # ),
        # (
        #     Path(
        #         "osg/extracted_labels/results/1.2.840.113619.6.95.31.0.3.4.1.3096.13.252228.json"
        #     ),
        #     Path("osg/reports/1.2.840.113619.6.95.31.0.3.4.1.3096.13.252228.txt"),
        # ),
        # (
        #     Path(
        #         "osg/extracted_labels/results/1.2.840.113619.6.95.31.0.3.4.1.3096.13.255909.json"
        #     ),
        #     Path("osg/reports/1.2.840.113619.6.95.31.0.3.4.1.3096.13.255909.txt"),
        # ),
        (
            Path(
                "osg/extracted_labels/results/1.2.840.113619.6.95.31.0.3.4.1.3096.13.263561.json"
            ),
            Path("osg/reports/1.2.840.113619.6.95.31.0.3.4.1.3096.13.263561.txt"),
        ),
        (
            Path(
                "osg/extracted_labels/results/1.2.840.113619.6.95.31.0.3.4.1.3096.13.260937.json"
            ),
            Path("osg/reports/1.2.840.113619.6.95.31.0.3.4.1.3096.13.260937.txt"),
        ),
    ]

    for extr_schema, report_text_path in extractions:
        with open(report_text_path, "r", encoding="utf-8") as f:
            report = f.read()

        schema = read_jsonified_schema(extr_schema)
        edits = match_extraction_on_label_tree(
            extraction=schema,
            parsed_label_tree=ch_labels,
            report_id="1",
            report_text=report,
        )

        report = list(
            Report.objects(externalID=report_text_path.name.split(".txt")[0]).all()
        )[0]

        for edit in edits:
            edit.reportID = report
            edit.save()

        continue

    print("Done")
