from mongoengine import Document, fields


class Label(Document):
    meta = {"collection": "labels"}

    parentID = fields.ObjectIdField(null=True)
    text = fields.StringField(required=True)


def extract_child_labels(labels: list[Label]) -> dict[str, list[str]]:
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

    def build_paths(label, current_path_texts, current_path_ids):
        # Add current label to the paths
        new_path_texts = current_path_texts + [label.text]
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

    return result
