import json
import re
from pathlib import Path
from collections import defaultdict

# Load the JSON data
with open("atis.json", "r") as f:
    atis_data = json.load(f)

# Helper function to normalize whitespace in SQL
def normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip())

# Step 1: Extract unique shortest SQL templates with placeholders
sql_templates = {}
template_to_id = {}
template_id_counter = 0

# Will store: list of (text_with_vars_replaced, template_id)
template_classification_data = []

# Will store: list of (tokenized_text, tags)
tagging_data = []

# Collect all variable names
all_tags = set()

for item in atis_data:
    sql_list = item['sql']
    # Choose the shortest SQL (by character count, then alphabetically)
    shortest_sql = sorted(sql_list, key=lambda x: (len(x), x))[0]
    normalized_sql = normalize_sql(shortest_sql)

    # Assign template ID
    if normalized_sql not in template_to_id:
        template_to_id[normalized_sql] = template_id_counter
        sql_templates[template_id_counter] = normalized_sql
        template_id_counter += 1

    template_id = template_to_id[normalized_sql]

    # Process all sentences
    for sentence in item['sentences']:
        text = sentence['text']
        variables = sentence['variables']

        # Replaced text for classification task
        replaced_text = text
        for var, val in variables.items():
            replaced_text = replaced_text.replace(var, val)

        template_classification_data.append((replaced_text, template_id))

        # Tagging: assign tags word by word
        words = replaced_text.split()
        tags = ['O'] * len(words)

        for var, val in variables.items():
            val_tokens = val.split()
            for i in range(len(words)):
                if words[i:i+len(val_tokens)] == val_tokens:
                    tags[i] = var
                    for j in range(1, len(val_tokens)):
                        tags[i + j] = var  # Use the same tag for now
                    break

            all_tags.add(var)

        tagging_data.append((words, tags))

# Create final tag list
tag_list = sorted(list(all_tags)) + ['O']

import pandas as pd

# Prepare dataframes for visualization
template_df = pd.DataFrame([
    {"template_id": tid, "sql_template": sql_templates[tid]}
    for tid in sql_templates
])

classification_df = pd.DataFrame(template_classification_data, columns=["text", "template_id"])

tagging_df = pd.DataFrame(tagging_data, columns=["tokens", "tags"])


classification_df.head(), tagging_df.head(), tag_list
