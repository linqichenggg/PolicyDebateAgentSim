import pandas as pd

def _get_row_value(row, keys, default=""):
    """Return the first non-null value for candidate column names."""
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def _to_float(value, default=0.0):
    """Convert a value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_real_users(file_path):
    """Load user profiles from CSV/XLSX into normalized agent dictionaries."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if df.empty:
            raise ValueError("User data file is empty")

        users = []

        for _, row in df.iterrows():
            traits = {
                "openness": _get_row_value(row, ["openness"], "medium"),
                "conscientiousness": _get_row_value(row, ["conscientiousness"], "medium"),
                "extraversion": _get_row_value(row, ["extraversion"], "medium"),
                "agreeableness": _get_row_value(row, ["agreeableness"], "medium"),
                "neuroticism": _get_row_value(row, ["neuroticism"], "medium")
            }

            user = {
                "id": str(_get_row_value(row, ["user_id", "id"], str(len(users)))),
                "name": _get_row_value(row, ["username", "name"], f"user_{len(users)}"),
                "description": _get_row_value(row, ["description", "bio"], ""),
                "education": _get_row_value(row, ["education"], "unknown"),
                "traits": traits,
                "policy_opinion": _get_row_value(row, ["policy_opinion"], ""),
                "party_affiliation": _get_row_value(row, ["party_affiliation"], "independent"),
                "ideology_score": _to_float(_get_row_value(row, ["ideology_score"], 0.0), 0.0),
                "issue_interest": _get_row_value(row, ["issue_interest"], "medium"),
            }
            users.append(user)

        if not users:
            raise ValueError("No valid user rows were loaded from file")

        print("User data loaded successfully")
        return users
    except Exception as e:
        print(f"Failed to load user data: {e}")
        raise RuntimeError(f"A valid user data file with enough rows is required. Error: {e}")
