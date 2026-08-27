#!/usr/bin/env python3
"""Read-only Google Cloud billing inspection for the KikiFast project.

This uses Cloud Billing, Cloud Billing Budgets, and (when configured in this
project) BigQuery billing-export APIs. A service-account key does not itself
contain spend or trial-credit information.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import quote

import requests
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


BILLING = "https://cloudbilling.googleapis.com/v1"
BUDGETS = "https://billingbudgets.googleapis.com/v1"
BIGQUERY = "https://bigquery.googleapis.com/bigquery/v2"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def get(session, url, params=None):
    response = session.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}: {response.text[:500]}"
    return response.json(), None


def money(value, units=0, nanos=0):
    return Decimal(str(units)) + Decimal(str(nanos)) / Decimal(1_000_000_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="example-project-347dc1f51500.json")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    with open(args.key, encoding="utf-8") as handle:
        key = json.load(handle)
    project = args.project or key["project_id"]
    creds = service_account.Credentials.from_service_account_info(key, scopes=[SCOPE])
    session = AuthorizedSession(creds)

    result = {"project": project, "checked_at": datetime.now(timezone.utc).isoformat()}

    info, error = get(session, f"{BILLING}/projects/{quote(project, safe='')}/billingInfo")
    if error:
        result["billing_info_error"] = error
        print(json.dumps(result, indent=2))
        return 2
    result["billing_info"] = info
    account = info.get("billingAccountName")
    if not account:
        result["message"] = "No billing account is attached to this project."
        print(json.dumps(result, indent=2))
        return 0

    account_data, error = get(session, f"{BILLING}/{account}")
    result["billing_account"] = account_data if not error else {"error": error}

    budgets, error = get(session, f"{BUDGETS}/{quote(account, safe='/')}/budgets")
    result["budgets"] = budgets if not error else {"error": error}

    # Billing export is the authoritative source for historical SKU charges.
    # This checks datasets in the project; export may instead live in another
    # BigQuery project, in which case the console/export project is required.
    datasets, error = get(session, f"{BIGQUERY}/projects/{quote(project, safe='')}/datasets")
    exports = []
    if datasets:
        for dataset in datasets.get("datasets", []):
            dataset_id = dataset["datasetReference"]["datasetId"]
            tables, table_error = get(
                session,
                f"{BIGQUERY}/projects/{quote(project, safe='')}/datasets/{quote(dataset_id, safe='')}/tables",
            )
            if table_error:
                continue
            for table in (tables or {}).get("tables", []):
                table_id = table["tableReference"]["tableId"]
                if table_id.startswith("gcp_billing_export_v1_") or "billing" in table_id.lower():
                    exports.append({"dataset": dataset_id, "table": table_id})
    result["billing_export_tables"] = exports
    if error:
        result["bigquery_error"] = error
    result["interpretation"] = {
        "historical_spend": "Requires a readable BigQuery billing-export table; no spend is present in the service-account JSON.",
        "trial_credit_balance": "Cloud Billing REST does not expose the promotional-credit balance. Check Billing > Credits in the Google Cloud Console.",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
